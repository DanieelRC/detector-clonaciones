"""Figuras de los resultados (etiquetas en español).

Lee resultados.json, puntuaciones.csv, el manifiesto y wavlm.npz; no vuelve a entrenar nada.
Uso:  python figuras.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from matplotlib.patches import FancyBboxPatch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve

from comun import SR, cargar_audio, umbral_eer

FIG = Path("figuras")
PETROLEO, CORAL, MOSTAZA, VERDE, PIZARRA, CIELO, CIRUELA = "#0E4D64", "#E4572E", "#F3A712", "#5B9A8B", "#3D405B", "#3A8FB7", "#8E6C8A"
NOMBRE_GEN = {"natural": "Real", "elevenlabs": "ElevenLabs", "f5-tts": "F5-TTS", "xtts-v2": "XTTS-v2", "your-tts": "YourTTS"}
COLOR_GEN = {"natural": VERDE, "elevenlabs": CIELO, "f5-tts": CORAL, "xtts-v2": MOSTAZA, "your-tts": CIRUELA}
ORDEN = ["natural", "elevenlabs", "f5-tts", "xtts-v2", "your-tts"]
SPLITS = ["train", "val", "test"]
NOM_SPLIT = {"train": "Entrenamiento", "val": "Validación", "test": "Prueba"}
CONFIGS = ["LFCC + GMM", "LFCC + MLP", "LFCC + LPC + MLP", "WavLM + RL", "WavLM + MLP"]
COL = dict(zip(CONFIGS, [CORAL, MOSTAZA, CIRUELA, PETROLEO, CIELO]))

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Calibri", "Segoe UI", "DejaVu Sans"],
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "legend.fontsize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": PIZARRA,
    "axes.labelcolor": PIZARRA, "xtick.color": PIZARRA, "ytick.color": PIZARRA, "text.color": PIZARRA,
    "savefig.facecolor": "white",
})


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(FIG / f"{nombre}.pdf")
    fig.savefig(FIG / f"{nombre}.png", dpi=220)
    plt.close(fig)


def leer_puntuaciones():
    """{(configuración, semilla): {"val": array, "test": array}}, y etiquetas y generadores por partición."""
    P = defaultdict(lambda: {"val": [], "test": []})
    meta = {"val": [], "test": []}
    with open("puntuaciones.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            P[(r["configuracion"], r["semilla"])][r["particion"]].append(float(r["puntuacion"]))
            if r["configuracion"] == "LFCC + GMM":
                meta[r["particion"]].append((int(r["etiqueta"]), r["archivo"]))
    P = {k: {p: np.array(v) for p, v in d.items()} for k, d in P.items()}
    y = {p: np.array([e for e, _ in meta[p]]) for p in meta}
    gens = {p: [n.split("_")[1] for _, n in meta[p]] for p in meta}
    return P, y, gens


def principal(P, cfg):
    """Corrida que se muestra en figuras de una sola puntuación: semilla 0 para los MLP."""
    return P[(cfg, "0")] if (cfg, "0") in P else P[(cfg, "")]


def main():
    FIG.mkdir(exist_ok=True)
    R = json.loads(Path("resultados.json").read_text(encoding="utf-8"))
    C = R["configuraciones"]
    P, y, gens = leer_puntuaciones()
    yv, yt = y["val"], y["test"]
    # Las puntuaciones guardadas reproducen las cifras de resultados.json
    for cfg in CONFIGS:
        for c in C[cfg]["corridas"]:
            sem = "" if c["semilla"] is None else str(c["semilla"])
            assert abs(umbral_eer(yt, P[(cfg, sem)]["test"])[1] - c["eer_test"]) < 1e-9, cfg

    def corrida0(cfg):
        return next(c for c in C[cfg]["corridas"] if c["semilla"] in (None, 0))

    def acierto_medio(cfg, clave, grupo):
        cs = C[cfg]["corridas"]
        if clave == "por_generador":
            return 100 * np.mean([c[clave][grupo]["acierto"] for c in cs])
        return 100 * np.mean([1 - c[clave][grupo]["errores"] / c[clave][grupo]["n"] for c in cs])

    # 1. Datos por partición y origen, con número de hablantes
    fig, ax = plt.subplots(figsize=(9, 3.6))
    base = np.zeros(3)
    for g in ORDEN:
        v = np.array([R["datos"][s]["por_generador"].get(g, 0) for s in SPLITS])
        ax.barh([NOM_SPLIT[s] for s in SPLITS], v, left=base, color=COLOR_GEN[g], label=NOMBRE_GEN[g])
        base += v
    for i, s in enumerate(SPLITS):
        ax.text(base[i] + 15, i, f"{int(base[i]):,} audios · {len(R['datos'][s]['hablantes'])} hablantes".replace(",", " "),
                va="center", fontsize=13)
    ax.invert_yaxis(); ax.set_xlabel("Número de audios"); ax.set_xlim(0, 2300)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=12)
    guardar(fig, "fig_datos")

    # 2. Onda y mel-espectrograma: real frente a sintética del mismo hablante, tal como entran al detector
    real, sint = [Path("datos/audios") / n for n in R["ejemplo_audios"]]
    mel_tf = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=256, n_mels=80)
    fig, axs = plt.subplots(2, 2, figsize=(10, 5), sharex="col")
    for j, (p, tit, col) in enumerate([(real, "Voz real", VERDE), (sint, "Voz sintética (ElevenLabs)", CORAL)]):
        w = cargar_audio(p)
        dur = len(w) / SR
        axs[0, j].plot(np.arange(len(w)) / SR, w.numpy(), color=col, lw=0.4)
        axs[0, j].set_title(f"{tit} · hablante {p.name.split('_')[3]}"); axs[0, j].set_ylim(-1, 1)
        axs[1, j].imshow(torch.log(mel_tf(w) + 1e-6).numpy(), origin="lower", aspect="auto", extent=[0, dur, 0, 80], cmap="magma")
        axs[1, j].set_xlabel("Tiempo (s)")
    axs[0, 0].set_ylabel("Amplitud"); axs[1, 0].set_ylabel("Banda Mel")
    guardar(fig, "fig_onda_mel")

    # 3. Filtros lineales frente a Mel (esquemático)
    def tri(centros, f):
        return [np.clip(np.minimum((f - a) / (b - a), (c - f) / (c - b)), 0, 1) for a, b, c in zip(centros[:-2], centros[1:-1], centros[2:])]
    f = np.linspace(0, 8000, 2000)
    lin = np.linspace(0, 8000, 18)
    mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    melc = imel(np.linspace(0, mel(8000), 18))
    fig, axs = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    for ax, cs, tit, col in [(axs[0], lin, "Escala lineal (LFCC): mismo detalle en todo el espectro", CORAL),
                             (axs[1], melc, "Escala Mel (MFCC): mucho detalle en graves, poco en agudos", PETROLEO)]:
        for k, t in enumerate(tri(cs, f)):
            ax.fill_between(f / 1000, t, color=col, alpha=.18 + .12 * (k % 2)); ax.plot(f / 1000, t, color=col, lw=1)
        ax.set_title(tit, loc="left", fontsize=14); ax.set_yticks([]); ax.axvspan(4, 8, color=MOSTAZA, alpha=.12)
        ax.set_ylim(0, 1.3)
    axs[1].set_xlabel("Frecuencia (kHz)")
    guardar(fig, "fig_filtros")

    # 4. Idea del GMM (didáctica, una dimensión)
    x = np.linspace(-6, 8, 600)
    nd = lambda x, m, s: np.exp(-(x - m) ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi))
    pr = .5 * nd(x, -1.5, 1) + .3 * nd(x, 0.5, .8) + .2 * nd(x, -3.5, .9)
    ps = .4 * nd(x, 2.5, .9) + .4 * nd(x, 4.2, 1.1) + .2 * nd(x, 0.8, .7)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.fill_between(x, pr, color=VERDE, alpha=.35); ax.plot(x, pr, color=VERDE, lw=2, label="GMM de voz real")
    ax.fill_between(x, ps, color=CORAL, alpha=.30); ax.plot(x, ps, color=CORAL, lw=2, label="GMM de voz sintética")
    x0 = 2.0
    ax.axvline(x0, color=PIZARRA, ls="--", lw=1.4)
    ax.plot([x0], [np.interp(x0, x, pr)], "o", color=VERDE, ms=9); ax.plot([x0], [np.interp(x0, x, ps)], "o", color=CORAL, ms=9)
    ax.annotate("cuadro nuevo:\nmás probable bajo el\nGMM sintético → puntuación > 0", xy=(x0, np.interp(x0, x, ps)), xytext=(4.6, .3),
                fontsize=12, arrowprops=dict(arrowstyle="->", color=PIZARRA))
    ax.set_xlabel("Un coeficiente LFCC (ilustrativo)"); ax.set_ylabel("Densidad"); ax.set_yticks([])
    ax.legend(frameon=False, loc="upper left")
    guardar(fig, "fig_gmm_idea")

    # 5. FAR y FRR frente al umbral (validación, WavLM + RL)
    RLc = corrida0("WavLM + RL")
    s = principal(P, "WavLM + RL")["val"]
    th = np.linspace(0, 1, 501)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(th, [(s[yv == 1] < t).mean() * 100 for t in th], color=CORAL, lw=2.5, label="Sintéticas aceptadas como reales (FAR)")
    ax.plot(th, [(s[yv == 0] >= t).mean() * 100 for t in th], color=VERDE, lw=2.5, label="Reales rechazadas (FRR)")
    u = RLc["umbral"]
    ax.axvline(u, color=PIZARRA, ls="--", lw=1.3)
    ax.plot([u], [RLc["eer_val"] * 100], "o", color=PETROLEO, ms=11, zorder=5)
    ax.annotate(f"EER = {RLc['eer_val'] * 100:.1f} %\numbral = {u:.2f}", xy=(u, RLc["eer_val"] * 100), xytext=(min(u + .12, .75), 45),
                fontsize=13, color=PETROLEO, weight="bold", arrowprops=dict(arrowstyle="->", color=PETROLEO))
    ax.set_xlabel("Umbral sobre la probabilidad de voz sintética"); ax.set_ylabel("Error (%)"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=2, fontsize=12)
    guardar(fig, "fig_eer")

    # 6. EER de prueba de las cinco configuraciones con IC 95 % (remuestreo por hablante) y valores por semilla
    fig, ax = plt.subplots(figsize=(9.5, 4))
    for i, cfg in enumerate(CONFIGS):
        e = C[cfg]["eer_test_media"] * 100
        lo, hi = [v * 100 for v in C[cfg]["eer_test_ic95"]]
        ax.barh(i, e, color=COL[cfg], height=.6)
        ax.errorbar(e, i, xerr=[[e - lo], [hi - e]], fmt="none", ecolor=PIZARRA, capsize=6, lw=1.8)
        if len(C[cfg]["corridas"]) > 1:
            ax.scatter([c["eer_test"] * 100 for c in C[cfg]["corridas"]], [i] * len(C[cfg]["corridas"]), s=22, color="white", edgecolor=PIZARRA, zorder=4)
        ax.text(hi + .3, i, f"{e:.2f} %", va="center", fontsize=13)
    ax.set_yticks(range(len(CONFIGS)), CONFIGS); ax.invert_yaxis()
    ax.set_xlabel("EER en prueba (%) · menor es mejor")
    ax.set_xlim(0, max(C[c]["eer_test_ic95"][1] for c in CONFIGS) * 100 + 3)
    guardar(fig, "fig_eer_ic")

    # 7. ROC (semilla 0 para los MLP)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for cfg in CONFIGS:
        fpr, tpr, _ = roc_curve(yt, principal(P, cfg)["test"])
        ax.plot(fpr * 100, tpr * 100, color=COL[cfg], lw=2.2, label=f"{cfg} · AUC {corrida0(cfg)['auc_test']:.3f}")
    ax.plot([0, 100], [100, 0], ls=":", color=PIZARRA, lw=1, label="FAR = FRR")
    ax.set_xlabel("Reales marcadas como sintéticas (%)"); ax.set_ylabel("Sintéticas detectadas (%)")
    ax.legend(loc="lower right", frameon=False, fontsize=10.5)
    guardar(fig, "fig_roc")

    # 8. Histogramas de puntuaciones de prueba (cinco configuraciones y el control)
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.4))
    paneles = [(cfg, principal(P, cfg)["test"], corrida0(cfg)["umbral"]) for cfg in CONFIGS]
    paneles.append(("Control (duración, silencio, frecuencia)", P[("Control", "")]["test"], R["control"]["umbral"]))
    for ax, (nom, sc, u) in zip(axs.flat, paneles):
        bins = np.linspace(*np.percentile(sc, [0.5, 99.5]), 40)  # se omiten los valores extremos para ver el solapamiento
        ax.hist(sc[yt == 0], bins, color=VERDE, alpha=.8, density=True, label="Real")
        ax.hist(sc[yt == 1], bins, color=CORAL, alpha=.6, density=True, label="Sintética")
        ax.axvline(u, color=PETROLEO, ls="--", lw=2, label="Umbral")
        ax.set_title(nom, fontsize=13); ax.set_yticks([])
    axs[0, 0].legend(frameon=False, fontsize=11)
    guardar(fig, "fig_histogramas")

    # 9. Matrices de confusión con el umbral de validación (semilla 0 para los MLP)
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.9))
    for ax, cfg in zip(axs, CONFIGS):
        cm = np.array(corrida0(cfg)["confusion"])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="GnBu", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                fr = cm[i, j] / cm[i].sum()
                ax.text(j, i, f"{cm[i, j]}\n({fr * 100:.1f} %)", ha="center", va="center", fontsize=12, color="white" if fr > .5 else PIZARRA)
        ax.set_xticks([0, 1], ["Real", "Sintética"]); ax.set_yticks([0, 1], ["Real", "Sintética"])
        ax.set_xlabel("Predicción"); ax.set_title(cfg, fontsize=13)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axs[0].set_ylabel("Etiqueta verdadera")
    guardar(fig, "fig_confusion")

    # 10. Acierto por origen (media de semillas)
    fig, ax = plt.subplots(figsize=(11, 4))
    xs = np.arange(len(ORDEN)); w = .16
    for k, cfg in enumerate(CONFIGS):
        v = [acierto_medio(cfg, "por_generador", g) for g in ORDEN]
        b = ax.bar(xs + (k - 2) * w, v, w, color=COL[cfg], label=cfg)
        ax.bar_label(b, fmt="%.0f", fontsize=9.5)
    ax.set_xticks(xs, [NOMBRE_GEN[g] for g in ORDEN]); ax.set_ylabel("Acierto en prueba (%)"); ax.set_ylim(0, 112)
    ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(.5, 1.0), fontsize=11)
    guardar(fig, "fig_generador")

    # 11. Frecuencia de muestreo original: reales a 16 y 48 kHz y YourTTS (16 kHz)
    grupos = [("natural 16000", "Reales\n16 kHz"), ("natural 48000", "Reales\n48 kHz"), ("your-tts 16000", "YourTTS\n16 kHz (sintética)")]
    fig, ax = plt.subplots(figsize=(10, 3.9))
    xs = np.arange(len(grupos))
    for k, cfg in enumerate(CONFIGS):
        b = ax.bar(xs + (k - 2) * w, [acierto_medio(cfg, "por_origen_sr", g) for g, _ in grupos], w, color=COL[cfg], label=cfg)
        ax.bar_label(b, fmt="%.0f", fontsize=9.5)
    n = C["LFCC + GMM"]["corridas"][0]["por_origen_sr"]
    ax.set_xticks(xs, [f"{t}\nn = {n[g]['n']}" for g, t in grupos]); ax.set_ylabel("Acierto (%)"); ax.set_ylim(0, 115)
    ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(.5, 1.0), fontsize=11)
    guardar(fig, "fig_muestreo")

    # 12. PCA de las representaciones de WavLM (ajustada con entrenamiento)
    W = np.load("wavlm.npz")
    with open("datos/manifiesto.csv", encoding="utf-8") as fh:
        man = [r for r in csv.DictReader(fh) if not r["exclusion"]]
    pm = np.array([r["particion"] for r in man]); gm = np.array([r["generador"] for r in man])
    Zp = PCA(2, random_state=0).fit(W["X"][pm == "train"]).transform(W["X"][pm == "test"])
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for g in ORDEN:
        m = gm[pm == "test"] == g
        ax.scatter(Zp[m, 0], Zp[m, 1], s=14, alpha=.7, color=COLOR_GEN[g], label=NOMBRE_GEN[g])
    ax.set_xlabel("Componente principal 1"); ax.set_ylabel("Componente principal 2")
    ax.legend(frameon=False, markerscale=2)
    guardar(fig, "fig_pca")

    # 13. Diferencias de EER entre pares de configuraciones (por semilla, media e IC 95 %)
    comp = R["comparaciones"]
    etiquetas = {"LPC": "Aporte de LPC\n(LFCC + LPC + MLP − LFCC + MLP)", "MLP frente a RL": "MLP frente a RL\n(WavLM + MLP − WavLM + RL)",
                 "MLP acústico": "MLP acústico frente a GMM\n(LFCC + MLP − LFCC + GMM)"}
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (k, v) in enumerate(comp.items()):
        lo, hi = v["ic95_pp"]
        ax.plot([lo, hi], [i, i], color=PIZARRA, lw=3, solid_capstyle="round")
        ax.scatter(v["por_semilla_pp"], [i] * len(v["por_semilla_pp"]), s=40, color=MOSTAZA, edgecolor=PIZARRA, zorder=4)
        ax.plot([v["media_pp"]], [i], "D", color=PETROLEO, ms=10, zorder=5)
        ax.text(hi + 1.0, i, f"{v['media_pp']:+.2f} pp", va="center", fontsize=13)
    ax.axvline(0, color=CORAL, ls="--", lw=1.5)
    ax.set_yticks(range(len(comp)), [next(t for p, t in etiquetas.items() if k.startswith(p)) for k in comp], fontsize=12)
    ax.invert_yaxis(); ax.set_xlabel("Diferencia de EER en prueba (puntos porcentuales) · negativo = la segunda reduce la EER")
    guardar(fig, "fig_comparaciones")

    # 14. Control: EER de un clasificador que solo ve duración, silencio y frecuencia original
    fig, ax = plt.subplots(figsize=(9, 3.8))
    nombres = ["Control"] + CONFIGS
    vals = [R["control"]["eer_test"] * 100] + [C[c]["eer_test_media"] * 100 for c in CONFIGS]
    b = ax.bar(range(len(nombres)), vals, color=[PIZARRA] + [COL[c] for c in CONFIGS])
    ax.bar_label(b, fmt="%.1f %%", fontsize=12)
    ax.set_xticks(range(len(nombres)), [n.replace(" + ", "\n+ ") for n in nombres], fontsize=11)
    ax.set_ylabel("EER en prueba (%)")
    guardar(fig, "fig_control")

    # 15. Flujo del detector: cinco configuraciones
    fig, ax = plt.subplots(figsize=(12, 4.6)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(-0.05, 4.75)
    def caja(x0, y0, wd, ht, txt, col, fs=11):
        ax.add_patch(FancyBboxPatch((x0, y0), wd, ht, boxstyle="round,pad=0.04", fc=col, ec="none"))
        ax.text(x0 + wd / 2, y0 + ht / 2, txt, ha="center", va="center", color="white", fontsize=fs)
    def flecha(a, b):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=PIZARRA, lw=1.3))
    caja(0.1, 1.9, 1.6, 0.9, "Audio\n16 kHz · mono\n≤ 4 s · pico 0.95", PIZARRA, 10.5)
    caja(2.4, 3.4, 2.2, 0.8, "WavLM congelado\ncapa 6 · 768 valores", PETROLEO)
    caja(2.4, 1.9, 2.2, 0.8, "LFCC\n60 valores por ventana", CORAL)
    caja(2.4, 0.4, 2.2, 0.8, "LPC de orden 16\nventanas con energía", CIRUELA)
    caja(5.4, 1.15, 2.0, 0.8, "Resumen por audio\nmedia y desviación", MOSTAZA, 10.5)
    filas_cls = [(3.95, "Regresión logística", PETROLEO), (3.0, "MLP de WavLM", CIELO), (2.05, "GMM real y\nGMM sintético", CORAL),
                 (1.1, "MLP con LFCC (120)", MOSTAZA), (0.15, "MLP con LFCC + LPC (152)", CIRUELA)]
    for yy, txt, col in filas_cls:
        caja(8.1, yy, 2.2, 0.65, txt, col, 10.5)
        flecha((10.35, yy + .33), (10.75, 2.35))
    caja(10.8, 1.9, 1.1, 0.9, "Umbral\npor modelo\n→ real o\nsintética", VERDE, 10)
    for yy in (3.8,):
        flecha((1.75, 2.35), (2.35, yy))
    flecha((1.75, 2.35), (2.35, 2.3)); flecha((1.75, 2.35), (2.35, 0.8))
    flecha((4.65, 3.8), (8.05, 4.27)); flecha((4.65, 3.8), (8.05, 3.33))
    flecha((4.65, 2.3), (8.05, 2.38)); flecha((4.65, 2.3), (5.35, 1.6)); flecha((4.65, 0.8), (5.35, 1.4))
    flecha((7.45, 1.55), (8.05, 1.43)); flecha((7.45, 1.55), (8.05, 0.48))
    guardar(fig, "fig_pipeline")

    print("Figuras listas:", sorted(p.stem for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
