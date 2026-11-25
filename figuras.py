"""Figuras de los resultados (etiquetas en español).

No reentrena WavLM: usa los vectores en caché (feats_*.npz) y los LFCC en caché
(lfcc_*.npz). Las puntuaciones se guardan en puntuaciones.npz.
Uso:  .venv\\Scripts\\python.exe figuras.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
from matplotlib.patches import FancyBboxPatch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from comun import SR, cargar_audio
from entrenar import features, umbral_eer
from experimentos import SPLITS, archivos, generador, hablante

FIG = Path("figuras")
FIG.mkdir(exist_ok=True)

# Paleta nueva
PETROLEO, CORAL, MOSTAZA, VERDE, PIZARRA, CIELO, CIRUELA = "#0E4D64", "#E4572E", "#F3A712", "#5B9A8B", "#3D405B", "#3A8FB7", "#8E6C8A"
FONDO = "#FAF7F2"
NOMBRE_GEN = {"natural": "Real", "elevenlabs": "ElevenLabs", "f5-tts": "F5-TTS", "xtts-v2": "XTTS-v2", "your-tts": "YourTTS"}
COLOR_GEN = {"natural": VERDE, "elevenlabs": CIELO, "f5-tts": CORAL, "xtts-v2": MOSTAZA, "your-tts": CIRUELA}
ORDEN = ["natural", "elevenlabs", "f5-tts", "xtts-v2", "your-tts"]
NOM_SPLIT = {"train": "Entrenamiento", "val": "Validación", "test": "Prueba"}
COL_MOD = {"LFCC + GMM": CORAL, "WavLM + RL": PETROLEO}

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


def puntuaciones():
    """Puntuaciones de validación y prueba de ambos modelos (reproducibles con un hilo)."""
    cache = Path("puntuaciones.npz")
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    X, y = {}, {}
    for s in SPLITS:
        X[s], y[s] = features(s)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X["train"], y["train"])
    L = {s: list(np.load(f"lfcc_{s}.npz", allow_pickle=True)["F"]) for s in SPLITS}
    with threadpool_limits(1):
        rng = np.random.default_rng(0)
        g = {}
        for c in (0, 1):
            fr = np.concatenate([f for f, yy in zip(L["train"], y["train"]) if yy == c])
            fr = fr[rng.choice(len(fr), min(len(fr), 100_000), replace=False)]
            g[c] = GaussianMixture(32, covariance_type="diag", max_iter=200, random_state=0).fit(fr)
        out = {}
        for s in ("val", "test"):
            out[f"wavlm_{s}"] = clf.predict_proba(X[s])[:, 1]
            out[f"gmm_{s}"] = np.array([g[1].score(f) - g[0].score(f) for f in L[s]])
            out[f"y_{s}"] = y[s]
    out["pca_test"] = PCA(2, random_state=0).fit(X["train"]).transform(X["test"])
    np.savez(cache, **out)
    return out


def main():
    R = json.loads(Path("resultados.json").read_text(encoding="utf-8"))
    GMM, WAV = R["modelos"]
    P = puntuaciones()
    yv, yt = P["y_val"], P["y_test"]
    gens_t = [generador(p) for p in archivos("test")]
    # Las puntuaciones recalculadas deben reproducir resultados.json
    assert abs(umbral_eer(yt, P["gmm_test"])[1] - GMM["eer_test"]) < 1e-9, "El GMM no reproduce resultados.json"
    assert abs(umbral_eer(yt, P["wavlm_test"])[1] - WAV["eer_test"]) < 1e-9

    # 1. Datos por partición y origen
    fig, ax = plt.subplots(figsize=(8, 3.4))
    base = np.zeros(3)
    for gname in ORDEN:
        v = np.array([R["datos"][s]["por_generador"].get(gname, 0) for s in SPLITS])
        ax.barh([NOM_SPLIT[s] for s in SPLITS], v, left=base, color=COLOR_GEN[gname], label=NOMBRE_GEN[gname])
        base += v
    for i, s in enumerate(SPLITS):
        ax.text(base[i] + 15, i, f"{int(base[i]):,}".replace(",", " "), va="center", fontsize=13)
    ax.invert_yaxis(); ax.set_xlabel("Número de audios"); ax.set_xlim(0, 1800)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=12)
    guardar(fig, "fig_datos")

    # 2. Onda y mel-espectrograma: real frente a sintética del mismo hablante
    real, sint = (Path("datos/test/real") / R["ejemplo_audios"][0], Path("datos/test/sintetico") / R["ejemplo_audios"][1])
    mel_tf = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=256, n_mels=80)
    fig, axs = plt.subplots(2, 2, figsize=(10, 5), sharex="col")
    for j, (p, tit, col) in enumerate([(real, "Voz real", VERDE), (sint, "Voz sintética (ElevenLabs)", CORAL)]):
        w = cargar_audio(p)
        axs[0, j].plot(np.arange(len(w)) / SR, w.numpy(), color=col, lw=0.4)
        axs[0, j].set_title(f"{tit} · hablante {hablante(p)}")
        axs[1, j].imshow(torch.log(mel_tf(w) + 1e-6).numpy(), origin="lower", aspect="auto", extent=[0, 4, 0, 80], cmap="magma")
        axs[1, j].set_xlabel("Tiempo (s)")
    axs[0, 0].set_ylabel("Amplitud"); axs[1, 0].set_ylabel("Banda Mel")
    guardar(fig, "fig_onda_mel")

    # 3. Filtros lineales frente a Mel
    def tri(centros, f):
        out = []
        for a, b, c in zip(centros[:-2], centros[1:-1], centros[2:]):
            out.append(np.clip(np.minimum((f - a) / (b - a), (c - f) / (c - b)), 0, 1))
        return out
    f = np.linspace(0, 8000, 2000)
    n = 16
    lin = np.linspace(0, 8000, n + 2)
    mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    melc = imel(np.linspace(0, mel(8000), n + 2))
    fig, axs = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    for ax, cs, tit, col in [(axs[0], lin, "Escala lineal (LFCC): mismo detalle en todo el espectro", CORAL),
                             (axs[1], melc, "Escala Mel (MFCC): mucho detalle en graves, poco en agudos", PETROLEO)]:
        for k, t in enumerate(tri(cs, f)):
            ax.fill_between(f / 1000, t, color=col, alpha=.18 + .12 * (k % 2)); ax.plot(f / 1000, t, color=col, lw=1)
        ax.set_title(tit, loc="left", fontsize=14); ax.set_yticks([]); ax.axvspan(4, 8, color=MOSTAZA, alpha=.12)
        ax.set_ylim(0, 1.3)
    axs[1].set_xlabel("Frecuencia (kHz)")
    axs[0].text(6, 1.1, "agudos: donde suelen quedar artefactos", ha="center", fontsize=12, color="#A36F00")
    guardar(fig, "fig_filtros")

    # 4. Idea del GMM (didáctica, 1 dimensión)
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

    # 5. FAR y FRR frente al umbral (validación, WavLM)
    s = P["wavlm_val"]
    th = np.linspace(0, 1, 501)
    far = [(s[yv == 1] < t).mean() * 100 for t in th]   # sintética aceptada como real
    frr = [(s[yv == 0] >= t).mean() * 100 for t in th]  # real rechazada como sintética
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(th, far, color=CORAL, lw=2.5, label="Sintéticas aceptadas como reales (FAR)")
    ax.plot(th, frr, color=VERDE, lw=2.5, label="Reales rechazadas (FRR)")
    u = WAV["umbral"]
    ax.axvline(u, color=PIZARRA, ls="--", lw=1.3)
    ax.plot([u], [WAV["eer_val"] * 100], "o", color=PETROLEO, ms=11, zorder=5)
    ax.annotate(f"EER = {WAV['eer_val']*100:.1f} %\numbral = {u:.2f}", xy=(u, WAV["eer_val"] * 100), xytext=(u + .12, 45),
                fontsize=13, color=PETROLEO, weight="bold", arrowprops=dict(arrowstyle="->", color=PETROLEO))
    ax.set_xlabel("Umbral sobre la probabilidad de voz sintética"); ax.set_ylabel("Error (%)"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=2, fontsize=12)
    guardar(fig, "fig_eer")

    # 6. EER con intervalo de confianza
    fig, ax = plt.subplots(figsize=(8, 2.8))
    for i, m in enumerate([GMM, WAV]):
        e, (lo, hi) = m["eer_test"] * 100, [v * 100 for v in m["eer_test_ic95"]]
        ax.barh(i, e, color=COL_MOD[m["modelo"]], height=.55)
        ax.errorbar(e, i, xerr=[[e - lo], [hi - e]], fmt="none", ecolor=PIZARRA, capsize=8, lw=2)
        ax.text(hi + .25, i, f"{e:.2f} %  (IC 95 %: {lo:.1f} – {hi:.1f})", va="center", fontsize=13)
    ax.set_yticks([0, 1], ["LFCC + GMM\n(referencia)", "WavLM + RL\n(principal)"]); ax.invert_yaxis()
    ax.set_xlim(0, 13); ax.set_xlabel("EER en prueba (%) · menor es mejor")
    ax.axvspan(WAV["eer_test_ic95"][0] * 100, GMM["eer_test_ic95"][1] * 100, color=MOSTAZA, alpha=.18)
    ax.set_ylim(1.5, -0.75)
    ax.text((WAV["eer_test_ic95"][0] + GMM["eer_test_ic95"][1]) * 50, -0.5, "intervalos que se solapan", ha="center", fontsize=12, color="#A36F00")
    guardar(fig, "fig_eer_ic")

    # 7. ROC
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for m, key in [(GMM, "gmm_test"), (WAV, "wavlm_test")]:
        fpr, tpr, _ = roc_curve(yt, P[key])
        ax.plot(fpr * 100, tpr * 100, color=COL_MOD[m["modelo"]], lw=2.5, label=f"{m['modelo']} · AUC {m['auc_test']:.3f}")
    ax.plot([0, 100], [100, 0], ls=":", color=PIZARRA, lw=1, label="Puntos donde FAR = FRR")
    ax.plot([0, 100], [0, 100], ls="--", color="#BBBBBB", lw=1, label="Azar")
    ax.set_xlabel("Reales marcadas como sintéticas (%)"); ax.set_ylabel("Sintéticas detectadas (%)")
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    guardar(fig, "fig_roc")

    # 8. Histogramas
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, m, key, xl in [(axs[0], GMM, "gmm_test", "Razón de log-verosimilitud"), (axs[1], WAV, "wavlm_test", "Probabilidad de voz sintética")]:
        sc = P[key]; bins = np.linspace(sc.min(), sc.max(), 40)
        ax.hist(sc[yt == 0], bins, color=VERDE, alpha=.8, density=True, label="Real")
        ax.hist(sc[yt == 1], bins, color=CORAL, alpha=.6, density=True, label="Sintética")
        ax.axvline(m["umbral"], color=PETROLEO, ls="--", lw=2, label="Umbral")
        ax.set_title(m["modelo"]); ax.set_xlabel(xl); ax.set_yticks([]); ax.legend(frameon=False, fontsize=11)
    guardar(fig, "fig_histogramas")

    # 9. Matrices de confusión
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, m in zip(axs, [GMM, WAV]):
        cm = np.array(m["confusion"])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="GnBu", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                fr = cm[i, j] / cm[i].sum()
                ax.text(j, i, f"{cm[i, j]}\n({fr*100:.1f} %)", ha="center", va="center", fontsize=13,
                        color="white" if fr > .5 else PIZARRA)
        ax.set_xticks([0, 1], ["Real", "Sintética"]); ax.set_yticks([0, 1], ["Real", "Sintética"])
        ax.set_xlabel("Predicción"); ax.set_title(m["modelo"])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axs[0].set_ylabel("Etiqueta verdadera")
    guardar(fig, "fig_confusion")

    # 10. Acierto por origen
    fig, ax = plt.subplots(figsize=(9, 3.8))
    xs = np.arange(len(ORDEN)); w = .38
    for k, m in enumerate([GMM, WAV]):
        v = [m["por_generador"][gname]["acierto"] * 100 for gname in ORDEN]
        b = ax.bar(xs + (k - .5) * w, v, w, color=COL_MOD[m["modelo"]], label=m["modelo"])
        ax.bar_label(b, fmt="%.0f", fontsize=12)
    ax.set_xticks(xs, [NOMBRE_GEN[gname] for gname in ORDEN]); ax.set_ylabel("Acierto en prueba (%)"); ax.set_ylim(0, 112)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(.5, 1.0))
    guardar(fig, "fig_generador")

    # 11. Control de frecuencia de muestreo
    cs = R["control_sr_reales_test"]
    grupos = ["Reales\n16 kHz", "Reales\n48 kHz", "YourTTS\n16 kHz (sintética)"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    xs = np.arange(3)
    for k, m in enumerate([GMM, WAV]):
        v = [cs[m["modelo"]]["16000"]["acierto"] * 100, cs[m["modelo"]]["48000"]["acierto"] * 100, m["por_generador"]["your-tts"]["acierto"] * 100]
        b = ax.bar(xs + (k - .5) * .38, v, .38, color=COL_MOD[m["modelo"]], label=m["modelo"])
        ax.bar_label(b, fmt="%.0f", fontsize=12)
    ax.set_xticks(xs, grupos); ax.set_ylabel("Acierto (%)"); ax.set_ylim(0, 115)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(.5, 1.0))
    guardar(fig, "fig_muestreo")

    # 12. PCA
    Z = P["pca_test"]
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for gname in ORDEN:
        msk = np.array([x == gname for x in gens_t])
        ax.scatter(Z[msk, 0], Z[msk, 1], s=14, alpha=.7, color=COLOR_GEN[gname], label=NOMBRE_GEN[gname])
    ax.set_xlabel("Componente principal 1"); ax.set_ylabel("Componente principal 2")
    ax.legend(frameon=False, markerscale=2)
    guardar(fig, "fig_pca")

    # 13. Flujo del detector
    fig, ax = plt.subplots(figsize=(10, 2.8)); ax.axis("off"); ax.set_xlim(-0.1, 10.1); ax.set_ylim(-0.25, 2.8)
    W_, H_ = 1.6, 0.72
    cajas = [(0.0, 1.0, "Audio", PIZARRA), (2.1, 1.0, "Preproceso\n16 kHz · mono · 4 s", PIZARRA),
             (4.2, 1.85, "WavLM congelado\ncapa 6 → promedio [768]", PETROLEO), (6.3, 1.85, "Regresión logística\n(pesos de clase)", PETROLEO),
             (4.2, 0.15, "LFCC 20 + Δ + ΔΔ\n→ [T, 60]", CORAL), (6.3, 0.15, "GMM real / GMM sint.\n32 componentes", CORAL),
             (8.4, 1.0, "Puntuación ≥ umbral\n→ real o sintética", VERDE)]
    for x0, y0, txt, col in cajas:
        ax.add_patch(FancyBboxPatch((x0, y0), W_, H_, boxstyle="round,pad=0.05", fc=col, ec="none"))
        ax.text(x0 + W_ / 2, y0 + H_ / 2, txt, ha="center", va="center", color="white", fontsize=11)
    c = lambda x0, y0, lado: (x0 + (W_ + 0.08 if lado == "d" else -0.08), y0 + H_ / 2)
    for a, b in [(c(0, 1, "d"), c(2.1, 1, "i")), (c(2.1, 1, "d"), c(4.2, 1.85, "i")), (c(2.1, 1, "d"), c(4.2, 0.15, "i")),
                 (c(4.2, 1.85, "d"), c(6.3, 1.85, "i")), (c(4.2, 0.15, "d"), c(6.3, 0.15, "i")),
                 (c(6.3, 1.85, "d"), c(8.4, 1, "i")), (c(6.3, 0.15, "d"), c(8.4, 1, "i"))]:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=PIZARRA, lw=1.4))
    ax.text(5.95, 2.75, "Modelo principal", ha="center", color=PETROLEO, fontsize=13, weight="bold")
    ax.text(5.95, -0.08, "Modelo de referencia", ha="center", color=CORAL, fontsize=13, weight="bold", va="top")
    guardar(fig, "fig_pipeline")

    print("Figuras listas:", sorted(p.stem for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
