"""Experimentos: WavLM + regresión logística frente a LFCC + GMM.

Reutiliza las representaciones en caché de entrenar.py (feats_*.npz) y escribe
resultados.json y las figuras en figuras/.
"""
import json
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from comun import SR, cargar_audio, embedding_de_archivo
from entrenar import DATOS, EXT, features, umbral_eer

FIG = Path("figuras")
FIG.mkdir(exist_ok=True)
SPLITS = ["train", "val", "test"]
AZUL, NARANJA, AMBAR, AZUL2, ROSA, VERDE, GRIS = "#1F2761", "#C9711A", "#F2A541", "#3A6FD8", "#D95F8B", "#2A9D8F", "#5C6178"
COLOR_GEN = {"natural": VERDE, "elevenlabs": AZUL2, "f5-tts": NARANJA, "xtts-v2": ROSA, "your-tts": AMBAR}
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def archivos(split):
    """Mismo orden que entrenar.features: real y luego sintético, ordenados."""
    out = []
    for clase in ["real", "sintetico"]:
        out += sorted(p for p in (DATOS / split / clase).glob("*") if p.suffix.lower() in EXT)
    return out


def generador(p):
    return re.match(r"detection_(.+?)_speaker", p.name).group(1)


def hablante(p):
    return int(re.search(r"speaker_(\d+)", p.name).group(1))


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(FIG / f"{nombre}.pdf")
    fig.savefig(FIG / f"{nombre}.png", dpi=200)
    plt.close(fig)


# ---------- LFCC + GMM ----------
lfcc_tf = torchaudio.transforms.LFCC(sample_rate=SR, n_lfcc=20, speckwargs={"n_fft": 512, "win_length": 320, "hop_length": 160})


def lfcc(p):
    c = lfcc_tf(cargar_audio(p))  # [20, T]
    d = torchaudio.functional.compute_deltas(c)
    dd = torchaudio.functional.compute_deltas(d)
    return torch.cat([c, d, dd]).T.numpy()  # [T, 60]


def lfcc_split(split):
    cache = Path(f"lfcc_{split}.npz")
    if cache.exists():
        return list(np.load(cache, allow_pickle=True)["F"])
    F = [lfcc(p) for p in archivos(split)]
    np.savez(cache, F=np.array(F, dtype=object))
    return F


def fig_pipeline():
    """Diagrama del flujo de ambos modelos (cajas de 1.6 × 0.7)."""
    W, H = 1.6, 0.7
    fig, ax = plt.subplots(figsize=(10, 2.6)); ax.axis("off"); ax.set_xlim(-0.1, 10.1); ax.set_ylim(-0.2, 2.75)
    cajas = [(0.0, 1.0, "Audio\n.wav / .flac", GRIS), (2.1, 1.0, "Preproceso\n16 kHz · mono · 4 s", GRIS),
             (4.2, 1.85, "WavLM-base congelado\ncapa 6 → media [768]", AZUL), (6.3, 1.85, "Regresión logística\n(pesos de clase)", AZUL),
             (4.2, 0.15, "LFCC 20 + Δ + ΔΔ\n→ [T, 60]", NARANJA), (6.3, 0.15, "GMM real / GMM sint.\n32 comp. diagonales", NARANJA),
             (8.4, 1.0, "p(sintética) ≥ umbral\n(EER en val)\n→ etiqueta", VERDE)]
    for x0, y0, txt, col in cajas:
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x0, y0), W, H, boxstyle="round,pad=0.05", fc=col, ec="none"))
        ax.text(x0 + W / 2, y0 + H / 2, txt, ha="center", va="center", color="white", fontsize=8.3)
    c = lambda x0, y0, lado: (x0 + (W + 0.08 if lado == "d" else -0.08), y0 + H / 2)
    flechas = [(c(0, 1, "d"), c(2.1, 1, "i")), (c(2.1, 1, "d"), c(4.2, 1.85, "i")), (c(2.1, 1, "d"), c(4.2, 0.15, "i")),
               (c(4.2, 1.85, "d"), c(6.3, 1.85, "i")), (c(4.2, 0.15, "d"), c(6.3, 0.15, "i")),
               (c(6.3, 1.85, "d"), c(8.4, 1, "i")), (c(6.3, 0.15, "d"), c(8.4, 1, "i"))]
    for a, b in flechas:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=GRIS, lw=1.3))
    ax.text(5.95, 2.7, "Modelo principal", ha="center", color=AZUL, fontsize=9, weight="bold")
    ax.text(5.95, -0.05, "Baseline", ha="center", color=NARANJA, fontsize=9, weight="bold", va="top")
    guardar(fig, "fig_pipeline")


def bootstrap_eer(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            vals.append(umbral_eer(y[i], s[i])[1])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def evaluar(nombre, s_val, y_val, s_te, y_te, gens_te):
    umbral, eer_val = umbral_eer(y_val, s_val)
    _, eer_te = umbral_eer(y_te, s_te)
    pred = (s_te >= umbral).astype(int)
    por_gen = {}
    for g in sorted(set(gens_te)):
        m = np.array([x == g for x in gens_te])
        por_gen[g] = {"n": int(m.sum()), "acierto": float((pred[m] == y_te[m]).mean())}
        if g != "natural":
            mm = m | (y_te == 0)
            por_gen[g]["eer"] = float(umbral_eer(y_te[mm], s_te[mm])[1])
    return {
        "modelo": nombre, "umbral": float(umbral), "eer_val": float(eer_val), "eer_test": float(eer_te),
        "eer_test_ic95": bootstrap_eer(y_te, s_te), "auc_test": float(roc_auc_score(y_te, s_te)),
        "exactitud_test": float((pred == y_te).mean()),
        "confusion": confusion_matrix(y_te, pred).tolist(), "por_generador": por_gen,
    }


if __name__ == "__main__":
    t0 = time.time()
    rutas = {s: archivos(s) for s in SPLITS}
    gens = {s: [generador(p) for p in rutas[s]] for s in SPLITS}
    X, y = {}, {}
    for s in SPLITS:
        X[s], y[s] = features(s)
        assert len(X[s]) == len(rutas[s]), "El orden de feats_*.npz no coincide con datos/"

    # --- Modelo principal: WavLM + RL (idéntico a entrenar.py) ---
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X["train"], y["train"])
    s_wavlm = {s: clf.predict_proba(X[s])[:, 1] for s in SPLITS}
    r_wavlm = evaluar("WavLM + RL", s_wavlm["val"], y["val"], s_wavlm["test"], y["test"], gens["test"])
    print("WavLM:", r_wavlm["eer_test"], r_wavlm["auc_test"])

    # --- Baseline: LFCC + GMM ---
    print("Extrayendo LFCC...")
    L = {s: lfcc_split(s) for s in SPLITS}
    rng = np.random.default_rng(0)
    gmms = {}
    t_gmm = time.time()
    with threadpool_limits(1):  # con un solo hilo el ajuste del GMM es reproducible
        for c in (0, 1):
            frames = np.concatenate([f for f, yy in zip(L["train"], y["train"]) if yy == c])
            frames = frames[rng.choice(len(frames), min(len(frames), 100_000), replace=False)]
            gmms[c] = GaussianMixture(32, covariance_type="diag", max_iter=200, random_state=0).fit(frames)
    t_gmm = time.time() - t_gmm
    s_gmm = {s: np.array([gmms[1].score(f) - gmms[0].score(f) for f in L[s]]) for s in ["val", "test"]}
    r_gmm = evaluar("LFCC + GMM", s_gmm["val"], y["val"], s_gmm["test"], y["test"], gens["test"])
    print("GMM:", r_gmm["eer_test"], r_gmm["auc_test"])

    # --- Latencia por audio (CPU) ---
    muestra = rutas["test"][:20]
    embedding_de_archivo(muestra[0])  # calentamiento
    t = time.perf_counter(); [clf.predict_proba([embedding_de_archivo(p)]) for p in muestra]
    lat_wavlm = (time.perf_counter() - t) / len(muestra) * 1000
    t = time.perf_counter(); [gmms[1].score(lfcc(p)) - gmms[0].score(lfcc(p)) for p in muestra]
    lat_gmm = (time.perf_counter() - t) / len(muestra) * 1000

    # --- Datos ---
    datos = {s: {"real": int((y[s] == 0).sum()), "sintetico": int((y[s] == 1).sum()),
                 "por_generador": {g: gens[s].count(g) for g in sorted(set(gens[s]))}} for s in SPLITS}
    hablantes = sorted({hablante(p) for s in SPLITS for p in rutas[s]})
    import soundfile as sf
    infos = [sf.info(str(p)) for s in SPLITS for p in rutas[s]]
    durs = [i.duration for i in infos]
    res_sr = sorted({i.samplerate for i in infos})

    # Control de atajo: cada generador llega con una frecuencia de muestreo distinta.
    sr_gen = {}
    for p, i in zip([p for s in SPLITS for p in rutas[s]], infos):
        sr_gen.setdefault(generador(p), {}).setdefault(str(i.samplerate), 0)
        sr_gen[generador(p)][str(i.samplerate)] += 1
    sr_test = np.array([sf.info(str(p)).samplerate for p in rutas["test"]])
    real_test = y["test"] == 0
    control_sr = {}
    for r, s in [(r_gmm, s_gmm["test"]), (r_wavlm, s_wavlm["test"])]:
        control_sr[r["modelo"]] = {str(f): {"n": int((real_test & (sr_test == f)).sum()),
                                            "acierto": float((s[real_test & (sr_test == f)] < r["umbral"]).mean())}
                                   for f in sorted(set(sr_test[real_test]))}

    res = {
        "sr_por_generador": sr_gen, "control_sr_reales_test": control_sr,
        "datos": datos, "total": int(sum(len(rutas[s]) for s in SPLITS)), "hablantes": len(hablantes),
        "duracion_s": {"media": float(np.mean(durs)), "min": float(np.min(durs)), "max": float(np.max(durs)),
                       "frac_mayor_4s": float(np.mean(np.array(durs) > 4))},
        "frecuencias_muestreo": res_sr,
        "modelos": [r_gmm, r_wavlm],
        "latencia_ms": {"WavLM + RL": lat_wavlm, "LFCC + GMM": lat_gmm},
        "tiempo_entrenamiento_gmm_s": t_gmm,
        "versiones": {"torch": torch.__version__, "torchaudio": torchaudio.__version__},
    }
    Path("resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    # Ejemplo para las figuras: un audio real y uno sintético del mismo hablante
    real = next(p for p in rutas["test"] if generador(p) == "natural")
    sint = next(p for p in rutas["test"] if generador(p) != "natural" and hablante(p) == hablante(real))
    res["ejemplo_audios"] = [real.name, sint.name]
    Path("resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    # Figuras: figuras.py las genera a partir de resultados.json y de las cachés
    Path("puntuaciones.npz").unlink(missing_ok=True)
    import figuras
    figuras.main()

    Path("resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ["latencia_ms", "hablantes", "duracion_s"]}, indent=1))
    print(f"Listo en {time.time() - t0:.0f} s")
