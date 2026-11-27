"""Extrae las representaciones de todos los audios del manifiesto y entrena el detector desplegable.

Cachés:
  wavlm.npz      vector de 768 valores por audio (media temporal de la capa 6)
  acusticas.npz  cuadros LFCC [T, 60] (para el GMM), resúmenes de 120 y 152 valores,
                 y la proporción de ventanas de baja energía (para el control)
Modelo:
  modelo.joblib  normalización + regresión logística sobre WavLM y umbral de validación
"""
import csv
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from comun import acusticas, cargar_audio, embedding, prop_baja_energia, resumen, umbral_eer

AUDIOS = Path("datos/audios")
MANIFIESTO = Path("datos/manifiesto.csv")


def manifiesto():
    """Audios incluidos, en el orden del manifiesto."""
    with open(MANIFIESTO, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if not r["exclusion"]]


def extraer(filas):
    nombres = [r["archivo"] for r in filas]
    if not Path("wavlm.npz").exists():
        X, t0 = [], time.time()
        for i, n in enumerate(nombres, 1):
            X.append(embedding(cargar_audio(AUDIOS / n)))
            if i % 200 == 0:
                print(f"  WavLM {i}/{len(nombres)}")
        np.savez("wavlm.npz", archivos=nombres, X=np.array(X), segundos=time.time() - t0)
    if not Path("acusticas.npz").exists():
        cuadros, r120, r152, baja, t0 = [], [], [], [], time.time()
        for n in nombres:
            wav = cargar_audio(AUDIOS / n)
            lfcc, mascara, lpc = acusticas(wav)
            cuadros.append(lfcc)
            r120.append(resumen(lfcc, mascara))
            r152.append(resumen(lfcc, mascara, lpc))
            baja.append(prop_baja_energia(wav))
        np.savez("acusticas.npz", archivos=nombres, cuadros=np.array(cuadros, dtype=object),
                 lfcc=np.array(r120), lfcc_lpc=np.array(r152), baja_energia=np.array(baja), segundos=time.time() - t0)
    w, a = np.load("wavlm.npz"), np.load("acusticas.npz", allow_pickle=True)
    assert list(w["archivos"]) == nombres == list(a["archivos"]), "Las cachés no coinciden con el manifiesto"
    return w, a


if __name__ == "__main__":
    filas = manifiesto()
    w, _ = extraer(filas)
    part = np.array([r["particion"] for r in filas])
    y = np.array([int(r["etiqueta"]) for r in filas])
    X = w["X"]

    esc = StandardScaler().fit(X[part == "train"])
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(esc.transform(X[part == "train"]), y[part == "train"])
    s = clf.predict_proba(esc.transform(X))[:, 1]
    umbral, eer_val = umbral_eer(y[part == "val"], s[part == "val"])
    te = part == "test"
    _, eer_te = umbral_eer(y[te], s[te])

    print(f"Modelo: WavLM-base capa 6 + regresión logística | dimensión {X.shape[1]} | "
          f"audios: {', '.join(f'{p} {(part == p).sum()}' for p in ('train', 'val', 'test'))}")
    print(f"Umbral (validación): {umbral:.4f}  EER val: {eer_val:.2%}")
    print(f"Prueba -> EER: {eer_te:.2%}  AUC: {roc_auc_score(y[te], s[te]):.4f}  "
          f"Exactitud: {((s[te] >= umbral) == y[te]).mean():.2%}")
    joblib.dump({"escalador": esc, "clf": clf, "umbral": umbral}, "modelo.joblib")
    print("Modelo guardado en modelo.joblib")
