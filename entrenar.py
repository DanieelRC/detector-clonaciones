"""Entrena regresión logística sobre embeddings de WavLM.

Estructura esperada:
  datos/{train,val,test}/{real,sintetico}/*.wav  (también .flac/.mp3/.ogg)
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from comun import embedding_de_archivo

DATOS = Path("datos")
EXT = {".wav", ".flac", ".mp3", ".ogg"}


def features(split):
    cache = Path(f"feats_{split}.npz")
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"]
    X, y = [], []
    for etiqueta, clase in enumerate(["real", "sintetico"]):
        archivos = sorted(p for p in (DATOS / split / clase).glob("*") if p.suffix.lower() in EXT)
        for i, p in enumerate(archivos, 1):
            X.append(embedding_de_archivo(p))
            y.append(etiqueta)
            if i % 100 == 0:
                print(f"  {split}/{clase}: {i}/{len(archivos)}")
    X, y = np.array(X), np.array(y)
    np.savez(cache, X=X, y=y)
    return X, y


def umbral_eer(y, score):
    fpr, tpr, umbrales = roc_curve(y, score)
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return umbrales[i], (fpr[i] + 1 - tpr[i]) / 2


if __name__ == "__main__":
    Xtr, ytr = features("train")
    Xva, yva = features("val")
    Xte, yte = features("test")

    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xtr, ytr)

    umbral, eer_val = umbral_eer(yva, clf.predict_proba(Xva)[:, 1])
    p_te = clf.predict_proba(Xte)[:, 1]
    _, eer_te = umbral_eer(yte, p_te)

    print(f"Umbral (val): {umbral:.4f}  EER val: {eer_val:.2%}")
    print(f"Test -> EER: {eer_te:.2%}  AUC: {roc_auc_score(yte, p_te):.4f}  "
          f"Exactitud: {((p_te >= umbral) == yte).mean():.2%}")

    joblib.dump({"clf": clf, "umbral": float(umbral)}, "modelo.joblib")
    print("Modelo guardado en modelo.joblib")
