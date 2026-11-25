"""Uso: python detectar.py audio.wav"""
import sys

import joblib

from comun import embedding_de_archivo

if len(sys.argv) != 2:
    sys.exit(__doc__)

m = joblib.load("modelo.joblib")
p = m["clf"].predict_proba([embedding_de_archivo(sys.argv[1])])[0, 1]
print(f"Probabilidad de voz sintética: {p:.1%}")
print("Etiqueta:", "SINTÉTICA" if p >= m["umbral"] else "REAL")
