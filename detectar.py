"""Uso: python detectar.py audio.wav

Aplica el modelo guardado por entrenar.py (WavLM + regresión logística) a un audio.
"""
import sys

import joblib

from comun import cargar_audio, embedding

if len(sys.argv) != 2:
    sys.exit(__doc__)

m = joblib.load("modelo.joblib")
v = m["escalador"].transform([embedding(cargar_audio(sys.argv[1]))])
p = m["clf"].predict_proba(v)[0, 1]
print(f"Probabilidad de voz sintética: {p:.1%}")
print("Etiqueta:", "SINTÉTICA" if p >= m["umbral"] else "REAL")
