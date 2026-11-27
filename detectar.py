"""Uso: python detectar.py audio.wav

Usa el mejor detector (WavLM + regresión logística) que guardó el paso 3.
"""
import sys

import joblib

from audio import capas_de_wavlm, cargar_audio
from configuracion import CARPETA_MODELOS
from modelos import puntuar_regresion

if len(sys.argv) != 2:
    sys.exit(__doc__)

detector = joblib.load(CARPETA_MODELOS / "detector.joblib")
representacion = capas_de_wavlm(cargar_audio(sys.argv[1]))[detector["capa"]]
probabilidad = puntuar_regresion(detector["regresion"], detector["escalador"].transform([representacion]))[0]

print(f"Probabilidad de voz sintética: {probabilidad:.1%}")
print("Resultado:", "SINTÉTICA" if probabilidad >= detector["umbral"] else "REAL")
