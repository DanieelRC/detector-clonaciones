"""Paso 2. Convierte cada audio en números que los modelos puedan usar.

Uso:  python paso2_extraer_caracteristicas.py            (las tres particiones)
      python paso2_extraer_caracteristicas.py train      (solo una)
"""
import sys
import time

import numpy as np

from audio import calcular_lfcc, calcular_lpc, capas_de_wavlm, cargar_audio, proporcion_de_silencio, ventanas_con_voz
from configuracion import CARPETA_AUDIOS, CARPETA_CARACTERISTICAS, PARTICIONES
from paso1_preparar_datos import leer_manifiesto


def extraer(particion):
    destino = CARPETA_CARACTERISTICAS / f"{particion}.npz"
    if destino.exists():
        print(f"{particion}: ya extraído")
        return
    audios = leer_manifiesto(particion)
    wavlm, lfcc, voz, lpc, silencio = [], [], [], [], []
    tiempo_wavlm = tiempo_acustico = 0.0
    for numero, audio in enumerate(audios, 1):
        senal = cargar_audio(CARPETA_AUDIOS / particion / audio["archivo"])

        inicio = time.time()
        wavlm.append(capas_de_wavlm(senal).astype(np.float16))
        tiempo_wavlm += time.time() - inicio

        inicio = time.time()
        lfcc.append(calcular_lfcc(senal).astype(np.float16))
        voz.append(ventanas_con_voz(senal).numpy())
        lpc.append(calcular_lpc(senal).astype(np.float16))
        silencio.append(proporcion_de_silencio(senal))
        tiempo_acustico += time.time() - inicio

        if numero % 500 == 0:
            print(f"  {particion}: {numero}/{len(audios)}", flush=True)

    CARPETA_CARACTERISTICAS.mkdir(exist_ok=True)
    np.savez(destino, archivos=[a["archivo"] for a in audios], wavlm=np.array(wavlm),
             lfcc=np.array(lfcc, dtype=object), voz=np.array(voz, dtype=object), lpc=np.array(lpc, dtype=object), silencio=np.array(silencio, dtype=np.float32),
             segundos_wavlm=tiempo_wavlm, segundos_acusticas=tiempo_acustico)
    print(f"{particion}: {len(audios)} audios listos (WavLM {tiempo_wavlm / 60:.1f} min, LFCC y LPC {tiempo_acustico:.0f} s)")


if __name__ == "__main__":
    for particion in sys.argv[1:] or PARTICIONES:
        extraer(particion)
