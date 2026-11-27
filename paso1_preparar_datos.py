"""Paso 1. Descarga HISPASpoof, elige los audios que se usan y escribe el manifiesto.

Uso:  python paso1_preparar_datos.py            (las tres particiones)
      python paso1_preparar_datos.py train      (solo una)
"""
import csv
import random
import re
import sys
import tarfile
import urllib.request

import soundfile as sf

from audio import cargar_audio, ventanas_con_voz
from configuracion import (CARPETA_AUDIOS, CARPETA_DESCARGAS, GENERADORES_NUEVOS, MANIFIESTO, PARTICIONES,
                           PICO_MINIMO, SEMILLA, SINTETICOS_POR_GENERADOR_EN_ENTRENAMIENTO,
                           SINTETICOS_POR_GENERADOR_NUEVO_EN_PRUEBA, SINTETICOS_POR_GENERADOR_VISTO_EN_PRUEBA,
                           URL_HISPASPOOF)

COLUMNAS = ["archivo", "particion", "etiqueta", "generador", "hablante", "hablante_nuevo", "generador_nuevo",
            "frecuencia_original", "duracion", "exclusion"]


def descargar(url, destino):
    if destino.exists():
        return
    parcial = destino.with_suffix(destino.suffix + ".part")
    ya_descargado = parcial.stat().st_size if parcial.exists() else 0
    peticion = urllib.request.Request(url, headers={"Range": f"bytes={ya_descargado}-"})
    print(f"Descargando {destino.name} ...")
    with urllib.request.urlopen(peticion) as respuesta, open(parcial, "ab") as archivo:
        while bloque := respuesta.read(1 << 20):
            archivo.write(bloque)
    parcial.rename(destino)


def leer_metadatos(particion):
    descargar(f"{URL_HISPASPOOF}/protocols/{particion}_metadata.csv", CARPETA_DESCARGAS / f"{particion}_metadata.csv")
    filas = []
    with open(CARPETA_DESCARGAS / f"{particion}_metadata.csv", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            generador, hablante = re.match(r"detection_(.+?)_speaker_(\d+)_", fila["filename"]).groups()
            filas.append({"archivo": fila["filename"], "particion": particion, "etiqueta": int(fila["label"]),
                          "generador": generador, "hablante": int(hablante)})
    return filas


def elegir_subconjunto(filas, particion):
    if particion == "val":
        return filas
    azar = random.Random(SEMILLA)
    elegidas = [f for f in filas if f["etiqueta"] == 0]
    for generador in sorted({f["generador"] for f in filas if f["etiqueta"] == 1}):
        del_generador = sorted((f for f in filas if f["generador"] == generador), key=lambda f: f["archivo"])
        if particion == "train":
            cantidad = SINTETICOS_POR_GENERADOR_EN_ENTRENAMIENTO
        elif generador in GENERADORES_NUEVOS:
            cantidad = SINTETICOS_POR_GENERADOR_NUEVO_EN_PRUEBA
        else:
            cantidad = SINTETICOS_POR_GENERADOR_VISTO_EN_PRUEBA
        elegidas += azar.sample(del_generador, min(cantidad, len(del_generador)))
    return sorted(elegidas, key=lambda f: f["archivo"])


def extraer_del_tar(particion, nombres):
    carpeta = CARPETA_AUDIOS / particion
    carpeta.mkdir(parents=True, exist_ok=True)
    faltan = {n for n in nombres if not (carpeta / n).exists()}
    if not faltan:
        return
    archivo_tar = CARPETA_DESCARGAS / f"{particion}.tar"
    descargar(f"{URL_HISPASPOOF}/{particion}.tar", archivo_tar)
    print(f"Extrayendo {len(faltan)} audios de {archivo_tar.name} ...")
    with tarfile.open(archivo_tar) as tar:
        for miembro in tar:
            nombre = miembro.name.split("/")[-1]
            if nombre in faltan:
                (carpeta / nombre).write_bytes(tar.extractfile(miembro).read())
    if particion != "val":
        archivo_tar.unlink()


def revisar_audio(ruta):
    try:
        informacion = sf.info(str(ruta))
        senal = cargar_audio(ruta, normalizar_volumen=False)
    except Exception:
        return None, None, "ilegible"
    if informacion.duration < 0.5:
        return informacion.samplerate, informacion.duration, "dura menos de 0.5 s"
    if senal.abs().max() < PICO_MINIMO:
        return informacion.samplerate, informacion.duration, "silencio total"
    if not ventanas_con_voz(cargar_audio(ruta)).any():
        return informacion.samplerate, informacion.duration, "sin ventanas con voz"
    return informacion.samplerate, informacion.duration, ""


def preparar(particion, hablantes_de_entrenamiento):
    filas = elegir_subconjunto(leer_metadatos(particion), particion)
    extraer_del_tar(particion, [f["archivo"] for f in filas])
    for f in filas:
        f["hablante_nuevo"] = int(f["hablante"] not in hablantes_de_entrenamiento)
        f["generador_nuevo"] = int(f["generador"] in GENERADORES_NUEVOS)
        f["frecuencia_original"], f["duracion"], f["exclusion"] = revisar_audio(CARPETA_AUDIOS / particion / f["archivo"])
    with open(MANIFIESTO.with_name(f"manifiesto_{particion}.csv"), "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS)
        escritor.writeheader()
        escritor.writerows(filas)
    excluidos = sum(1 for f in filas if f["exclusion"])
    print(f"{particion}: {len(filas)} audios ({sum(f['etiqueta'] == 0 for f in filas)} reales), {excluidos} excluidos")


def unir_manifiestos():
    filas = []
    for particion in PARTICIONES:
        with open(MANIFIESTO.with_name(f"manifiesto_{particion}.csv"), encoding="utf-8") as f:
            filas += list(csv.DictReader(f))
    with open(MANIFIESTO, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS)
        escritor.writeheader()
        escritor.writerows(filas)


def leer_manifiesto(particion=None):
    """Audios incluidos, en el orden del manifiesto."""
    archivo = MANIFIESTO.with_name(f"manifiesto_{particion}.csv") if particion else MANIFIESTO
    with open(archivo, encoding="utf-8") as f:
        return [fila for fila in csv.DictReader(f) if not fila["exclusion"]]


if __name__ == "__main__":
    CARPETA_DESCARGAS.mkdir(exist_ok=True)
    hablantes_de_entrenamiento = {f["hablante"] for f in leer_metadatos("train")}
    for particion in sys.argv[1:] or PARTICIONES:
        preparar(particion, hablantes_de_entrenamiento)
    if all(MANIFIESTO.with_name(f"manifiesto_{p}.csv").exists() for p in PARTICIONES):
        unir_manifiestos()
        print(f"Manifiesto completo en {MANIFIESTO}")
