"""Descarga el subconjunto de validación de HISPASpoof y escribe el manifiesto común.

Resultado:
  datos/audios/*.wav
  datos/manifiesto.csv  (archivo, etiqueta, generador, hablante, sr_original, duracion, particion, exclusion)

Partición 60/20/20 por hablante: ningún hablante aparece en dos particiones, y validación
y prueba tienen al menos 4 hablantes cada una.
Uso:  python preparar_datos.py   (descarga unos 730 MB a descargas/ si no hay audios)
"""
import csv
import re
import shutil
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from comun import PICO_MIN, acusticas, cargar_audio

BASE = "https://huggingface.co/datasets/purdueviperlab/HISPASpoof/resolve/main/detection"
DESCARGAS = Path("descargas")
DATOS = Path("datos")
AUDIOS = DATOS / "audios"
MANIFIESTO = DATOS / "manifiesto.csv"
SPLITS = ["train", "val", "test"]
OBJETIVO = np.array([0.6, 0.2, 0.2])
GENERADORES = ["elevenlabs", "f5-tts", "xtts-v2", "your-tts"]
MIN_HABLANTES = 4  # en validación y prueba, para que el remuestreo por hablante tenga más de 3 grupos


def descargar(url, destino):
    if destino.exists():
        return
    print(f"Descargando {url} ...")
    tmp = destino.with_suffix(destino.suffix + ".part")

    def progreso(bloques, tam_bloque, total):
        if total > 0:
            print(f"\r  {min(bloques * tam_bloque, total) / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="")

    urllib.request.urlretrieve(url, tmp, progreso)
    print()
    tmp.rename(destino)


def reunir_audios():
    """Deja todos los audios en datos/audios/ (descargando o moviendo los de una versión anterior)."""
    AUDIOS.mkdir(parents=True, exist_ok=True)
    anteriores = [p for s in SPLITS for p in (DATOS / s).rglob("*.wav")]
    if not anteriores and not any(AUDIOS.glob("*.wav")):
        DESCARGAS.mkdir(exist_ok=True)
        descargar(f"{BASE}/val.tar", DESCARGAS / "val.tar")
        print("Extrayendo val.tar ...")
        with tarfile.open(DESCARGAS / "val.tar") as tar:
            tar.extractall(DESCARGAS / "val_extraido", filter="data")
        anteriores = list((DESCARGAS / "val_extraido").rglob("*.wav"))
    for p in anteriores:
        shutil.move(str(p), AUDIOS / p.name)
    for s in SPLITS:
        shutil.rmtree(DATOS / s, ignore_errors=True)
    DESCARGAS.mkdir(exist_ok=True)
    descargar(f"{BASE}/protocols/val_metadata.csv", DESCARGAS / "val_metadata.csv")


def describir(p):
    m = re.match(r"detection_(.+?)_speaker_(\d+)_", p.name)
    gen = m.group(1)
    fila = {"archivo": p.name, "etiqueta": int(gen != "natural"), "generador": gen, "hablante": int(m.group(2)),
            "sr_original": "", "duracion": "", "particion": "", "exclusion": ""}
    try:
        info = sf.info(str(p))
        fila["sr_original"], fila["duracion"] = info.samplerate, round(info.duration, 4)
        wav = cargar_audio(p, normalizar=False)
    except Exception:
        fila["exclusion"] = "ilegible"
        return fila
    if info.duration < 0.5:
        fila["exclusion"] = "duracion<0.5s"
    elif wav.abs().max() < PICO_MIN:
        fila["exclusion"] = "amplitud<1e-5"
    elif not acusticas(cargar_audio(p))[1].any():
        fila["exclusion"] = "sin_ventanas_con_energia"
    return fila


def particion_por_hablante(filas, intentos=20_000):
    """Asigna hablantes a train/val/test lo más cerca posible de 60/20/20 en reales y en sintéticos."""
    hablantes = sorted({f["hablante"] for f in filas})
    cuenta = {h: np.array([sum(1 for f in filas if f["hablante"] == h and f["etiqueta"] == c) for c in (0, 1)]) for h in hablantes}
    gens = {h: {f["generador"] for f in filas if f["hablante"] == h} for h in hablantes}
    total = sum(cuenta.values())
    rng = np.random.default_rng(0)
    mejor, costo_min = None, np.inf
    for _ in range(intentos):
        asig = rng.integers(0, 3, len(hablantes))
        n = np.zeros((3, 2))
        for h, a in zip(hablantes, asig):
            n[a] += cuenta[h]
        if (n == 0).any() or min((asig == 1).sum(), (asig == 2).sum()) < MIN_HABLANTES:
            continue
        if any(not set(GENERADORES) <= set().union(*[gens[h] for h, a in zip(hablantes, asig) if a == k]) for k in range(3)):
            continue
        costo = ((n / total - OBJETIVO[:, None]) ** 2).sum()
        if costo < costo_min:
            mejor, costo_min = asig, costo
    return {h: SPLITS[a] for h, a in zip(hablantes, mejor)}


def main():
    reunir_audios()
    print("Revisando audios y criterios de inclusión ...")
    filas = [describir(p) for p in sorted(AUDIOS.glob("*.wav"))]

    meta = DESCARGAS / "val_metadata.csv"
    if meta.exists():  # comprobación de etiquetas contra los metadatos oficiales
        with open(meta, encoding="utf-8") as f:
            oficial = {r["filename"]: int(r["label"]) for r in csv.DictReader(f)}
        assert all(oficial[f["archivo"]] == f["etiqueta"] for f in filas if f["archivo"] in oficial)

    incluidas = [f for f in filas if not f["exclusion"]]
    asignacion = particion_por_hablante(incluidas)
    for f in incluidas:
        f["particion"] = asignacion[f["hablante"]]

    with open(MANIFIESTO, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    excl = Counter((f["exclusion"], f["etiqueta"]) for f in filas if f["exclusion"])
    print(f"Exclusiones: {dict(excl) or 'ninguna'}")
    for s in SPLITS:
        fs = [f for f in incluidas if f["particion"] == s]
        hs = sorted({f["hablante"] for f in fs})
        print(f"{s:>5}: {len(fs):4d} audios  reales {sum(1 - f['etiqueta'] for f in fs):3d}  "
              f"sintéticos {sum(f['etiqueta'] for f in fs):4d}  hablantes {hs}")
        print("       " + ", ".join(f"{g} {c}" for g, c in sorted(Counter(f['generador'] for f in fs).items())))
    print(f"Manifiesto escrito en {MANIFIESTO}")


if __name__ == "__main__":
    main()
