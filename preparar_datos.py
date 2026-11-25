"""Descarga el subconjunto de validación de HISPASpoof y arma la carpeta datos/.

Resultado:  datos/{train,val,test}/{real,sintetico}/*.wav
División estratificada por clase 60/20/20 con semilla 0.

Uso:  python preparar_datos.py
Descarga unos 730 MB desde Hugging Face a la carpeta descargas/.
"""
import csv
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/purdueviperlab/HISPASpoof/resolve/main/detection"
DESCARGAS = Path("descargas")
EXTRAIDO = DESCARGAS / "val_extraido"
DATOS = Path("datos")


def descargar(url, destino):
    if destino.exists():
        print(f"Ya existe {destino}, no se descarga de nuevo.")
        return
    print(f"Descargando {url} ...")
    tmp = destino.with_suffix(destino.suffix + ".part")

    def progreso(bloques, tam_bloque, total):
        if total > 0:
            print(f"\r  {min(bloques * tam_bloque, total) / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="")

    urllib.request.urlretrieve(url, tmp, progreso)
    print()
    tmp.rename(destino)


def main():
    if DATOS.exists() and any(DATOS.rglob("*.wav")):
        print("La carpeta datos/ ya tiene audios; no se hace nada.")
        return
    DESCARGAS.mkdir(exist_ok=True)
    csv_path = DESCARGAS / "val_metadata.csv"
    tar_path = DESCARGAS / "val.tar"
    descargar(f"{BASE}/protocols/val_metadata.csv", csv_path)
    descargar(f"{BASE}/val.tar", tar_path)

    print("Extrayendo val.tar ...")
    EXTRAIDO.mkdir(exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(EXTRAIDO, filter="data")

    rutas = {p.name: p for p in EXTRAIDO.rglob("*.wav")}
    with open(csv_path, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))  # columnas: filename, true_target, label (1 = sintético)

    random.seed(0)
    for label in ("0", "1"):
        grupo = [f for f in filas if f["label"] == label and f["filename"] in rutas]
        random.shuffle(grupo)
        n = len(grupo)
        cortes = {"train": grupo[: int(0.6 * n)], "val": grupo[int(0.6 * n): int(0.8 * n)], "test": grupo[int(0.8 * n):]}
        clase = "sintetico" if label == "1" else "real"
        for split, fs in cortes.items():
            d = DATOS / split / clase
            d.mkdir(parents=True, exist_ok=True)
            for f in fs:
                shutil.move(str(rutas[f["filename"]]), d / f["filename"])
            print(f"{split:>5} / {clase:<9} {len(fs):>5} audios")
    print("Listo. Puedes borrar la carpeta descargas/ si necesitas espacio.")


if __name__ == "__main__":
    main()
