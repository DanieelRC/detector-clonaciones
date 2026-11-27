"""Paso 4. Dibuja las gráficas a partir de resultados.json y puntuaciones.csv."""
import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve

from audio import cargar_audio
from configuracion import ARCHIVO_PUNTUACIONES, ARCHIVO_RESULTADOS, CARPETA_AUDIOS, CARPETA_FIGURAS, FRECUENCIA
from metricas import calcular_eer

PETROLEO, CORAL, MOSTAZA, VERDE, PIZARRA, CIELO, CIRUELA, ARENA = (
    "#0E4D64", "#E4572E", "#F3A712", "#5B9A8B", "#3D405B", "#3A8FB7", "#8E6C8A", "#C9A66B")
NOMBRE = {"natural": "Real", "elevenlabs": "ElevenLabs", "f5-tts": "F5-TTS", "xtts-v2": "XTTS-v2",
          "your-tts": "YourTTS", "fish-speech": "Fish-Speech*", "xtts-v1": "XTTS-v1*"}
COLOR_ORIGEN = {"natural": VERDE, "elevenlabs": CIELO, "f5-tts": CORAL, "xtts-v2": MOSTAZA, "your-tts": CIRUELA,
                "fish-speech": PIZARRA, "xtts-v1": ARENA}
ORIGENES = ["natural", "elevenlabs", "f5-tts", "xtts-v2", "your-tts", "fish-speech", "xtts-v1"]
NOMBRE_PARTICION = {"train": "Entrenamiento", "val": "Validación", "test": "Prueba"}
CONFIGURACIONES = ["LFCC + GMM", "LFCC + MLP", "LFCC + LPC + MLP", "WavLM + RL", "WavLM + MLP"]
COLOR = dict(zip(CONFIGURACIONES + ["Control"], [CORAL, MOSTAZA, CIRUELA, PETROLEO, CIELO, PIZARRA]))

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Calibri", "DejaVu Sans"],
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "legend.fontsize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": PIZARRA,
    "axes.labelcolor": PIZARRA, "xtick.color": PIZARRA, "ytick.color": PIZARRA, "text.color": PIZARRA,
    "savefig.facecolor": "white",
})


def guardar(figura, nombre):
    figura.tight_layout()
    figura.savefig(CARPETA_FIGURAS / f"{nombre}.pdf")
    figura.savefig(CARPETA_FIGURAS / f"{nombre}.png", dpi=220)
    plt.close(figura)


def leer_puntuaciones():
    puntuaciones = defaultdict(lambda: {"val": [], "test": []})
    etiquetas, archivos = {"val": [], "test": []}, {"val": [], "test": []}
    with open(ARCHIVO_PUNTUACIONES, encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            puntuaciones[(fila["configuracion"], fila["semilla"])][fila["particion"]].append(float(fila["puntuacion"]))
            if fila["configuracion"] == "LFCC + GMM":
                etiquetas[fila["particion"]].append(int(fila["etiqueta"]))
                archivos[fila["particion"]].append(fila["archivo"])
    puntuaciones = {k: {p: np.array(v) for p, v in d.items()} for k, d in puntuaciones.items()}
    return puntuaciones, {p: np.array(v) for p, v in etiquetas.items()}, archivos


def puntuacion_principal(puntuaciones, nombre):
    return puntuaciones[(nombre, "0")] if (nombre, "0") in puntuaciones else puntuaciones[(nombre, "")]


def main():
    CARPETA_FIGURAS.mkdir(exist_ok=True)
    R = json.loads(ARCHIVO_RESULTADOS.read_text(encoding="utf-8"))
    C = R["configuraciones"]
    P, y, archivos = leer_puntuaciones()
    for nombre in CONFIGURACIONES:
        corrida = C[nombre]["corridas"][0]
        clave = (nombre, str(corrida["semilla"]))
        assert abs(calcular_eer(y["test"], P[clave]["test"]) - corrida["condiciones"]["Prueba completa"]["eer"]) < 1e-9

    def resumen(nombre, condicion="Prueba completa", medida="eer"):
        return C[nombre]["resumen"][condicion][medida]

    def acierto_medio(nombre, clave, grupo):
        return 100 * np.mean([c[clave][grupo]["acierto"] for c in C[nombre]["corridas"]])

    # Datos por partición y origen
    figura, eje = plt.subplots(figsize=(10, 3.8))
    particiones = ["train", "val", "test"]
    acumulado = np.zeros(3)
    for g in ORIGENES:
        v = np.array([R["datos"][p]["por_generador"].get(g, 0) for p in particiones])
        eje.barh([NOMBRE_PARTICION[p] for p in particiones], v, left=acumulado, color=COLOR_ORIGEN[g], label=NOMBRE[g])
        acumulado += v
    for i, p in enumerate(particiones):
        eje.text(acumulado[i] + 80, i, f"{int(acumulado[i]):,} audios · {len(R['datos'][p]['hablantes'])} hablantes".replace(",", " "),
                 va="center", fontsize=13)
    eje.invert_yaxis(); eje.set_xlabel("Número de audios"); eje.set_xlim(0, acumulado.max() * 1.45)
    eje.legend(ncol=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=11)
    guardar(figura, "fig_datos")

    # Forma de onda y espectrograma de un audio real y uno sintético del mismo hablante
    prueba = archivos["test"]
    real = next(a for a in prueba if "_natural_" in a)
    hablante = real.split("_speaker_")[1].split("_")[0]
    sintetico = next(a for a in prueba if "_elevenlabs_" in a and f"_speaker_{hablante}_" in a)
    espectrograma = torchaudio.transforms.MelSpectrogram(FRECUENCIA, n_fft=1024, hop_length=256, n_mels=80)
    figura, ejes = plt.subplots(2, 2, figsize=(10, 5), sharex="col")
    for j, (nombre, titulo, color) in enumerate([(real, "Voz real", VERDE), (sintetico, "Voz sintética (ElevenLabs)", CORAL)]):
        senal = cargar_audio(CARPETA_AUDIOS / "test" / nombre)
        segundos = len(senal) / FRECUENCIA
        ejes[0, j].plot(np.arange(len(senal)) / FRECUENCIA, senal.numpy(), color=color, lw=0.4)
        ejes[0, j].set_title(f"{titulo} · hablante {hablante}"); ejes[0, j].set_ylim(-1, 1)
        ejes[1, j].imshow(torch.log(espectrograma(senal) + 1e-6).numpy(), origin="lower", aspect="auto",
                          extent=[0, segundos, 0, 80], cmap="magma")
        ejes[1, j].set_xlabel("Tiempo (s)")
    ejes[0, 0].set_ylabel("Amplitud"); ejes[1, 0].set_ylabel("Banda Mel")
    guardar(figura, "fig_onda_mel")

    # Filtros lineales frente a Mel (esquema)
    def triangulos(centros, f):
        return [np.clip(np.minimum((f - a) / (b - a), (c - f) / (c - b)), 0, 1) for a, b, c in zip(centros[:-2], centros[1:-1], centros[2:])]
    f = np.linspace(0, 8000, 2000)
    mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    inversa = lambda m: 700 * (10 ** (m / 2595) - 1)
    figura, ejes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    for eje, centros, titulo, color in [(ejes[0], np.linspace(0, 8000, 18), "Escala lineal (LFCC): mismo detalle en todo el espectro", CORAL),
                                        (ejes[1], inversa(np.linspace(0, mel(8000), 18)), "Escala Mel (MFCC): más detalle en graves", PETROLEO)]:
        for k, t in enumerate(triangulos(centros, f)):
            eje.fill_between(f / 1000, t, color=color, alpha=.18 + .12 * (k % 2)); eje.plot(f / 1000, t, color=color, lw=1)
        eje.set_title(titulo, loc="left", fontsize=14); eje.set_yticks([]); eje.set_ylim(0, 1.3)
    ejes[1].set_xlabel("Frecuencia (kHz)")
    guardar(figura, "fig_filtros")

    # FAR y FRR frente al umbral (validación, LFCC + GMM: su error de validación permite ver el cruce)
    corrida = C["LFCC + GMM"]["corridas"][0]
    s, yv = puntuacion_principal(P, "LFCC + GMM")["val"], y["val"]
    umbrales = np.linspace(*np.percentile(s, [0.5, 99.5]), 501)
    figura, eje = plt.subplots(figsize=(8.5, 4))
    eje.plot(umbrales, [(s[yv == 1] < u).mean() * 100 for u in umbrales], color=CORAL, lw=2.5, label="Sintéticas aceptadas (FAR)")
    eje.plot(umbrales, [(s[yv == 0] >= u).mean() * 100 for u in umbrales], color=VERDE, lw=2.5, label="Reales rechazadas (FRR)")
    eje.axvline(corrida["umbral"], color=PIZARRA, ls="--", lw=1.3)
    eje.plot([corrida["umbral"]], [100 * corrida["eer_validacion"]], "o", color=PETROLEO, ms=10, zorder=5)
    eje.set_xlabel("Umbral sobre la puntuación del detector (LFCC + GMM)"); eje.set_ylabel("Error (%)"); eje.set_ylim(0, 100)
    eje.annotate(f"EER = {100 * corrida['eer_validacion']:.1f} %", xy=(corrida["umbral"], 100 * corrida["eer_validacion"]),
                 xytext=(corrida["umbral"] + 0.12 * (umbrales[-1] - umbrales[0]), 45), fontsize=14, color=PETROLEO, weight="bold",
                 arrowprops=dict(arrowstyle="->", color=PETROLEO))
    eje.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=2)
    guardar(figura, "fig_eer")

    # EER de validación por capa de WavLM
    eer_capas = 100 * np.array(R["eer_validacion_por_capa"])
    figura, eje = plt.subplots(figsize=(9, 3.6))
    colores = [CIELO if k == R["capa_elegida"] else "#B8C4CC" for k in range(len(eer_capas))]
    barras = eje.bar(range(len(eer_capas)), eer_capas, color=colores)
    eje.bar_label(barras, fmt="%.1f", fontsize=10)
    eje.annotate("elegida", xy=(R["capa_elegida"], eer_capas[R["capa_elegida"]] + 0.05),
                 xytext=(R["capa_elegida"], eer_capas.max() * 0.6), ha="center", color=CIELO, fontsize=13, weight="bold",
                 arrowprops=dict(arrowstyle="->", color=CIELO))
    eje.set_xticks(range(len(eer_capas))); eje.set_xlabel("Capa de WavLM (0 = entrada del transformer)")
    eje.set_ylabel("EER en validación (%)")
    guardar(figura, "fig_capas")

    # EER de prueba completa con intervalo
    figura, eje = plt.subplots(figsize=(9.5, 4))
    for i, nombre in enumerate(CONFIGURACIONES):
        e = resumen(nombre) * 100
        lo, hi = [100 * v for v in C[nombre]["intervalo_eer"]]
        eje.barh(i, e, color=COLOR[nombre], height=.6)
        eje.errorbar(e, i, xerr=[[e - lo], [hi - e]], fmt="none", ecolor=PIZARRA, capsize=6, lw=1.8)
        if len(C[nombre]["corridas"]) > 1:
            eje.scatter([100 * c["condiciones"]["Prueba completa"]["eer"] for c in C[nombre]["corridas"]], [i] * 5,
                        s=22, color="white", edgecolor=PIZARRA, zorder=4)
        eje.text(hi + .3, i, f"{e:.2f} %", va="center", fontsize=13)
    eje.set_yticks(range(len(CONFIGURACIONES)), CONFIGURACIONES); eje.invert_yaxis()
    eje.set_xlabel("EER en la prueba completa (%) · menor es mejor")
    eje.set_xlim(0, max(C[n]["intervalo_eer"][1] for n in CONFIGURACIONES) * 100 + 4)
    guardar(figura, "fig_eer_ic")

    # EER por condición de prueba
    condiciones = list(C["WavLM + MLP"]["resumen"])
    figura, eje = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(condiciones)); ancho = .16
    for k, nombre in enumerate(CONFIGURACIONES):
        barras = eje.bar(x + (k - 2) * ancho, [100 * resumen(nombre, c) for c in condiciones], ancho, color=COLOR[nombre], label=nombre)
        eje.bar_label(barras, fmt="%.1f", fontsize=9)
    eje.set_xticks(x, [c.replace(" y ", " y\n") for c in condiciones]); eje.set_ylabel("EER en prueba (%)")
    eje.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(.5, 1.0), fontsize=11)
    guardar(figura, "fig_condiciones")

    # ROC en la prueba completa
    figura, eje = plt.subplots(figsize=(6.2, 5.4))
    for nombre in CONFIGURACIONES:
        fpr, tpr, _ = roc_curve(y["test"], puntuacion_principal(P, nombre)["test"])
        eje.plot(fpr * 100, tpr * 100, color=COLOR[nombre], lw=2.2,
                 label=f"{nombre} · AUC {C[nombre]['corridas'][0]['condiciones']['Prueba completa']['auc']:.3f}")
    eje.plot([0, 100], [100, 0], ls=":", color=PIZARRA, lw=1, label="FAR = FRR")
    eje.set_xlabel("Reales marcadas como sintéticas (%)"); eje.set_ylabel("Sintéticas detectadas (%)")
    eje.legend(loc="lower right", frameon=False, fontsize=10.5)
    guardar(figura, "fig_roc")

    # Matrices de confusión en la prueba completa
    figura, ejes = plt.subplots(1, 5, figsize=(17, 3.9))
    for eje, nombre in zip(ejes, CONFIGURACIONES):
        matriz = np.array(C[nombre]["corridas"][0]["condiciones"]["Prueba completa"]["confusion"])
        eje.imshow(matriz / matriz.sum(1, keepdims=True), cmap="GnBu", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                fr = matriz[i, j] / matriz[i].sum()
                eje.text(j, i, f"{matriz[i, j]}\n({fr * 100:.1f} %)", ha="center", va="center", fontsize=12,
                         color="white" if fr > .5 else PIZARRA)
        eje.set_xticks([0, 1], ["Real", "Sintética"]); eje.set_yticks([0, 1], ["Real", "Sintética"])
        eje.set_xlabel("Predicción"); eje.set_title(nombre, fontsize=13)
        for borde in eje.spines.values():
            borde.set_visible(False)
    ejes[0].set_ylabel("Etiqueta verdadera")
    guardar(figura, "fig_confusion")

    # Acierto por origen
    origenes = [g for g in ORIGENES if g in C["WavLM + MLP"]["corridas"][0]["por_generador"]]
    figura, eje = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(origenes))
    for k, nombre in enumerate(CONFIGURACIONES):
        barras = eje.bar(x + (k - 2) * ancho, [acierto_medio(nombre, "por_generador", g) for g in origenes], ancho,
                         color=COLOR[nombre], label=nombre)
        eje.bar_label(barras, fmt="%.0f", fontsize=9)
    eje.set_xticks(x, [NOMBRE[g] for g in origenes]); eje.set_ylabel("Acierto en prueba (%)"); eje.set_ylim(0, 112)
    eje.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(.5, 1.0), fontsize=11)
    eje.text(1.0, -0.16, "* generador que no aparece en entrenamiento", transform=eje.transAxes, ha="right", fontsize=11)
    guardar(figura, "fig_generador")

    # Reales por frecuencia de muestreo original
    frecuencias = list(C["WavLM + MLP"]["corridas"][0]["por_frecuencia_real"])
    n = C["WavLM + MLP"]["corridas"][0]["por_frecuencia_real"]
    figura, eje = plt.subplots(figsize=(9, 3.9))
    x = np.arange(len(frecuencias))
    for k, nombre in enumerate(CONFIGURACIONES):
        barras = eje.bar(x + (k - 2) * ancho, [acierto_medio(nombre, "por_frecuencia_real", f) for f in frecuencias], ancho,
                         color=COLOR[nombre], label=nombre)
        eje.bar_label(barras, fmt="%.0f", fontsize=9.5)
    eje.set_xticks(x, [f"Reales a {int(f) / 1000:g} kHz\nn = {n[f]['n']}" for f in frecuencias])
    eje.set_ylabel("Acierto (%)"); eje.set_ylim(0, 115)
    eje.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(.5, 1.0), fontsize=11)
    guardar(figura, "fig_muestreo")

    # Diferencias entre configuraciones
    comparaciones = R["comparaciones"]
    etiquetas = {"H1": "H1: MLP frente a RL\n(WavLM + MLP − WavLM + RL)", "H2": "H2: aporte de LPC\n(LFCC + LPC + MLP − LFCC + MLP)",
                 "GMM frente a MLP acústico": "MLP frente a GMM\n(LFCC + MLP − LFCC + GMM)",
                 "Quitar el silencio en el GMM": "Quitar el silencio\n(GMM solo voz − GMM con silencio)",
                 "Quitar el silencio en el MLP": "Quitar el silencio\n(MLP solo voz − MLP con silencio)"}
    figura, eje = plt.subplots(figsize=(11, 5.4))
    for i, (clave, c) in enumerate(comparaciones.items()):
        lo, hi = c["intervalo_pp"]
        eje.plot([lo, hi], [i, i], color=PIZARRA, lw=3, solid_capstyle="round")
        eje.scatter(c["por_semilla_pp"], [i] * len(c["por_semilla_pp"]), s=40, color=MOSTAZA, edgecolor=PIZARRA, zorder=4)
        eje.plot([c["media_pp"]], [i], "D", color=PETROLEO, ms=10, zorder=5)
        eje.annotate(f"{c['media_pp']:+.2f} pp", xy=(hi, i), xytext=(10, 0), textcoords="offset points", va="center", fontsize=13)
    eje.axvline(0, color=CORAL, ls="--", lw=1.5)
    eje.set_yticks(range(len(comparaciones)), [etiquetas[k] for k in comparaciones], fontsize=12)
    eje.invert_yaxis(); eje.set_xlabel("Diferencia de EER (puntos porcentuales) · negativo = la segunda es mejor")
    guardar(figura, "fig_comparaciones")

    # Efecto del silencio en los modelos acústicos
    figura, eje = plt.subplots(figsize=(8.5, 3.8))
    pares = [("LFCC + GMM", "LFCC + GMM con silencio"), ("LFCC + MLP", "LFCC + MLP con silencio")]
    x = np.arange(len(pares))
    for k, (texto, color) in enumerate([("Solo ventanas con voz", CORAL), ("Todas las ventanas (con silencio)", PIZARRA)]):
        barras = eje.bar(x + (k - .5) * .38, [100 * resumen(par[k]) for par in pares], .38, color=color, label=texto)
        eje.bar_label(barras, fmt="%.1f %%", fontsize=12)
    eje.set_xticks(x, [par[0] for par in pares]); eje.set_ylabel("EER en la prueba completa (%)")
    eje.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(.5, 1.0))
    guardar(figura, "fig_silencio")

    # Control frente a detectores
    figura, eje = plt.subplots(figsize=(9, 3.8))
    nombres = ["Control"] + CONFIGURACIONES
    barras = eje.bar(range(len(nombres)), [100 * resumen(n) for n in nombres], color=[COLOR[n] for n in nombres])
    eje.bar_label(barras, fmt="%.1f %%", fontsize=12)
    eje.set_xticks(range(len(nombres)), [n.replace(" + ", "\n+ ") for n in nombres], fontsize=11)
    eje.set_ylabel("EER en la prueba completa (%)")
    guardar(figura, "fig_control")

    # Diagrama del flujo
    figura, eje = plt.subplots(figsize=(12, 4.6)); eje.axis("off"); eje.set_xlim(0, 12); eje.set_ylim(-0.05, 4.75)

    def caja(x0, y0, ancho_c, alto, texto, color, letra=11):
        eje.add_patch(FancyBboxPatch((x0, y0), ancho_c, alto, boxstyle="round,pad=0.04", fc=color, ec="none"))
        eje.text(x0 + ancho_c / 2, y0 + alto / 2, texto, ha="center", va="center", color="white", fontsize=letra)

    def flecha(a, b):
        eje.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=PIZARRA, lw=1.3))
    caja(0.1, 1.9, 1.6, 0.9, "Audio\n16 kHz · mono\n≤ 4 s · pico 0.95", PIZARRA, 10.5)
    caja(2.4, 3.4, 2.2, 0.8, f"WavLM congelado\ncapa {R['capa_elegida']} · 1 536 valores", PETROLEO)
    caja(2.4, 1.9, 2.2, 0.8, "LFCC (ventanas con voz)\n60 por ventana", CORAL)
    caja(2.4, 0.4, 2.2, 0.8, "LPC de orden 16\n16 por ventana", CIRUELA)
    caja(5.4, 0.25, 2.0, 0.8, "LFCC + LPC\n76 por ventana", MOSTAZA, 10.5)
    for yy, texto, color in [(3.95, "Regresión logística", PETROLEO), (3.0, "MLP de WavLM", CIELO), (2.05, "GMM real y\nGMM sintético", CORAL),
                             (1.1, "MLP con LFCC", MOSTAZA), (0.15, "MLP con LFCC + LPC", CIRUELA)]:
        caja(8.1, yy, 2.2, 0.65, texto, color, 10.5)
        flecha((10.35, yy + .33), (10.75, 2.35))
    caja(10.8, 1.9, 1.1, 0.9, "Umbral\npor modelo\n→ real o\nsintética", VERDE, 10)
    for destino in [(2.35, 3.8), (2.35, 2.3), (2.35, 0.8)]:
        flecha((1.75, 2.35), destino)
    for a, b in [((4.65, 3.8), (8.05, 4.27)), ((4.65, 3.8), (8.05, 3.33)), ((4.65, 2.3), (8.05, 2.38)), ((4.65, 2.3), (8.05, 1.43)),
                 ((4.65, 2.3), (5.35, 0.95)), ((4.65, 0.8), (5.35, 0.65)), ((7.45, 0.65), (8.05, 0.48))]:
        flecha(a, b)
    guardar(figura, "fig_pipeline")

    print("Gráficas listas en", CARPETA_FIGURAS)


if __name__ == "__main__":
    main()
