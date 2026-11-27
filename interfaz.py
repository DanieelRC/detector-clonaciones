"""Interfaz mínima: elige un audio de prueba al azar, lo analiza con los cinco detectores y decide por votación.

Uso:  python interfaz.py            (ventana)
      python interfaz.py --consola  (sin ventana: analiza un audio al azar e imprime el resultado)
"""
import os
import random
import subprocess
import sys
import tkinter as tk
from tkinter import ttk

import joblib
import numpy as np

from audio import calcular_lfcc, calcular_lpc, capas_de_wavlm, cargar_audio, ventanas_con_voz
from configuracion import CARPETA_AUDIOS, CARPETA_MODELOS
from modelos import puntuar_gmm, puntuar_perceptron, puntuar_perceptron_de_cuadros, puntuar_regresion
from paso1_preparar_datos import leer_manifiesto

NOMBRES_DE_ARCHIVO = ["lfcc_gmm", "lfcc_mlp", "lfcc_lpc_mlp", "wavlm_rl", "wavlm_mlp"]
modelos = {nombre: joblib.load(CARPETA_MODELOS / f"{nombre}.joblib") for nombre in NOMBRES_DE_ARCHIVO}
audios_de_prueba = leer_manifiesto("test")


def puntuar_con_cada_modelo(ruta):
    senal = cargar_audio(ruta)
    lfcc = calcular_lfcc(senal)[ventanas_con_voz(senal).numpy()]
    lfcc_y_lpc = np.hstack([lfcc, calcular_lpc(senal)])
    capas = capas_de_wavlm(senal)

    def representacion_wavlm(modelo):
        return modelo["escalador"].transform([capas[modelo["capa"]]])

    return {
        "lfcc_gmm": puntuar_gmm(modelos["lfcc_gmm"]["gmm"], lfcc),
        "lfcc_mlp": puntuar_perceptron_de_cuadros(modelos["lfcc_mlp"]["modelo"], lfcc),
        "lfcc_lpc_mlp": puntuar_perceptron_de_cuadros(modelos["lfcc_lpc_mlp"]["modelo"], lfcc_y_lpc),
        "wavlm_rl": puntuar_regresion(modelos["wavlm_rl"]["regresion"], representacion_wavlm(modelos["wavlm_rl"]))[0],
        "wavlm_mlp": puntuar_perceptron(modelos["wavlm_mlp"]["red"], representacion_wavlm(modelos["wavlm_mlp"]))[0],
    }


def votar(ruta):
    """Cada detector vota 'clonada' si su puntuación alcanza su umbral; gana la mayoría (3 de 5)."""
    votos = []
    for nombre, puntuacion in puntuar_con_cada_modelo(ruta).items():
        modelo = modelos[nombre]
        votos.append({"detector": modelo["nombre"], "puntuacion": float(puntuacion), "umbral": modelo["umbral"],
                      "dice_clonada": puntuacion >= modelo["umbral"]})
    votos_clonada = sum(v["dice_clonada"] for v in votos)
    return votos, votos_clonada, votos_clonada > len(votos) / 2


def elegir_audio_al_azar():
    audio = random.choice(audios_de_prueba)
    return CARPETA_AUDIOS / "test" / audio["archivo"], audio


def descripcion(audio):
    origen = "voz real" if audio["etiqueta"] == "0" else f"voz clonada con {audio['generador']}"
    extra = []
    if audio["hablante_nuevo"] == "1":
        extra.append("hablante nuevo")
    if audio["generador_nuevo"] == "1":
        extra.append("generador nuevo")
    return f"Respuesta correcta: {origen}" + (f" ({', '.join(extra)})" if extra else "")


def escuchar(ruta):
    if sys.platform == "win32":
        os.startfile(ruta)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(ruta)])


# ---------------------------------------------------------------- ventana
def abrir_ventana():
    ventana = tk.Tk()
    ventana.title("Detector de voces clonadas")
    ventana.configure(padx=24, pady=20, bg="white")
    ventana.resizable(False, False)
    estilo = ttk.Style()
    estilo.configure("Treeview", font=("Segoe UI", 11), rowheight=28)
    estilo.configure("Treeview.Heading", font=("Segoe UI", 11, "bold"))
    actual = {"ruta": None}

    tk.Label(ventana, text="Detector de voces clonadas", font=("Segoe UI", 18, "bold"), bg="white", fg="#0B2E3C").pack(anchor="w")
    tk.Label(ventana, text="Elige un audio de prueba al azar y escucha qué opinan los cinco detectores.",
             font=("Segoe UI", 11), bg="white", fg="#5F6273").pack(anchor="w", pady=(0, 14))

    botones = tk.Frame(ventana, bg="white")
    botones.pack(anchor="w")
    archivo = tk.Label(ventana, text="Ningún audio elegido", font=("Segoe UI", 11), bg="white", fg="#3D405B")
    verdad = tk.Label(ventana, text="", font=("Segoe UI", 11), bg="white", fg="#5F6273")
    tabla = ttk.Treeview(ventana, columns=("puntuacion", "umbral", "voto"), height=5)
    tabla.heading("#0", text="Detector"); tabla.column("#0", width=200)
    for columna, titulo, ancho in [("puntuacion", "Puntuación", 120), ("umbral", "Umbral", 120), ("voto", "Voto", 140)]:
        tabla.heading(columna, text=titulo); tabla.column(columna, width=ancho, anchor="center")
    veredicto = tk.Label(ventana, text="", font=("Segoe UI", 16, "bold"), bg="white")

    def analizar_audio_al_azar():
        ruta, audio = elegir_audio_al_azar()
        actual["ruta"] = ruta
        archivo.config(text=f"Audio: {ruta.name}")
        verdad.config(text="")
        veredicto.config(text="Analizando...", fg="#5F6273")
        tabla.delete(*tabla.get_children())
        ventana.config(cursor="watch"); ventana.update()
        votos, votos_clonada, es_clonada = votar(ruta)
        for v in votos:
            tabla.insert("", "end", text=v["detector"],
                         values=(f"{v['puntuacion']:.3f}", f"{v['umbral']:.3f}", "clonada" if v["dice_clonada"] else "real"))
        acierto = es_clonada == (audio["etiqueta"] == "1")
        veredicto.config(text=f"{'VOZ CLONADA' if es_clonada else 'VOZ REAL'}  ·  {votos_clonada} de {len(votos)} votos dicen clonada",
                         fg="#E4572E" if es_clonada else "#5B9A8B")
        verdad.config(text=f"{descripcion(audio)}  ·  la votación {'acertó' if acierto else 'se equivocó'}")
        ventana.config(cursor="")

    tk.Button(botones, text="Elegir audio de prueba al azar", font=("Segoe UI", 11, "bold"), bg="#0E4D64", fg="white",
              relief="flat", padx=14, pady=6, command=analizar_audio_al_azar).pack(side="left")
    tk.Button(botones, text="Escuchar", font=("Segoe UI", 11), relief="flat", bg="#E6E7EC", padx=14, pady=6,
              command=lambda: actual["ruta"] and escuchar(actual["ruta"])).pack(side="left", padx=10)
    archivo.pack(anchor="w", pady=(16, 6))
    tabla.pack(anchor="w")
    veredicto.pack(anchor="w", pady=(14, 2))
    verdad.pack(anchor="w")
    ventana.mainloop()


if __name__ == "__main__":
    if "--consola" in sys.argv:
        ruta, audio = elegir_audio_al_azar()
        votos, votos_clonada, es_clonada = votar(ruta)
        print(ruta.name)
        for v in votos:
            print(f"  {v['detector']:<18} puntuación {v['puntuacion']:9.3f}  umbral {v['umbral']:9.3f}  "
                  f"vota {'clonada' if v['dice_clonada'] else 'real'}")
        print(f"Resultado: {'VOZ CLONADA' if es_clonada else 'VOZ REAL'} ({votos_clonada} de {len(votos)} votos)")
        print(descripcion(audio))
    else:
        abrir_ventana()
