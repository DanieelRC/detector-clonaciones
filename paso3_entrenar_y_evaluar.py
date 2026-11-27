"""Paso 3. Entrena los cinco detectores y el control, y los evalúa con los mismos audios.

  1. LFCC + GMM          ventanas de 20 ms, un modelo de mezcla gaussiana por clase
  2. LFCC + MLP          las mismas ventanas, un perceptrón multicapa
  3. LFCC + LPC + MLP    las mismas ventanas con 16 coeficientes LPC más
  4. WavLM + RL          una capa de WavLM, regresión logística
  5. WavLM + MLP         la misma capa, perceptrón multicapa
  Control                solo duración, silencio y frecuencia de muestreo original
"""
import csv
import json
import platform
import time

import joblib
import numpy as np
import sklearn
import torch
import torchaudio
import transformers
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from audio import calcular_lfcc, calcular_lpc, capas_de_wavlm, cargar_audio, ventanas_con_voz
from configuracion import (ARCHIVO_PUNTUACIONES, ARCHIVO_RESULTADOS, CARPETA_AUDIOS, CARPETA_CARACTERISTICAS,
                           CARPETA_MODELOS, PARTICIONES, SEMILLAS_PERCEPTRON)
from metricas import calcular_eer, elegir_umbral_y_eer, intervalo_95, replicas_por_hablante, tasas_de_error
from modelos import (entrenar_gmm, entrenar_perceptron, entrenar_perceptron_de_cuadros, entrenar_regresion, puntuar_gmm,
                     puntuar_perceptron, puntuar_perceptron_de_cuadros, puntuar_regresion)
from paso1_preparar_datos import leer_manifiesto

GMM, MLP_LFCC, MLP_LPC, RL_WAVLM, MLP_WAVLM = "LFCC + GMM", "LFCC + MLP", "LFCC + LPC + MLP", "WavLM + RL", "WavLM + MLP"
CONFIGURACIONES = [GMM, MLP_LFCC, MLP_LPC, RL_WAVLM, MLP_WAVLM]
GMM_CON_SILENCIO, MLP_CON_SILENCIO = "LFCC + GMM con silencio", "LFCC + MLP con silencio"
TODOS = CONFIGURACIONES + ["Control", GMM_CON_SILENCIO, MLP_CON_SILENCIO]


# ---------------------------------------------------------------- datos
audios = leer_manifiesto()
caracteristicas = {p: np.load(CARPETA_CARACTERISTICAS / f"{p}.npz", allow_pickle=True) for p in PARTICIONES}
for p in PARTICIONES:
    assert list(caracteristicas[p]["archivos"]) == [a["archivo"] for a in audios if a["particion"] == p]


def columna(nombre, tipo=str):
    return np.array([tipo(a[nombre]) for a in audios])


def unir(nombre):
    return np.concatenate([caracteristicas[p][nombre] for p in PARTICIONES])


particion = columna("particion")
etiqueta = columna("etiqueta", int)
generador = columna("generador")
hablante = columna("hablante", int)
hablante_nuevo = columna("hablante_nuevo", int) == 1
generador_nuevo = columna("generador_nuevo", int) == 1
frecuencia_original = columna("frecuencia_original", int)
duracion = columna("duracion", float)
archivo = columna("archivo")

entrenamiento, validacion, prueba = particion == "train", particion == "val", particion == "test"
CONDICIONES = {
    "Prueba completa": prueba,
    "Hablantes y generadores vistos": prueba & ~hablante_nuevo & ~generador_nuevo,
    "Hablantes nuevos": prueba & hablante_nuevo & ~generador_nuevo,
    "Generadores nuevos": prueba & ~hablante_nuevo & (generador_nuevo | (etiqueta == 0)),
}

wavlm_por_capa = unir("wavlm")                       # audios x 13 capas x 1536
cuadros_lfcc_con_silencio = list(unir("lfcc"))     # por audio: todas las ventanas x 60
cuadros_lfcc = [c[voz] for c, voz in zip(cuadros_lfcc_con_silencio, unir("voz"))]   # solo ventanas con voz
cuadros_lfcc_lpc = [np.hstack([a, b]) for a, b in zip(cuadros_lfcc, unir("lpc"))]   # ventanas x 76
silencio = unir("silencio")


def solo(lista, mascara):
    return [lista[i] for i in np.flatnonzero(mascara)]


def puntuar_cada_audio(funcion, cuadros_por_audio):
    """Califica validación y prueba; entrenamiento no se usa para medir."""
    return np.array([np.nan if entrenamiento[i] else funcion(c) for i, c in enumerate(cuadros_por_audio)])


def normalizar(X):
    escalador = StandardScaler().fit(X[entrenamiento])
    return escalador.transform(X).astype(np.float32), escalador


# ---------------------------------------------------------------- evaluación
def evaluar(puntuaciones):
    umbral, eer_validacion = elegir_umbral_y_eer(etiqueta[validacion], puntuaciones[validacion])
    resultado = {"umbral": umbral, "eer_validacion": eer_validacion, "condiciones": {}, "por_generador": {},
                 "por_frecuencia_real": {}}
    for nombre, mascara in CONDICIONES.items():
        y, s = etiqueta[mascara], puntuaciones[mascara]
        far, frr = tasas_de_error(y, s, umbral)
        resultado["condiciones"][nombre] = {
            "reales": int((y == 0).sum()), "sinteticos": int((y == 1).sum()), "hablantes": int(len(set(hablante[mascara]))),
            "eer": calcular_eer(y, s), "auc": float(roc_auc_score(y, s)), "exactitud": float(((s >= umbral) == y).mean()),
            "far": far, "frr": frr, "confusion": confusion_matrix(y, (s >= umbral).astype(int)).tolist()}
    reales_de_prueba = prueba & (etiqueta == 0)
    for g in sorted(set(generador[prueba])):
        mascara = prueba & (generador == g)
        acierto = float(((puntuaciones[mascara] >= umbral) == etiqueta[mascara]).mean())
        resultado["por_generador"][g] = {"n": int(mascara.sum()), "acierto": acierto}
        if g != "natural":
            juntos = mascara | reales_de_prueba
            resultado["por_generador"][g]["eer"] = calcular_eer(etiqueta[juntos], puntuaciones[juntos])
    for f in sorted(set(frecuencia_original[reales_de_prueba])):
        mascara = reales_de_prueba & (frecuencia_original == f)
        resultado["por_frecuencia_real"][str(f)] = {"n": int(mascara.sum()),
                                                    "acierto": float((puntuaciones[mascara] < umbral).mean())}
    return resultado


def resumir_semillas(corridas):
    resumen = {}
    for nombre in CONDICIONES:
        for medida in ("eer", "auc", "exactitud", "far", "frr"):
            valores = [c["condiciones"][nombre][medida] for c in corridas]
            resumen.setdefault(nombre, {})[medida] = float(np.mean(valores))
            resumen[nombre][medida + "_desviacion"] = float(np.std(valores))
    return resumen


if __name__ == "__main__":
    inicio_total = time.time()
    puntuaciones, segundos_de_ajuste, epocas = {}, {}, {}
    CARPETA_MODELOS.mkdir(exist_ok=True)

    # 1. Elegir la capa de WavLM con validación
    eer_por_capa = []
    for capa in range(wavlm_por_capa.shape[1]):
        X, _ = normalizar(wavlm_por_capa[:, capa].astype(np.float32))
        modelo = entrenar_regresion(X[entrenamiento], etiqueta[entrenamiento])
        eer_por_capa.append(calcular_eer(etiqueta[validacion], puntuar_regresion(modelo, X[validacion])))
        print(f"Capa {capa:2d}: EER de validación {100 * eer_por_capa[-1]:.2f} %", flush=True)
    mejor_capa = int(np.argmin(eer_por_capa))
    X_wavlm, escalador_wavlm = normalizar(wavlm_por_capa[:, mejor_capa].astype(np.float32))
    print(f"Capa elegida: {mejor_capa}")

    # 2. LFCC + GMM
    inicio = time.time()
    gmm = entrenar_gmm(solo(cuadros_lfcc, entrenamiento), etiqueta[entrenamiento])
    segundos_de_ajuste[GMM] = time.time() - inicio
    with threadpool_limits(1):
        puntuaciones[(GMM, "")] = puntuar_cada_audio(lambda cuadros: puntuar_gmm(gmm, cuadros), cuadros_lfcc)

    # 3. WavLM + regresión logística
    inicio = time.time()
    regresion = entrenar_regresion(X_wavlm[entrenamiento], etiqueta[entrenamiento])
    segundos_de_ajuste[RL_WAVLM] = time.time() - inicio
    puntuaciones[(RL_WAVLM, "")] = puntuar_regresion(regresion, X_wavlm)

    # 4. Perceptrones multicapa, cinco semillas cada uno
    redes = {}
    for nombre, cuadros in [(MLP_LFCC, cuadros_lfcc), (MLP_LPC, cuadros_lfcc_lpc)]:
        inicio = time.time()
        for semilla in SEMILLAS_PERCEPTRON:
            modelo, epoca = entrenar_perceptron_de_cuadros(solo(cuadros, entrenamiento), etiqueta[entrenamiento],
                                                           solo(cuadros, validacion), etiqueta[validacion], semilla)
            redes[(nombre, semilla)], epocas[f"{nombre} | {semilla}"] = modelo, epoca
            puntuaciones[(nombre, str(semilla))] = puntuar_cada_audio(lambda c: puntuar_perceptron_de_cuadros(modelo, c), cuadros)
            print(f"  {nombre}, semilla {semilla}: mejor época {epoca}", flush=True)
        segundos_de_ajuste[nombre] = (time.time() - inicio) / len(SEMILLAS_PERCEPTRON)
    inicio = time.time()
    for semilla in SEMILLAS_PERCEPTRON:
        red, epoca = entrenar_perceptron(X_wavlm[entrenamiento], etiqueta[entrenamiento], X_wavlm[validacion], etiqueta[validacion], semilla)
        redes[(MLP_WAVLM, semilla)], epocas[f"{MLP_WAVLM} | {semilla}"] = red, epoca
        puntuaciones[(MLP_WAVLM, str(semilla))] = puntuar_perceptron(red, X_wavlm)
    segundos_de_ajuste[MLP_WAVLM] = (time.time() - inicio) / len(SEMILLAS_PERCEPTRON)

    # 5. Análisis del silencio: los mismos modelos acústicos, pero también con las ventanas sin voz
    gmm_con_silencio = entrenar_gmm(solo(cuadros_lfcc_con_silencio, entrenamiento), etiqueta[entrenamiento])
    with threadpool_limits(1):
        puntuaciones[(GMM_CON_SILENCIO, "")] = puntuar_cada_audio(lambda c: puntuar_gmm(gmm_con_silencio, c), cuadros_lfcc_con_silencio)
    for semilla in SEMILLAS_PERCEPTRON:
        modelo, _ = entrenar_perceptron_de_cuadros(solo(cuadros_lfcc_con_silencio, entrenamiento), etiqueta[entrenamiento],
                                                   solo(cuadros_lfcc_con_silencio, validacion), etiqueta[validacion], semilla)
        puntuaciones[(MLP_CON_SILENCIO, str(semilla))] = puntuar_cada_audio(lambda c: puntuar_perceptron_de_cuadros(modelo, c),
                                                                            cuadros_lfcc_con_silencio)

    # 6. Control: solo rasgos de cómo se prepararon los audios
    frecuencias = sorted(set(frecuencia_original))
    X_control = np.column_stack([duracion, silencio] + [(frecuencia_original == f).astype(float) for f in frecuencias])
    X_control, _ = normalizar(X_control)
    control = entrenar_regresion(X_control[entrenamiento], etiqueta[entrenamiento])
    puntuaciones[("Control", "")] = puntuar_regresion(control, X_control)

    # 7. Métricas
    resultados = {}
    for nombre in TODOS:
        corridas = [dict(semilla=s, **evaluar(p)) for (n, s), p in puntuaciones.items() if n == nombre]
        resultados[nombre] = {"corridas": corridas, "resumen": resumir_semillas(corridas)}
        completa = resultados[nombre]["resumen"]["Prueba completa"]
        print(f"{nombre:<25} EER prueba completa {100 * completa['eer']:.2f} %   AUC {completa['auc']:.3f}", flush=True)

    # 8. Intervalos y comparaciones con remuestreo por hablante (prueba completa)
    y_prueba, h_prueba = etiqueta[prueba], hablante[prueba]
    replicas = replicas_por_hablante(y_prueba, h_prueba)

    def lista(nombre):
        return [p[prueba] for (n, _), p in puntuaciones.items() if n == nombre]

    for nombre in TODOS:
        corridas = lista(nombre)
        resultados[nombre]["intervalo_eer"] = intervalo_95(
            [np.mean([calcular_eer(y_prueba[r], s[r]) for s in corridas]) for r in replicas])

    def comparar(base, nueva):
        b, n = lista(base), lista(nueva)
        pares = [(b[k % len(b)], n[k % len(n)]) for k in range(max(len(b), len(n)))]
        diferencias = [100 * (calcular_eer(y_prueba, sn) - calcular_eer(y_prueba, sb)) for sb, sn in pares]
        intervalo = intervalo_95([100 * np.mean([calcular_eer(y_prueba[r], sn[r]) - calcular_eer(y_prueba[r], sb[r])
                                                 for sb, sn in pares]) for r in replicas])
        return {"base": base, "nueva": nueva, "por_semilla_pp": diferencias, "media_pp": float(np.mean(diferencias)),
                "semillas_que_reducen": int(sum(d < 0 for d in diferencias)), "intervalo_pp": intervalo}

    comparaciones = {"H1": comparar(RL_WAVLM, MLP_WAVLM), "H2": comparar(MLP_LFCC, MLP_LPC),
                     "GMM frente a MLP acústico": comparar(GMM, MLP_LFCC),
                     "Quitar el silencio en el GMM": comparar(GMM_CON_SILENCIO, GMM),
                     "Quitar el silencio en el MLP": comparar(MLP_CON_SILENCIO, MLP_LFCC)}
    for clave, c in comparaciones.items():
        print(f"{clave}: {c['media_pp']:+.2f} pp, intervalo {c['intervalo_pp'][0]:+.2f} a {c['intervalo_pp'][1]:+.2f}")

    # 9. Tiempo de respuesta por audio (lectura + características + puntuación), un solo hilo
    muestra = [CARPETA_AUDIOS / "test" / a for a in archivo[prueba][:20]]
    def lfcc_con_voz(senal):
        return calcular_lfcc(senal)[ventanas_con_voz(senal).numpy()]

    respuesta = {
        GMM: lambda r: puntuar_gmm(gmm, lfcc_con_voz(cargar_audio(r))),
        MLP_LFCC: lambda r: puntuar_perceptron_de_cuadros(redes[(MLP_LFCC, 0)], lfcc_con_voz(cargar_audio(r))),
        MLP_LPC: lambda r: puntuar_perceptron_de_cuadros(
            redes[(MLP_LPC, 0)], np.hstack([lfcc_con_voz(s := cargar_audio(r)), calcular_lpc(s)])),
        RL_WAVLM: lambda r: puntuar_regresion(regresion, escalador_wavlm.transform([capas_de_wavlm(cargar_audio(r))[mejor_capa]])),
        MLP_WAVLM: lambda r: puntuar_perceptron(redes[(MLP_WAVLM, 0)],
                                                escalador_wavlm.transform([capas_de_wavlm(cargar_audio(r))[mejor_capa]])),
    }
    latencia = {}
    with threadpool_limits(1):
        for nombre, funcion in respuesta.items():
            funcion(muestra[0])
            inicio = time.perf_counter()
            for ruta in muestra:
                funcion(ruta)
            latencia[nombre] = (time.perf_counter() - inicio) / len(muestra) * 1000

    # 10. Guardar
    def umbral_de(nombre):
        return resultados[nombre]["corridas"][0]["umbral"]

    modelos_guardados = {
        "lfcc_gmm": {"nombre": GMM, "gmm": gmm, "umbral": umbral_de(GMM)},
        "lfcc_mlp": {"nombre": MLP_LFCC, "modelo": redes[(MLP_LFCC, 0)], "umbral": umbral_de(MLP_LFCC)},
        "lfcc_lpc_mlp": {"nombre": MLP_LPC, "modelo": redes[(MLP_LPC, 0)], "umbral": umbral_de(MLP_LPC)},
        "wavlm_rl": {"nombre": RL_WAVLM, "capa": mejor_capa, "escalador": escalador_wavlm, "regresion": regresion,
                     "umbral": umbral_de(RL_WAVLM)},
        "wavlm_mlp": {"nombre": MLP_WAVLM, "capa": mejor_capa, "escalador": escalador_wavlm, "red": redes[(MLP_WAVLM, 0)],
                      "umbral": umbral_de(MLP_WAVLM)},
    }
    for archivo_modelo, contenido in modelos_guardados.items():
        joblib.dump(contenido, CARPETA_MODELOS / f"{archivo_modelo}.joblib")

    with open(ARCHIVO_PUNTUACIONES, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["archivo", "particion", "etiqueta", "configuracion", "semilla", "puntuacion"])
        for (nombre, semilla), p in puntuaciones.items():
            for i in np.flatnonzero(~entrenamiento):
                escritor.writerow([archivo[i], particion[i], etiqueta[i], nombre, semilla, f"{p[i]:.8g}"])

    datos = {}
    for p in PARTICIONES:
        m = particion == p
        datos[p] = {"reales": int((etiqueta[m] == 0).sum()), "sinteticos": int((etiqueta[m] == 1).sum()),
                    "hablantes": sorted(set(hablante[m].tolist())),
                    "por_generador": {g: int((generador[m] == g).sum()) for g in sorted(set(generador[m]))}}
    resultados_finales = {
        "datos": datos, "total": int(len(audios)),
        "frecuencias_por_generador": {g: sorted(set(frecuencia_original[generador == g].tolist())) for g in sorted(set(generador))},
        "reales_por_frecuencia": {p: {str(f): int(((particion == p) & (etiqueta == 0) & (frecuencia_original == f)).sum())
                                      for f in frecuencias} for p in PARTICIONES},
        "duracion": {"media": float(duracion.mean()), "minima": float(duracion.min()), "maxima": float(duracion.max())},
        "eer_validacion_por_capa": eer_por_capa, "capa_elegida": mejor_capa,
        "configuraciones": resultados, "comparaciones": comparaciones, "epocas": epocas,
        "costos": {"extraccion_s": {"WavLM": float(sum(caracteristicas[p]["segundos_wavlm"] for p in PARTICIONES)),
                                    "LFCC y LPC": float(sum(caracteristicas[p]["segundos_acusticas"] for p in PARTICIONES))},
                   "ajuste_s": segundos_de_ajuste, "latencia_ms": latencia},
        "entorno": {"python": platform.python_version(), "torch": torch.__version__, "torchaudio": torchaudio.__version__,
                    "transformers": transformers.__version__, "scikit-learn": sklearn.__version__, "numpy": np.__version__},
    }
    ARCHIVO_RESULTADOS.write_text(json.dumps(resultados_finales, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultados guardados ({time.time() - inicio_total:.0f} s)")

    import paso4_graficas
    paso4_graficas.main()
