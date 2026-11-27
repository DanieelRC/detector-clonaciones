# Detector de voces sintéticas en español

Decide si un audio en español es **voz real** o **voz sintética** (generada por clonación de voz) y compara cinco detectores que funcionan en CPU, sin tarjeta gráfica.

| Detector | Qué mira del audio | Cómo decide |
|---|---|---|
| LFCC + GMM | 60 coeficientes LFCC de cada ventana de 20 ms con voz | Dos modelos de mezcla gaussiana (GMM): uno de voz real y otro de voz sintética |
| LFCC + MLP | Las mismas ventanas | Perceptrón multicapa (MLP) que califica cada ventana; se promedian |
| LFCC + LPC + MLP | Las mismas ventanas con 16 coeficientes de predicción lineal (LPC) más | Perceptrón multicapa |
| WavLM + RL | Una capa de WavLM, un modelo de voz preentrenado | Regresión logística (RL) |
| WavLM + MLP | La misma capa de WavLM | Perceptrón multicapa |

Además, un **clasificador de control** usa solo la duración, la proporción de silencio y la frecuencia de muestreo original. Sirve para saber si el conjunto de datos tiene "atajos" que delatan la respuesta.

## Resultados

EER (tasa de error igual: el punto donde las voces sintéticas aceptadas y las reales rechazadas son iguales; menor es mejor) en la partición de prueba:

| Detector | Prueba completa | Hablantes y generadores vistos | Hablantes nuevos | Generadores nuevos |
|---|---|---|---|---|
| LFCC + GMM | 11.87 % | 8.60 % | 17.53 % | **2.70 %** |
| LFCC + MLP | 11.36 % | 2.17 % | 14.81 % | 6.97 % |
| LFCC + LPC + MLP | 10.49 % | 1.24 % | 13.16 % | 5.92 % |
| **WavLM + RL** | **4.17 %** | **0.22 %** | **0.58 %** | 3.33 % |
| WavLM + MLP | 7.32 % | 0.24 % | 1.55 % | 6.78 % |
| Control (sin escuchar la voz) | 6.46 % | 12.52 % | 8.04 % | 0.00 % |

- **WavLM + RL** es el mejor detector y el único que supera al control. Generaliza a hablantes nuevos (0.58 %). Su punto débil es Fish-Speech: detecta el 72.5 % de esos audios.
- Los modelos acústicos usan solo **ventanas con voz**. Si también ven el silencio, su EER baja a 4–4.5 %, porque el silencio de las grabaciones reales y sintéticas es distinto. Es un atajo, no detección de voz clonada.
- El control obtiene 0.00 % con generadores nuevos porque Fish-Speech y XTTS-v1 entregan audio a frecuencias de muestreo que ningún audio real tiene.

## Datos

Se usa [HISPASpoof](https://huggingface.co/datasets/purdueviperlab/HISPASpoof) con su partición oficial:

| Partición | Archivo | Audios usados |
|---|---|---|
| Entrenamiento | `train.tar` (6 GB) | Todos los reales y 1 050 sintéticos por generador |
| Validación | `val.tar` (0.7 GB) | Todos |
| Prueba | `test.tar` (9.4 GB) | Todos los reales; 600 por generador conocido y 1 200 por generador nuevo |

La prueba incluye **6 hablantes** y **2 generadores** (Fish-Speech y XTTS-v1) que no aparecen en entrenamiento. Por eso los resultados se reportan en cuatro condiciones: prueba completa, hablantes y generadores vistos, hablantes nuevos y generadores nuevos.

## Requisitos

- **Python 3.12** (también debería funcionar con 3.10 o 3.11). En Windows usa el instalador oficial de [python.org](https://www.python.org/downloads/). El Python de MSYS2/MinGW no puede instalar PyTorch.
- **35 GB libres** durante la descarga. Al terminar el paso 1 se borran los `.tar` y quedan unos 6 GB de audios.
- No hace falta tarjeta gráfica.

## Instalación

```bash
git clone https://github.com/DanieelRC/detector-clonaciones.git
cd detector-clonaciones
python -m venv .venv
```

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Ejecución: cuatro pasos

```bash
python paso1_preparar_datos.py          # descarga, elige los audios y escribe datos/manifiesto.csv
python paso2_extraer_caracteristicas.py # convierte cada audio en números (WavLM, LFCC, LPC)
python paso3_entrenar_y_evaluar.py      # entrena los cinco detectores y el control, mide y grafica
python paso4_graficas.py                # (opcional) vuelve a dibujar las gráficas
```

| Paso | Qué hace | Tiempo aproximado (CPU de 6 núcleos) |
|---|---|---|
| 1 | Descarga unos 16 GB, extrae los ~17 000 audios elegidos y revisa que se puedan usar | 1–2 h, según la conexión |
| 2 | WavLM (13 capas) y LFCC/LPC por ventana; se guardan en `caracteristicas/` | ~1.5 h |
| 3 | Elige la capa de WavLM con validación, entrena, evalúa y guarda `resultados.json`, `puntuaciones.csv`, `modelos/` y `figuras/` | ~30 min |

### Interfaz con votación de los cinco detectores

```bash
python interfaz.py
```

Abre una ventana mínima.

- El botón **Elegir audio de prueba al azar** toma un audio de la partición de prueba y lo analiza con los cinco detectores.
- Cada detector vota "clonada" si su puntuación alcanza el umbral que eligió en validación. Si 3 o más de los 5 votan "clonada", el resultado es **voz clonada**.
- La ventana muestra la puntuación, el umbral y el voto de cada detector, la respuesta correcta y si la votación acertó.
- **Escuchar** abre el audio con el reproductor del sistema.

Sin ventana: `python interfaz.py --consola` imprime el mismo análisis.

### Analizar un audio propio

Para analizar un audio con el mejor detector (WavLM + regresión logística):

```bash
python detectar.py ruta/al/audio.wav
```

```
Probabilidad de voz sintética: 99.8%
Resultado: SINTÉTICA
```

## Archivos

| Archivo | Contenido |
|---|---|
| `configuracion.py` | Todos los parámetros en un solo lugar: rutas, semillas, tamaños, capas, épocas |
| `audio.py` | Cargar un audio, cortarlo en ventanas, LFCC, LPC y WavLM |
| `modelos.py` | GMM, regresión logística y perceptrón multicapa |
| `metricas.py` | Umbral, EER y remuestreo por hablante |
| `paso1_…` a `paso4_…` | Los cuatro pasos, en orden |
| `detectar.py` | Analiza un audio nuevo con el mejor detector |
| `interfaz.py` | Ventana que analiza un audio de prueba al azar con los cinco detectores y decide por votación |

## Cómo se evalúa

- El umbral de cada detector se fija en **validación**: el punto donde las voces sintéticas aceptadas (FAR) y las reales rechazadas (FRR) son iguales. Ese valor es la **tasa de error igual (EER)**.
- En prueba se reportan EER, área bajo la curva ROC (AUC) y exactitud.
- Los intervalos del 95 % se obtienen remuestreando **hablantes** de prueba.
- Los perceptrones se entrenan con cinco semillas y se reporta la media.

## Créditos de los datos

HISPASpoof: M. Risques, K. Bhagtani, A. K. S. Yadav y E. J. Delp, «HISPASpoof: A New Dataset For Spanish Speech Forensics», arXiv:2509.09155, 2025. Consulta las condiciones de uso en su página de Hugging Face.
