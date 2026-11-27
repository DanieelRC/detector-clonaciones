# Detector de voces sintéticas en español

Clasifica un audio en español como **voz real** o **voz sintética** y compara cinco configuraciones que se ejecutan en CPU:

| Configuración | Entrada | Clasificador | EER en prueba |
|---|---|---|---|
| LFCC + GMM | 20 LFCC + Δ + ΔΔ por ventana (60 valores) | Un GMM de 32 componentes por clase | 11.08 % |
| LFCC + MLP | Media y desviación de los LFCC por audio (120 valores) | Perceptrón multicapa | 30.97 ± 0.92 % |
| LFCC + LPC + MLP | Lo anterior más 16 coeficientes LPC (152 valores) | Perceptrón multicapa | 32.28 ± 0.39 % |
| WavLM + RL | WavLM-base congelado, media de la capa 6 (768 valores) | Regresión logística con pesos de clase | 11.08 % |
| WavLM + MLP | Los mismos 768 valores | Perceptrón multicapa | **9.05 ± 0.65 %** |

Los MLP tienen una capa oculta de 32 unidades y se entrenan con cinco semillas (0 a 4); se reportan la media y la desviación estándar.

Los datos son los 2 618 audios del subconjunto de detección de [HISPASpoof](https://huggingface.co/datasets/purdueviperlab/HISPASpoof) (535 reales y 2 083 de ElevenLabs, F5-TTS, XTTS-v2 y YourTTS). Se dividen **por hablante** en entrenamiento, validación y prueba (7, 5 y 6 hablantes), así que la prueba mide el desempeño con voces que el modelo no vio. El umbral de cada modelo se fija en validación.

> **Precaución al leer las cifras.** Un clasificador de control que solo ve la duración, la proporción de silencio y la frecuencia de muestreo original obtiene una EER de 2.57 %, porque en este conjunto cada generador entrega sus audios a una frecuencia distinta. Las cifras de los detectores son, por tanto, una cota optimista.

Proyecto del Equipo 1 (Chamaleon), Escuela Superior de Cómputo, IPN.

## Requisitos

- **Python 3.12** (también debería funcionar con 3.10 o 3.11).
  En Windows usa el instalador oficial de [python.org](https://www.python.org/downloads/). El Python de MSYS2/MinGW no puede instalar PyTorch.
- Unos **3 GB libres**: unos 730 MB de datos, unos 380 MB del modelo WavLM y unos 600 MB de cachés.
- No hace falta GPU.

## Instalación

```bash
git clone https://github.com/DanieelRC/detector-clonaciones.git
cd detector-clonaciones
```

Crea y activa un entorno virtual:

```bash
python -m venv .venv
```

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Instala primero PyTorch para CPU (más ligero que la versión con CUDA) y después el resto:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Ejecución

Ejecuta los pasos en este orden, con el entorno virtual activado y desde la raíz del proyecto.

### 1. Descargar los datos y escribir el manifiesto

```bash
python preparar_datos.py
```

Descarga `val.tar` (unos 730 MB) y sus metadatos desde Hugging Face a `descargas/`, deja los audios en `datos/audios/` y escribe `datos/manifiesto.csv` con una fila por audio:

```
archivo, etiqueta, generador, hablante, sr_original, duracion, particion, exclusion
```

- **Criterios de inclusión:** archivo legible, al menos 0.5 s, pico de al menos 1e-5 y al menos una ventana con energía suficiente. Con este archivo no se excluye ningún audio.
- **Partición por hablante:** se buscan, con semilla 0, asignaciones de hablantes cercanas a 60/20/20 en audios reales y sintéticos. Cada partición debe tener las dos clases, los cuatro generadores y al menos 4 hablantes en validación y en prueba. Ningún hablante aparece en dos particiones.

### 2. Extraer representaciones y entrenar el detector

```bash
python entrenar.py
```

- **Representaciones:** extrae y guarda en caché las de todos los audios.
  - `wavlm.npz`: 768 valores por audio.
  - `acusticas.npz`: cuadros LFCC, resúmenes de 120 y 152 valores y proporción de silencio.
- **Tiempo:** la primera vez descarga `microsoft/wavlm-base`. La extracción tarda unos 20 minutos en una CPU de 6 núcleos; después, segundos.
- **Detector:** al final entrena WavLM + regresión logística y lo guarda en `modelo.joblib` (normalización, clasificador y umbral) para el paso 4.

### 3. Evaluar las cinco configuraciones

```bash
python experimentos.py
```

Entrena y evalúa las cinco configuraciones con el mismo manifiesto:

- umbral fijado en validación;
- en prueba: EER, AUC, exactitud y matriz de confusión;
- diferencias de EER (aporte de LPC, MLP frente a regresión logística, MLP acústico frente a GMM) con intervalo del 95 % por remuestreo pareado de hablantes;
- el control de duración, silencio y frecuencia original;
- errores por generador y por frecuencia original;
- costos de extracción, ajuste y latencia.

Escribe:

- `resultados.json`, con todas las cifras;
- `puntuaciones.csv`, con una puntuación por archivo, configuración y semilla;
- las figuras en `figuras/` (PDF y PNG).

Para regenerar solo las figuras sin volver a entrenar:

```bash
python figuras.py
```

### 4. Analizar un audio

```bash
python detectar.py ruta/al/audio.wav
```

```
Probabilidad de voz sintética: 100.0%
Etiqueta: SINTÉTICA
```

Acepta `.wav`, `.flac`, `.mp3` y `.ogg`. Necesita `modelo.joblib`, que genera el paso 2. Solo analiza los primeros 4 segundos del audio.

## Preprocesamiento y características

- **Audio:** mono, 16 kHz y primeros 4 s, sin relleno. Se resta la media y el pico se escala a 0.95, para que el volumen de la grabación no sirva de pista.
- **Ventanas:** de 20 ms cada 10 ms desde el inicio, con ventana Hann.
- **LFCC:** FFT de 512 puntos, 40 filtros lineales de 0 a 8 kHz, DCT y 20 coeficientes (incluido c0), más Δ y ΔΔ.
- **LPC:** orden 16 por autocorrelación y Levinson-Durbin, solo en las ventanas con RMS ≥ 1e-3.
- **MLP:** 32 unidades ReLU, entropía cruzada con pesos de clase, Adam (1e-3) y lotes de 32. Se entrena hasta 50 épocas y se queda con los mejores pesos de validación, con parada tras 5 épocas sin mejora.

## Estructura del repositorio

| Archivo | Contenido |
|---|---|
| `comun.py` | Preprocesamiento, LFCC, LPC, WavLM y cálculo de la EER |
| `preparar_datos.py` | Descarga de HISPASpoof, criterios de inclusión, partición por hablante y manifiesto |
| `entrenar.py` | Extracción de representaciones y detector desplegable (WavLM + RL) |
| `experimentos.py` | Las cinco configuraciones, comparaciones, control y costos |
| `figuras.py` | Figuras de los resultados |
| `detectar.py` | Análisis de un audio con el modelo guardado |

El repositorio no incluye audios, modelos entrenados, cachés, resultados ni figuras; todo se genera con los pasos anteriores.

## Reproducibilidad

- Partición, MLP (semillas 0 a 4), GMM y remuestreo usan semillas fijas. Dos ejecuciones seguidas de `experimentos.py` producen las mismas cifras.
- El GMM se ajusta con un solo hilo de cómputo, porque con varios hilos su resultado varía ligeramente entre ejecuciones.
- Las cifras de este README se obtuvieron con:
  - Python 3.12;
  - torch 2.15.0 y torchaudio 2.11.1 (CPU);
  - transformers 5.23.0;
  - scikit-learn 1.9.1;
  - NumPy 2.5.5.

## Limitaciones

- Se usa un solo archivo de HISPASpoof, con 7 hablantes de entrenamiento y 6 de prueba. Los intervalos por remuestreo de hablantes son exploratorios.
- Los cuatro generadores aparecen en todas las particiones, así que no se evalúan generadores no vistos. Tampoco se evaluó la generalización a otros conjuntos de datos ni el audio comprimido.
- La frecuencia de muestreo original y la duración permiten anticipar la etiqueta en este conjunto (ver el control).
- El detector debe usarse como una **alerta** para verificar por otra vía, no como prueba forense.

## Créditos de los datos

HISPASpoof: M. Risques, K. Bhagtani, A. K. S. Yadav y E. J. Delp, «HISPASpoof: A New Dataset For Spanish Speech Forensics», arXiv:2509.09155, 2025. Consulta las condiciones de uso en su página de Hugging Face.
