# Detector de voces sintéticas en español

Clasifica un audio en español como **voz real** o **voz sintética** y compara dos modelos que se ejecutan en CPU:

| Modelo | Cómo funciona | EER en prueba |
|---|---|---|
| **Modelo de referencia** | 20 coeficientes LFCC (+ Δ y ΔΔ) por ventana y un modelo de mezcla gaussiana (GMM) de 32 componentes por clase | 2.96 % |
| **Modelo principal** | WavLM-base congelado: media temporal de la capa 6 (768 valores) y una regresión logística con pesos de clase | 4.49 % |

Los datos son 2 618 audios del subconjunto de detección de [HISPASpoof](https://huggingface.co/datasets/purdueviperlab/HISPASpoof) (535 reales y 2 083 de ElevenLabs, F5-TTS, XTTS-v2 y YourTTS), divididos 60/20/20 en entrenamiento, validación y prueba. El umbral de decisión se fija en validación.

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

### 1. Descargar y preparar los datos

```bash
python preparar_datos.py
```

Descarga `val.tar` (unos 730 MB) y su archivo de metadatos desde Hugging Face a `descargas/` y arma esta estructura:

```
datos/
├── train/{real,sintetico}/   1 570 audios
├── val/{real,sintetico}/       524 audios
└── test/{real,sintetico}/      524 audios
```

La división usa la semilla 0, así que siempre produce las mismas particiones.

### 2. Entrenar el modelo principal

```bash
python entrenar.py
```

La primera vez descarga los pesos de `microsoft/wavlm-base` y extrae una representación por audio. En una CPU de 6 núcleos tarda unos 15 minutos. Las representaciones se guardan en `feats_*.npz` y las siguientes ejecuciones tardan segundos. Al final imprime la EER, el AUC y la exactitud en prueba y guarda `modelo.joblib`.

### 3. Comparar con el modelo de referencia y generar figuras

```bash
python experimentos.py
```

Calcula los LFCC (en caché en `lfcc_*.npz`), entrena los GMM, mide ambos modelos (EER con intervalo de confianza, AUC, exactitud, matriz de confusión, resultados por generador, control de frecuencia de muestreo y tiempos) y escribe:

- `resultados.json`, con todas las cifras;
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
Probabilidad de voz sintética: 89.2%
Etiqueta: SINTÉTICA
```

Acepta `.wav`, `.flac`, `.mp3` y `.ogg`. Necesita `modelo.joblib`, que genera el paso 2. Solo analiza los primeros 4 segundos del audio.

## Estructura del repositorio

| Archivo | Contenido |
|---|---|
| `comun.py` | Preprocesamiento (16 kHz, mono, 4 s) y representación con WavLM |
| `preparar_datos.py` | Descarga de HISPASpoof y partición 60/20/20 |
| `entrenar.py` | Extracción de representaciones, regresión logística y umbral |
| `experimentos.py` | Modelo de referencia LFCC + GMM, métricas y controles |
| `figuras.py` | Figuras de los resultados |
| `detectar.py` | Análisis de un audio con el modelo guardado |

El repositorio no incluye audios, modelos entrenados, cachés, resultados ni figuras; todo se genera con los pasos anteriores.

## Reproducibilidad

- Particiones, regresión logística y GMM usan semillas fijas.
- El GMM se ajusta con un solo hilo de cómputo, porque con varios hilos su resultado varía ligeramente entre ejecuciones.
- Las cifras de este README se obtuvieron con Python 3.12, torch 2.15.0 y torchaudio 2.11.1 (CPU), transformers 5.23.0 y scikit-learn 1.9.1.

## Limitaciones

- Se usa un solo archivo de HISPASpoof, y la partición es por audio, no por hablante, así que los mismos hablantes aparecen en entrenamiento y en prueba.
- No se evaluó la generalización a otros conjuntos de datos (por ejemplo, entrenar con HABLA y probar con HISPASpoof) ni el audio comprimido.
- El detector debe usarse como una **alerta** para verificar por otra vía, no como prueba forense: con el umbral elegido, alrededor de 5 de cada 100 voces reales se marcan como sintéticas.

## Créditos de los datos

HISPASpoof: M. Risques, K. Bhagtani, A. K. S. Yadav y E. J. Delp, «HISPASpoof: A New Dataset For Spanish Speech Forensics», arXiv:2509.09155, 2025. Consulta las condiciones de uso en su página de Hugging Face.
