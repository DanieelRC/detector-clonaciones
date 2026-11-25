import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import WavLMModel

SR = 16000
LARGO = 4 * SR  # 4 segundos
CAPA = 6        # capa intermedia de WavLM-base

_modelo = None


def cargar_audio(ruta):
    """Audio a 16 kHz mono, recortado o rellenado a 4 s."""
    datos, sr = sf.read(ruta, dtype="float32", always_2d=True)
    wav = torch.from_numpy(datos.mean(axis=1))
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if len(wav) >= LARGO:
        return wav[:LARGO]
    return torch.nn.functional.pad(wav, (0, LARGO - len(wav)))


@torch.no_grad()
def embedding(wav):
    """Promedio temporal de la capa CAPA de WavLM-base congelado (vector de 768)."""
    global _modelo
    if _modelo is None:
        _modelo = WavLMModel.from_pretrained("microsoft/wavlm-base").eval()
    salida = _modelo(wav.unsqueeze(0), output_hidden_states=True)
    return salida.hidden_states[CAPA][0].mean(dim=0).numpy()


def embedding_de_archivo(ruta):
    return embedding(cargar_audio(ruta)).astype(np.float32)
