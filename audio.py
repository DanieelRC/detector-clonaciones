import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F
from transformers import WavLMModel

from configuracion import (COEFICIENTES_LFCC, ENERGIA_MINIMA_VENTANA, FILTROS_LINEALES, FRECUENCIA, ORDEN_LPC,
                           PICO_DESEADO, PICO_MINIMO, PUNTOS_FFT, MUESTRAS_POR_VENTANA, SALTO_ENTRE_VENTANAS,
                           SEGUNDOS_MAXIMOS)

ventana_hann = torch.hann_window(MUESTRAS_POR_VENTANA)
banco_de_filtros_lineales = F.linear_fbanks(PUNTOS_FFT // 2 + 1, 0.0, FRECUENCIA / 2, FILTROS_LINEALES, FRECUENCIA)
matriz_dct = F.create_dct(COEFICIENTES_LFCC, FILTROS_LINEALES, "ortho")
wavlm = None


def cargar_audio(ruta, normalizar_volumen=True):
    datos, frecuencia_original = sf.read(ruta, dtype="float32", always_2d=True)
    senal = torch.from_numpy(datos.mean(axis=1))
    if frecuencia_original != FRECUENCIA:
        senal = F.resample(senal, frecuencia_original, FRECUENCIA)
    senal = senal[: SEGUNDOS_MAXIMOS * FRECUENCIA]
    senal = senal - senal.mean()
    pico = senal.abs().max()
    if normalizar_volumen and pico >= PICO_MINIMO:
        senal = senal * (PICO_DESEADO / pico)
    return senal


def cortar_en_ventanas(senal):
    return senal.unfold(0, MUESTRAS_POR_VENTANA, SALTO_ENTRE_VENTANAS)


def energia_de_cada_ventana(ventanas):
    return ventanas.pow(2).mean(dim=1).sqrt()


def ventanas_con_voz(senal):
    return energia_de_cada_ventana(cortar_en_ventanas(senal)) >= ENERGIA_MINIMA_VENTANA


def calcular_lfcc(senal):
    ventanas = cortar_en_ventanas(senal) * ventana_hann
    espectro = torch.fft.rfft(ventanas, n=PUNTOS_FFT).abs().pow(2)
    energia_por_banda = (espectro @ banco_de_filtros_lineales).clamp_min(1e-10)
    coeficientes = (10 * torch.log10(energia_por_banda)) @ matriz_dct
    velocidad = F.compute_deltas(coeficientes.T)
    aceleracion = F.compute_deltas(velocidad)
    return torch.cat([coeficientes.T, velocidad, aceleracion]).T.numpy()


def levinson_durbin(autocorrelacion):
    coeficientes = np.zeros((len(autocorrelacion), ORDEN_LPC + 1))
    coeficientes[:, 0] = 1
    error = autocorrelacion[:, 0].copy()
    for i in range(1, ORDEN_LPC + 1):
        reflexion = -(autocorrelacion[:, i] + (coeficientes[:, 1:i] * autocorrelacion[:, i - 1:0:-1]).sum(axis=1)) / error
        coeficientes[:, 1:i] = coeficientes[:, 1:i] + reflexion[:, None] * coeficientes[:, i - 1:0:-1]
        coeficientes[:, i] = reflexion
        error = error * (1 - reflexion ** 2)
    return -coeficientes[:, 1:]


def calcular_lpc(senal):
    """Un vector de 16 coeficientes por cada ventana con voz."""
    ventanas = (cortar_en_ventanas(senal) * ventana_hann)[ventanas_con_voz(senal)].double().numpy()
    autocorrelacion = np.stack([(ventanas[:, : MUESTRAS_POR_VENTANA - k] * ventanas[:, k:]).sum(axis=1)
                                for k in range(ORDEN_LPC + 1)], axis=1)
    autocorrelacion[:, 0] *= 1 + 1e-6
    return levinson_durbin(autocorrelacion)


def proporcion_de_silencio(senal):
    energia = energia_de_cada_ventana(cortar_en_ventanas(senal))
    return float((energia < 0.01 * energia.max()).float().mean())


@torch.no_grad()
def capas_de_wavlm(senal):
    """Devuelve, para cada una de las 13 capas, la media y la desviación en el tiempo: matriz de 13 x 1536."""
    global wavlm
    if wavlm is None:
        wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base").eval()
    capas = wavlm(senal.unsqueeze(0), output_hidden_states=True).hidden_states
    return np.stack([torch.cat([capa[0].mean(dim=0), capa[0].std(dim=0)]).numpy() for capa in capas])
