"""Preprocesamiento, representaciones y métrica compartidos por todos los scripts."""
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F
from transformers import WavLMModel

SR = 16000
LARGO = 4 * SR          # se analizan como máximo los primeros 4 s
CAPA = 6                # hidden_states[6]: sexta capa transformer de WavLM-base
VENTANA, SALTO, NFFT = 320, 160, 512   # 20 ms, 10 ms
N_FILTROS, N_LFCC, ORDEN_LPC = 40, 20, 16
PICO, PICO_MIN, RMS_MIN = 0.95, 1e-5, 1e-3

_wavlm = None
_hann = torch.hann_window(VENTANA)
_bancos = F.linear_fbanks(NFFT // 2 + 1, 0.0, SR / 2, N_FILTROS, SR)   # [257, 40]
_dct = F.create_dct(N_LFCC, N_FILTROS, "ortho")                        # [40, 20]


def cargar_audio(ruta, normalizar=True):
    """Mono, 16 kHz, primeros 4 s sin relleno, sin media y con pico en 0.95."""
    datos, sr = sf.read(ruta, dtype="float32", always_2d=True)
    wav = torch.from_numpy(datos.mean(axis=1))
    if sr != SR:  # interpolación sinc con ventana Hann, ancho 6, corte 0.99 (valores por defecto)
        wav = F.resample(wav, sr, SR)
    wav = wav[:LARGO]
    wav = wav - wav.mean()
    pico = wav.abs().max()
    if normalizar and pico >= PICO_MIN:
        wav = wav * (PICO / pico)
    return wav


@torch.no_grad()
def embedding(wav):
    """Media temporal de la capa 6 de WavLM-base congelado (768 valores)."""
    global _wavlm
    if _wavlm is None:
        _wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base").eval()
    salida = _wavlm(wav.unsqueeze(0), output_hidden_states=True)
    return salida.hidden_states[CAPA][0].mean(dim=0).numpy().astype(np.float32)


def ventanas(wav):
    """Ventanas completas de 20 ms cada 10 ms desde el inicio: [T, 320]."""
    return wav.unfold(0, VENTANA, SALTO)


def rms(v):
    return v.pow(2).mean(dim=1).sqrt()


def levinson(r, p=ORDEN_LPC):
    """Levinson-Durbin vectorizado. r: [N, p+1] -> coeficientes a_k de x^[n] = sum a_k x[n-k]: [N, p]."""
    a = np.zeros((len(r), p + 1)); a[:, 0] = 1
    err = r[:, 0].copy()
    for i in range(1, p + 1):
        k = -(r[:, i] + (a[:, 1:i] * r[:, i - 1:0:-1]).sum(axis=1)) / err
        a[:, 1:i] = a[:, 1:i] + k[:, None] * a[:, i - 1:0:-1]
        a[:, i] = k
        err = err * (1 - k ** 2)
    return -a[:, 1:]


def acusticas(wav):
    """LFCC + Δ + ΔΔ de todas las ventanas [T, 60], máscara de energía [T] y LPC de las ventanas seleccionadas [Tm, 16]."""
    v = ventanas(wav)
    mascara = rms(v) >= RMS_MIN
    vh = v * _hann
    potencia = torch.fft.rfft(vh, n=NFFT).abs().pow(2)                  # [T, 257]
    c = (10 * torch.log10((potencia @ _bancos).clamp_min(1e-10))) @ _dct  # [T, 20], incluye c0
    d = F.compute_deltas(c.T)
    dd = F.compute_deltas(d)
    lfcc = torch.cat([c.T, d, dd]).T.numpy()
    x = vh[mascara].double().numpy()
    r = np.stack([(x[:, : VENTANA - k] * x[:, k:]).sum(axis=1) for k in range(ORDEN_LPC + 1)], axis=1)
    r[:, 0] *= 1 + 1e-6
    return lfcc.astype(np.float32), mascara.numpy(), levinson(r).astype(np.float32)


def resumen(lfcc, mascara, lpc=None):
    """Media y desviación estándar poblacional de las ventanas seleccionadas: 120 (LFCC) o 152 valores (LFCC + LPC)."""
    partes = [lfcc[mascara].mean(0), lfcc[mascara].std(0)]
    if lpc is not None:
        partes += [lpc.mean(0), lpc.std(0)]
    return np.concatenate(partes).astype(np.float32)


def prop_baja_energia(wav):
    """Proporción de ventanas con RMS menor que el 1 % del máximo RMS del fragmento (control de silencio)."""
    e = rms(ventanas(wav))
    return float((e < 0.01 * e.max()).float().mean())


def umbral_eer(y, s):
    """Umbral que minimiza |FAR - FRR| (empates: menor FAR) y EER = (FAR + FRR) / 2 en ese punto.

    Clase positiva = sintética (y = 1); s >= umbral se clasifica como sintética.
    FAR: sintéticos aceptados como reales. FRR: reales rechazados como sintéticos.
    """
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    cand = np.append(np.unique(s), np.inf)
    sint, real = np.sort(s[y == 1]), np.sort(s[y == 0])
    far = np.searchsorted(sint, cand, side="left") / len(sint)
    frr = 1 - np.searchsorted(real, cand, side="left") / len(real)
    i = np.lexsort((far, np.abs(far - frr)))[0]
    return float(cand[i]), float((far[i] + frr[i]) / 2)
