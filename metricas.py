import numpy as np

from configuracion import REPLICAS_BOOTSTRAP, SEMILLA


def tasas_de_error(etiquetas, puntuaciones, umbral):
    """FAR: sintéticas aceptadas como reales. FRR: reales rechazadas como sintéticas."""
    sinteticas, reales = puntuaciones[etiquetas == 1], puntuaciones[etiquetas == 0]
    return float((sinteticas < umbral).mean()), float((reales >= umbral).mean())


def elegir_umbral_y_eer(etiquetas, puntuaciones):
    """Busca el umbral donde FAR y FRR se parecen más; la EER es el promedio de ambas en ese punto."""
    etiquetas, puntuaciones = np.asarray(etiquetas), np.asarray(puntuaciones, dtype=float)
    candidatos = np.append(np.unique(puntuaciones), np.inf)
    sinteticas, reales = np.sort(puntuaciones[etiquetas == 1]), np.sort(puntuaciones[etiquetas == 0])
    far = np.searchsorted(sinteticas, candidatos, side="left") / len(sinteticas)
    frr = 1 - np.searchsorted(reales, candidatos, side="left") / len(reales)
    mejor = np.lexsort((far, np.abs(far - frr)))[0]
    return float(candidatos[mejor]), float((far[mejor] + frr[mejor]) / 2)


def calcular_eer(etiquetas, puntuaciones):
    return elegir_umbral_y_eer(etiquetas, puntuaciones)[1]


def replicas_por_hablante(etiquetas, hablantes):
    """Cada réplica elige hablantes al azar (con repetición) y toma todos sus audios."""
    generador = np.random.default_rng(SEMILLA)
    audios_de = {h: np.flatnonzero(hablantes == h) for h in np.unique(hablantes)}
    lista = list(audios_de)
    replicas = []
    while len(replicas) < REPLICAS_BOOTSTRAP:
        elegidos = generador.choice(lista, len(lista), replace=True)
        indices = np.concatenate([audios_de[h] for h in elegidos])
        if len(set(etiquetas[indices])) == 2:
            replicas.append(indices)
    return replicas


def intervalo_95(valores):
    return [float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))]
