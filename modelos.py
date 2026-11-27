import copy

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from configuracion import (APAGADO_DE_NEURONAS, COMPONENTES_GMM, CUADROS_POR_CLASE, CUADROS_POR_CLASE_VALIDACION,
                           EPOCAS_MAXIMAS, NEURONAS_OCULTAS, PACIENCIA, SEMILLA, TAMANO_LOTE, TAMANO_LOTE_CUADROS,
                           TASA_APRENDIZAJE)


# ---------- Cuadros: cada audio aporta muchas ventanas de 20 ms ----------
def muestrear_cuadros(cuadros_por_audio, etiquetas, por_clase, semilla=SEMILLA):
    """Junta las ventanas de todos los audios y toma la misma cantidad al azar de cada clase."""
    azar = np.random.default_rng(semilla)
    X, y = [], []
    for clase in (0, 1):
        cuadros = np.concatenate([c for c, e in zip(cuadros_por_audio, etiquetas) if e == clase]).astype(np.float32)
        X.append(cuadros[azar.choice(len(cuadros), min(len(cuadros), por_clase), replace=False)])
        y.append(np.full(len(X[-1]), clase))
    return np.concatenate(X), np.concatenate(y)


# ---------- Modelo de mezcla gaussiana: un modelo para la voz real y otro para la sintética ----------
def entrenar_gmm(cuadros_por_audio, etiquetas):
    X, y = muestrear_cuadros(cuadros_por_audio, etiquetas, CUADROS_POR_CLASE)
    with threadpool_limits(1):
        return {clase: GaussianMixture(COMPONENTES_GMM, covariance_type="diag", max_iter=200, random_state=SEMILLA).fit(X[y == clase])
                for clase in (0, 1)}


def puntuar_gmm(modelos, cuadros):
    cuadros = cuadros.astype(np.float32)
    return modelos[1].score(cuadros) - modelos[0].score(cuadros)


# ---------- Regresión logística ----------
def entrenar_regresion(X, etiquetas):
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(X, etiquetas)


def puntuar_regresion(modelo, X):
    return modelo.predict_proba(X)[:, 1]


# ---------- Perceptrón multicapa ----------
def crear_perceptron(entradas):
    return torch.nn.Sequential(
        torch.nn.Linear(entradas, NEURONAS_OCULTAS),
        torch.nn.ReLU(),
        torch.nn.Dropout(APAGADO_DE_NEURONAS),
        torch.nn.Linear(NEURONAS_OCULTAS, 1),
    )


def pesos_por_clase(etiquetas):
    reales, sinteticas = (etiquetas == 0).sum(), (etiquetas == 1).sum()
    total = len(etiquetas)
    return np.where(etiquetas == 1, total / (2 * sinteticas), total / (2 * reales)).astype(np.float32)


def perdida(red, X, y, pesos):
    return torch.nn.functional.binary_cross_entropy_with_logits(red(X).squeeze(1), y, weight=pesos)


def entrenar_perceptron(X_entrenamiento, y_entrenamiento, X_validacion, y_validacion, semilla, tamano_lote=TAMANO_LOTE):
    """Entrena por épocas y se queda con la red que tuvo menor error en validación."""
    torch.manual_seed(semilla)
    Xe, ye = torch.tensor(X_entrenamiento, dtype=torch.float32), torch.tensor(y_entrenamiento, dtype=torch.float32)
    Xv, yv = torch.tensor(X_validacion, dtype=torch.float32), torch.tensor(y_validacion, dtype=torch.float32)
    pe = torch.tensor(pesos_por_clase(y_entrenamiento))
    pv = torch.tensor(np.where(y_validacion == 1, pe[ye == 1][0].item(), pe[ye == 0][0].item()), dtype=torch.float32)

    red = crear_perceptron(Xe.shape[1])
    optimizador = torch.optim.Adam(red.parameters(), lr=TASA_APRENDIZAJE)
    orden = torch.Generator().manual_seed(semilla)
    mejor_error, mejor_red, mejor_epoca, epocas_sin_mejorar = np.inf, None, 0, 0

    for epoca in range(1, EPOCAS_MAXIMAS + 1):
        red.train()
        for lote in torch.randperm(len(Xe), generator=orden).split(tamano_lote):
            optimizador.zero_grad()
            perdida(red, Xe[lote], ye[lote], pe[lote]).backward()
            optimizador.step()
        red.eval()
        with torch.no_grad():
            error_validacion = perdida(red, Xv, yv, pv).item()
        if error_validacion < mejor_error:
            mejor_error, mejor_red, mejor_epoca, epocas_sin_mejorar = error_validacion, copy.deepcopy(red), epoca, 0
        else:
            epocas_sin_mejorar += 1
            if epocas_sin_mejorar == PACIENCIA:
                break
    return mejor_red.eval(), mejor_epoca


def puntuar_perceptron(red, X):
    with torch.no_grad():
        return torch.sigmoid(red(torch.tensor(np.asarray(X), dtype=torch.float32)).squeeze(1)).numpy()


def entrenar_perceptron_de_cuadros(cuadros_entrenamiento, y_entrenamiento, cuadros_validacion, y_validacion, semilla):
    """El perceptrón aprende a clasificar cada ventana; luego un audio se califica con el promedio de sus ventanas."""
    X, y = muestrear_cuadros(cuadros_entrenamiento, y_entrenamiento, CUADROS_POR_CLASE)
    Xv, yv = muestrear_cuadros(cuadros_validacion, y_validacion, CUADROS_POR_CLASE_VALIDACION)
    media, desviacion = X.mean(axis=0), X.std(axis=0) + 1e-6
    red, epoca = entrenar_perceptron((X - media) / desviacion, y, (Xv - media) / desviacion, yv, semilla, TAMANO_LOTE_CUADROS)
    return {"red": red, "media": media, "desviacion": desviacion}, epoca


def puntuar_perceptron_de_cuadros(modelo, cuadros):
    X = (cuadros.astype(np.float32) - modelo["media"]) / modelo["desviacion"]
    with torch.no_grad():
        return modelo["red"](torch.tensor(X)).mean().item()
