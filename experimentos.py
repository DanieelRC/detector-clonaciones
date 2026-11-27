"""Evaluación de las cinco configuraciones con el manifiesto común.

  1. LFCC + GMM            (cuadros LFCC, un GMM por clase)
  2. LFCC + MLP            (resumen de 120 valores)
  3. LFCC + LPC + MLP      (resumen de 152 valores)
  4. WavLM + RL            (768 valores, regresión logística)
  5. WavLM + MLP           (768 valores)

Usa las cachés de entrenar.py y escribe resultados.json, puntuaciones.csv y las figuras.
"""
import copy
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchaudio
import transformers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from comun import acusticas, cargar_audio, embedding, resumen, umbral_eer
from entrenar import AUDIOS, extraer, manifiesto

GMM, MLP_LFCC, MLP_LPC, RL, MLP_W = "LFCC + GMM", "LFCC + MLP", "LFCC + LPC + MLP", "WavLM + RL", "WavLM + MLP"
CONFIGS = [GMM, MLP_LFCC, MLP_LPC, RL, MLP_W]
SEMILLAS = [0, 1, 2, 3, 4]
N_BOOT = 1000


# ---------- datos ----------
filas = manifiesto()
w, a = extraer(filas)
nombres = np.array([r["archivo"] for r in filas])
part = np.array([r["particion"] for r in filas])
y = np.array([int(r["etiqueta"]) for r in filas])
gen = np.array([r["generador"] for r in filas])
hab = np.array([int(r["hablante"]) for r in filas])
sr = np.array([int(r["sr_original"]) for r in filas])
dur = np.array([float(r["duracion"]) for r in filas])
tr, va, te = (part == "train"), (part == "val"), (part == "test")
idx_tr, idx_va = np.flatnonzero(tr), np.flatnonzero(va)

entradas = {MLP_LFCC: a["lfcc"], MLP_LPC: a["lfcc_lpc"], RL: w["X"], MLP_W: w["X"]}
escaladores = {k: StandardScaler().fit(v[tr]) for k, v in entradas.items()}  # divisor 1 si la varianza es 0
Z = {k: escaladores[k].transform(v).astype(np.float32) for k, v in entradas.items()}


# ---------- clasificadores ----------
def ajustar_mlp(X, semilla):
    """Capa oculta de 32 ReLU, salida sigmoide, BCE con pesos de clase, Adam 1e-3, lotes de 32,
    hasta 50 épocas, mejores pesos por pérdida de validación y parada tras 5 épocas sin mejora."""
    torch.manual_seed(semilla)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).float()
    n0, n1 = (y[tr] == 0).sum(), (y[tr] == 1).sum()
    peso = torch.where(yt == 1, tr.sum() / (2 * n1), tr.sum() / (2 * n0)).float()
    red = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(red.parameters(), lr=1e-3)
    perdida = lambda i: torch.nn.functional.binary_cross_entropy_with_logits(red(Xt[i]).squeeze(1), yt[i], weight=peso[i])
    g = torch.Generator().manual_seed(semilla)
    itr, iva = torch.from_numpy(idx_tr), torch.from_numpy(idx_va)
    mejor, estado, sin_mejora, epoca_mejor = np.inf, None, 0, 0
    for epoca in range(1, 51):
        red.train()
        for lote in itr[torch.randperm(len(itr), generator=g)].split(32):
            opt.zero_grad(); perdida(lote).backward(); opt.step()
        red.eval()
        with torch.no_grad():
            pv = perdida(iva).item()
        if pv < mejor:
            mejor, estado, sin_mejora, epoca_mejor = pv, copy.deepcopy(red.state_dict()), 0, epoca
        else:
            sin_mejora += 1
            if sin_mejora == 5:
                break
    red.load_state_dict(estado)  # mejores pesos según validación
    red.eval()
    return red, epoca_mejor


def puntuar_mlp(red, X):
    with torch.no_grad():
        return torch.sigmoid(red(torch.from_numpy(np.asarray(X, dtype=np.float32))).squeeze(1)).numpy()


# ---------- métricas ----------
def evaluar(s):
    """Umbral en validación; métricas de prueba con ese umbral y EER de prueba por separado."""
    umbral, eer_val = umbral_eer(y[va], s[va])
    yt_, st = y[te], s[te]
    pred = (st >= umbral).astype(int)
    r = {"umbral": umbral, "eer_val": eer_val, "eer_test": umbral_eer(yt_, st)[1], "auc_test": float(roc_auc_score(yt_, st)),
         "exactitud_test": float((pred == yt_).mean()),
         "far_test": float((pred[yt_ == 1] == 0).mean()), "frr_test": float((pred[yt_ == 0] == 1).mean()),
         "confusion": confusion_matrix(yt_, pred).tolist(), "por_generador": {}, "por_origen_sr": {}}
    g_te, sr_te = gen[te], sr[te]
    for g in sorted(set(g_te)):
        m = g_te == g
        r["por_generador"][g] = {"n": int(m.sum()), "errores": int((pred[m] != yt_[m]).sum()), "acierto": float((pred[m] == yt_[m]).mean())}
        if g != "natural":
            mm = m | (yt_ == 0)
            r["por_generador"][g]["eer"] = umbral_eer(yt_[mm], st[mm])[1]
    for g, f in sorted(set(zip(g_te, sr_te))):
        m = (g_te == g) & (sr_te == f)
        r["por_origen_sr"][f"{g} {f}"] = {"n": int(m.sum()), "errores": int((pred[m] != yt_[m]).sum())}
    return r


# Remuestreo pareado por hablante sobre la partición de prueba
hab_te, y_te = hab[te], y[te]
grupos = {h: np.flatnonzero(hab_te == h) for h in sorted(set(hab_te))}
rng_boot = np.random.default_rng(0)
REPLICAS = []
while len(REPLICAS) < N_BOOT:
    elegidos = rng_boot.choice(list(grupos), len(grupos), replace=True)
    i = np.concatenate([grupos[h] for h in elegidos])
    if len(set(y_te[i])) == 2:  # se vuelve a muestrear si una réplica tiene una sola clase
        REPLICAS.append(i)


def ic95(fn):
    """fn(indices) -> valor; percentiles 2.5 y 97.5 sobre las réplicas."""
    v = [fn(i) for i in REPLICAS]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def eer_medio(lista_s, i):
    return np.mean([umbral_eer(y_te[i], s[te][i])[1] for s in lista_s])


def comparar(base, nueva):
    """Diferencia de EER (nueva - base) por semilla, en puntos porcentuales, con IC por remuestreo pareado."""
    pares = [(base[k % len(base)], nueva[k % len(nueva)]) for k in range(max(len(base), len(nueva)))]
    dif = [100 * (umbral_eer(y_te, sn[te])[1] - umbral_eer(y_te, sb[te])[1]) for sb, sn in pares]
    ic = ic95(lambda i: 100 * np.mean([umbral_eer(y_te[i], sn[te][i])[1] - umbral_eer(y_te[i], sb[te][i])[1] for sb, sn in pares]))
    return {"por_semilla_pp": dif, "media_pp": float(np.mean(dif)), "semillas_con_reduccion": int(sum(d < 0 for d in dif)),
            "ic95_pp": ic, "reduccion_concluyente": ic[1] < 0}


if __name__ == "__main__":
    t0 = time.time()
    punt = {}          # (configuración, semilla) -> puntuaciones de todos los audios (NaN en entrenamiento para el GMM)
    ajuste_s, epocas = {}, {}

    # 1. LFCC + GMM (100 000 cuadros de entrenamiento por clase, un solo hilo, semilla 0)
    cuadros = a["cuadros"]
    rng = np.random.default_rng(0)
    t = time.time()
    with threadpool_limits(1):
        gmm = {}
        for c in (0, 1):
            fr = np.concatenate([cuadros[i] for i in idx_tr if y[i] == c])
            fr = fr[rng.choice(len(fr), min(len(fr), 100_000), replace=False)]
            gmm[c] = GaussianMixture(32, covariance_type="diag", max_iter=200, random_state=0).fit(fr)
        ajuste_s[GMM] = time.time() - t
        s = np.full(len(y), np.nan)
        for i in np.flatnonzero(va | te):
            s[i] = gmm[1].score(cuadros[i]) - gmm[0].score(cuadros[i])
    punt[(GMM, None)] = s

    # 4. WavLM + regresión logística
    t = time.time()
    rl = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Z[RL][tr], y[tr])
    ajuste_s[RL] = time.time() - t
    punt[(RL, None)] = rl.predict_proba(Z[RL])[:, 1]

    # 2, 3, 5. Perceptrones multicapa, cinco semillas cada uno
    redes = {}
    for cfg in (MLP_LFCC, MLP_LPC, MLP_W):
        t = time.time()
        for sem in SEMILLAS:
            red, ep = ajustar_mlp(Z[cfg], sem)
            redes[(cfg, sem)] = red
            epocas[f"{cfg} | {sem}"] = ep
            punt[(cfg, sem)] = puntuar_mlp(red, Z[cfg])
        ajuste_s[cfg] = (time.time() - t) / len(SEMILLAS)

    # Evaluación
    res_cfg = {}
    dims = {GMM: "60 por cuadro", MLP_LFCC: 120, MLP_LPC: 152, RL: 768, MLP_W: 768}
    for cfg in CONFIGS:
        claves = [k for k in punt if k[0] == cfg]
        corridas = [dict(semilla=k[1], **evaluar(punt[k])) for k in claves]
        lista = [punt[k] for k in claves]
        eers = [c["eer_test"] for c in corridas]
        res_cfg[cfg] = {
            "dimension": dims[cfg], "corridas": corridas,
            "eer_test_media": float(np.mean(eers)), "eer_test_de": float(np.std(eers)),
            "eer_val_media": float(np.mean([c["eer_val"] for c in corridas])),
            "auc_test_media": float(np.mean([c["auc_test"] for c in corridas])),
            "exactitud_test_media": float(np.mean([c["exactitud_test"] for c in corridas])),
            "eer_test_ic95": ic95(lambda i: eer_medio(lista, i)),
        }
        umbrales = ", ".join(f"{c['umbral']:.3f}" for c in corridas)
        print(f"{cfg:<18} dim {str(dims[cfg]):>13} | audios {tr.sum()}/{va.sum()}/{te.sum()} | umbral(es) {umbrales} | "
              f"EER prueba {100 * np.mean(eers):.2f} ± {100 * np.std(eers):.2f} % | AUC {res_cfg[cfg]['auc_test_media']:.3f}")

    S = lambda cfg: [punt[k] for k in punt if k[0] == cfg]
    comparaciones = {
        "LPC (LFCC + LPC + MLP menos LFCC + MLP)": comparar(S(MLP_LFCC), S(MLP_LPC)),
        "MLP frente a RL sobre WavLM (WavLM + MLP menos WavLM + RL)": comparar(S(RL), S(MLP_W)),
        "MLP acústico frente a GMM (LFCC + MLP menos LFCC + GMM)": comparar(S(GMM), S(MLP_LFCC)),
    }
    for k, v in comparaciones.items():
        print(f"{k}: {v['media_pp']:+.2f} pp, IC95 [{v['ic95_pp'][0]:+.2f}, {v['ic95_pp'][1]:+.2f}], "
              f"reduce en {v['semillas_con_reduccion']}/{len(v['por_semilla_pp'])}")

    # Control: duración original, proporción de ventanas de baja energía y frecuencia original (categórica)
    frecs = sorted(set(sr))
    Xc = np.column_stack([dur, a["baja_energia"]] + [(sr == f).astype(float) for f in frecs])
    esc_c = StandardScaler().fit(Xc[tr])
    ctrl = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(esc_c.transform(Xc[tr]), y[tr])
    s_ctrl = ctrl.predict_proba(esc_c.transform(Xc))[:, 1]
    r_ctrl = evaluar(s_ctrl)
    r_ctrl["eer_test_ic95"] = ic95(lambda i: umbral_eer(y_te[i], s_ctrl[te][i])[1])
    r_ctrl["diferencia_eer_pp"] = {cfg: 100 * (r_ctrl["eer_test"] - res_cfg[cfg]["eer_test_media"]) for cfg in CONFIGS}
    punt[("Control", None)] = s_ctrl
    print(f"Control: EER prueba {100 * r_ctrl['eer_test']:.2f} %, AUC {r_ctrl['auc_test']:.3f}")

    # Latencia desde la lectura hasta la puntuación (media de 20 audios de prueba)
    muestra = [AUDIOS / n for n in nombres[te][:20]]
    red0 = {cfg: redes[(cfg, 0)] for cfg in (MLP_LFCC, MLP_LPC, MLP_W)}

    def fn_gmm(p):
        l = acusticas(cargar_audio(p))[0]
        return gmm[1].score(l) - gmm[0].score(l)

    def fn_acus(cfg):
        def f(p):
            l, m, c = acusticas(cargar_audio(p))
            v = resumen(l, m, c if cfg == MLP_LPC else None)
            return puntuar_mlp(red0[cfg], escaladores[cfg].transform([v]))
        return f

    def fn_wavlm(cfg):
        def f(p):
            v = escaladores[cfg].transform([embedding(cargar_audio(p))])
            return rl.predict_proba(v)[:, 1] if cfg == RL else puntuar_mlp(red0[cfg], v)
        return f

    latencia = {}
    with threadpool_limits(1):
        for cfg, f in [(GMM, fn_gmm), (MLP_LFCC, fn_acus(MLP_LFCC)), (MLP_LPC, fn_acus(MLP_LPC)), (RL, fn_wavlm(RL)), (MLP_W, fn_wavlm(MLP_W))]:
            f(muestra[0])  # calentamiento
            t = time.perf_counter()
            for p in muestra:
                f(p)
            latencia[cfg] = (time.perf_counter() - t) / len(muestra) * 1000

    # Puntuaciones por archivo, configuración y semilla
    with open("puntuaciones.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["archivo", "etiqueta", "particion", "configuracion", "semilla", "puntuacion"])
        for (cfg, sem), s in punt.items():
            for i in np.flatnonzero(va | te):
                wr.writerow([nombres[i], y[i], part[i], cfg, "" if sem is None else sem, f"{s[i]:.8g}"])

    # Descripción de los datos
    datos = {}
    for p in ("train", "val", "test"):
        m = part == p
        datos[p] = {"real": int((y[m] == 0).sum()), "sintetico": int((y[m] == 1).sum()), "hablantes": sorted(set(hab[m].tolist())),
                    "por_generador": {g: int((gen[m] == g).sum()) for g in sorted(set(gen))}}
    sr_gen = {g: {str(f): int(((gen == g) & (sr == f)).sum()) for f in frecs if ((gen == g) & (sr == f)).any()} for g in sorted(set(gen))}
    with open("datos/manifiesto.csv", encoding="utf-8") as f:
        excl = [r for r in csv.DictReader(f) if r["exclusion"]]

    ej_real = next(i for i in np.flatnonzero(te) if gen[i] == "natural")
    ej_sint = next(i for i in np.flatnonzero(te) if gen[i] == "elevenlabs" and hab[i] == hab[ej_real])

    res = {
        "datos": datos, "total": int(len(y)), "hablantes": int(len(set(hab))), "exclusiones": len(excl),
        "duracion_s": {"media": float(dur.mean()), "min": float(dur.min()), "max": float(dur.max()), "frac_mayor_4s": float((dur > 4).mean())},
        "sr_por_generador": sr_gen,
        "configuraciones": res_cfg, "comparaciones": comparaciones, "control": r_ctrl,
        "epocas_mejor_validacion": epocas,
        "costos": {"extraccion_s": {"WavLM": float(w["segundos"]), "LFCC + LPC": float(a["segundos"])},
                   "ajuste_s": ajuste_s, "latencia_ms": latencia},
        "remuestreo": {"replicas": N_BOOT, "unidad": "hablante", "grupos_prueba": len(grupos)},
        "ejemplo_audios": [str(nombres[ej_real]), str(nombres[ej_sint])],
        "entorno": {"python": platform.python_version(), "torch": torch.__version__, "torchaudio": torchaudio.__version__,
                    "transformers": transformers.__version__, "scikit-learn": sklearn.__version__,
                    "numpy": np.__version__, "procesador": platform.processor(), "sistema": platform.platform()},
    }
    Path("resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"resultados.json y puntuaciones.csv escritos ({time.time() - t0:.0f} s)")

    import figuras
    figuras.main()
