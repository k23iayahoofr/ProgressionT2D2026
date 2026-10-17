#!/usr/bin/env python3
"""
Reproduces the experiments of the T2D paper and writes the LaTeX tables that the
manuscript \\input's from the folder given by --out (default: results/).

Usage
-----
  python run_experiments.py --csv T2D_Simulation_Data.csv --cohortB 20   # your data (+ simulated Cohort B)
  python run_experiments.py --demo                                       # self-test on synthetic data only

Required columns (rename them in COLS below if your file differs):
  patient_id, time (numeric, in TIME_UNIT), glucose (mg/dL), medication (insulin/drug dose),
  diet, exercise. Optional: phase (diagnosis/control/follow-up), cohort.

Protocol (Section 7 of the paper)
  * every patient series is split chronologically: first 70 % train, last 30 % test
    (a gap of H samples is left between them to avoid target leakage);
  * population models (B3) are trained on the pooled training segments of all patients;
  * personalized models (B2, B4, B5, proposed) are trained on the patient's own training segment;
  * the proposed method refits the personalized Random Forest every UPDATE_EVERY test samples
    on all samples whose target is already known (online update).
"""
import argparse, os, pickle, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import wilcoxon, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

COLS = dict(patient="patient_id", time="time", glucose="glucose", medication="medication",
            diet="diet", exercise="exercise", phase="phase", cohort="cohort")
TIME_UNIT_PER_DAY = 1440.0      # time column in minutes -> 1440 per day
G_TH = 126.0                    # mg/dL
W, H = 6, 3                     # input window and horizon (samples)
TRAIN_FRAC = 0.7
UPDATE_EVERY = 24               # online refit period (test samples)
GMAX = 3                        # longest gap (samples) that is interpolated
G_MIN, G_MAX = 40.0, 400.0      # physiological range of the sensor (mg/dL)
N_TREES = 100
GRID = [(d, l) for d in (5, 10, None) for l in (1, 5)]
SEED = 0


# ----------------------------------------------------------------------------- simulator
def simulate_cohort(n, hyper, seed, start_id=0, steps=672, ts=15.0):
    """Closed-loop simulator of Eq. (3): dG/dt = alpha*D - beta*I - gamma*E (Euler, step ts minutes).
    Insulin I = clip(kI*(G-Gset), 0, 40) (feedback), meals = 3/day (1 h pulses), exercise = 3 steps on ~50 % of days,
    sensor noise N(0, 2^2). Cohort B (hyper=True) has lower insulin efficacy and a higher glucose set point."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n):
        alpha = rng.lognormal(np.log(1.0), 0.2)
        beta = rng.lognormal(np.log(0.014 if hyper else 0.020), 0.2)
        gamma = rng.lognormal(np.log(0.30), 0.2)
        gset, ki, db = (105.0 if hyper else 85.0), 0.5, 0.05
        D = np.zeros(steps); E = np.zeros(steps); I = np.zeros(steps); G = np.zeros(steps)
        G[0] = rng.normal(140 if hyper else 100, 5)
        for d in range(steps // 96):
            for m in (30, 50, 76):
                s = d * 96 + m + rng.integers(-2, 3)
                D[s:s + 4] += rng.uniform(0.8, 1.4)
            if rng.random() < 0.5:
                s = d * 96 + 70
                E[s:s + 3] = 1.0
        D += db
        for k in range(steps - 1):
            I[k] = np.clip(ki * (G[k] - gset), 0, 40)
            G[k + 1] = G[k] + ts * (alpha * D[k] - beta * I[k] - gamma * E[k]) + rng.normal(0, 0.5)
        I[-1] = I[-2]
        obs = G + rng.normal(0, 2.0, steps)
        ph = np.where(np.arange(steps) < steps / 3, "diagnosis", np.where(np.arange(steps) < 2 * steps / 3, "control", "follow-up"))
        rows.append(pd.DataFrame({COLS["patient"]: f"S{hyper:d}_{start_id + p}", COLS["time"]: np.arange(steps) * ts,
                                  COLS["glucose"]: obs, COLS["medication"]: I, COLS["diet"]: D, COLS["exercise"]: E,
                                  COLS["phase"]: ph, COLS["cohort"]: "B" if hyper else "A",
                                  "alpha_true": alpha, "beta_true": beta, "gamma_true": gamma}))
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- preprocessing
def preprocess(df):
    """Flag out-of-range glucose, interpolate gaps <= GMAX samples, add a median-filtered glucose for the inputs."""
    out, flagged, interp, dropped, total = [], 0, 0, 0, 0
    for pid, d in df.groupby(COLS["patient"], sort=False):
        d = d.sort_values(COLS["time"]).reset_index(drop=True)
        g = d[COLS["glucose"]].astype(float).copy()
        bad = (g < G_MIN) | (g > G_MAX)
        flagged += int(bad.sum()); g[bad] = np.nan
        miss = g.isna()
        g_i = g.interpolate(limit=GMAX, limit_area="inside")
        interp += int((miss & g_i.notna()).sum())
        d["interp"] = (miss & g_i.notna()).astype(float)
        d[COLS["glucose"]] = g_i
        d["g_med"] = g_i.rolling(3, center=False, min_periods=1).median()
        keep = d[COLS["glucose"]].notna()
        dropped += int((~keep).sum()); total += len(d)
        out.append(d[keep].reset_index(drop=True))
    stats = dict(flagged=flagged / total, interpolated=interp / total, dropped=dropped / total)
    return pd.concat(out, ignore_index=True), stats


def build_xy(d):
    """Feature rows for one patient: windows of filtered glucose, medication, diet, exercise + interpolation flag."""
    g, gm = d[COLS["glucose"]].values, d["g_med"].values
    m, di, e = d[COLS["medication"]].values, d[COLS["diet"]].values, d[COLS["exercise"]].values
    it = d["interp"].values
    n = len(d)
    idx = np.arange(W - 1, n - H)
    def win(a): return np.stack([a[idx - k] for k in range(W)], axis=1)
    X = np.hstack([win(gm), win(m), win(di), win(e), it[idx][:, None]])
    groups = {"Glucose history": list(range(0, W)), "Medication / insulin": list(range(W, 2 * W)),
              "Diet intake": list(range(2 * W, 3 * W)), "Exercise": list(range(3 * W, 4 * W)),
              "Data quality flag": [4 * W]}
    y = g[idx + H]
    return X, y, g[idx], idx, groups


def rf(depth, leaf, seed=SEED):
    return RandomForestRegressor(n_estimators=N_TREES, max_depth=depth, min_samples_leaf=leaf,
                                 max_features=0.5, n_jobs=-1, random_state=seed)


# ----------------------------------------------------------------------------- metrics
def clarke_zone(ref, pred):
    if (ref <= 70 and pred <= 70) or (0.8 * ref <= pred <= 1.2 * ref):
        return "A"
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"
    if ((70 <= ref <= 290) and pred >= ref + 110) or ((130 <= ref <= 180) and pred <= (7 / 5) * ref - 182):
        return "C"
    if (ref >= 240 and 70 <= pred <= 180) or (ref <= 175 / 3 and 70 <= pred <= 180) or \
            ((175 / 3 <= ref <= 70) and pred >= (6 / 5) * ref):
        return "D"
    return "B"


def reg_metrics(y, p):
    e = y - p
    ss = ((y - y.mean()) ** 2).sum()
    return dict(mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()), r2=1 - (e ** 2).sum() / ss if ss > 0 else np.nan)


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * pvals[i]); adj[i] = min(1.0, run)
    return adj


def fmt(x, nd=2):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def pm(a, nd=2):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return "--" if len(a) == 0 else f"{a.mean():.{nd}f} $\\pm$ {a.std(ddof=1) if len(a) > 1 else 0:.{nd}f}"


# ----------------------------------------------------------------------------- per-patient evaluation
def fit_ode(d, tr_mask):
    """NNLS estimate of (alpha, beta, gamma) from one-step differences of Eq. (3); returns coefficients and Ts."""
    t = d[COLS["time"]].values; ts = np.median(np.diff(t))
    g = d[COLS["glucose"]].values
    Dd, Id, Ed = d[COLS["diet"]].values, d[COLS["medication"]].values, d[COLS["exercise"]].values
    k = np.where(tr_mask[:-1])[0]
    Xd = np.column_stack([ts * Dd[k], -ts * Id[k], -ts * Ed[k]])
    th, _ = nnls(Xd, g[k + 1] - g[k])
    return th, ts


def run_patient(d, params, with_extras=False, online=True):
    X, y, g0, idx, groups = build_xy(d)
    n = len(X)
    if n < 60:
        return None
    n_tr = int(TRAIN_FRAC * n)
    tr = np.arange(0, n_tr - H); te = np.arange(n_tr, n)
    if len(tr) < 30 or len(te) < 10:
        return None
    depth, leaf = params
    res = {"y": y[te], "g0": g0[te], "t": d[COLS["time"]].values[idx[te] + H]}
    res["B1"] = g0[te]
    ar = Ridge(alpha=1.0).fit(X[tr][:, :W], y[tr]); res["B2"] = ar.predict(X[te][:, :W])
    m4 = rf(depth, leaf).fit(X[tr], y[tr]); res["B4"] = m4.predict(X[te])
    if online:
        pred = np.empty(len(te)); model = m4
        for s in range(0, len(te), UPDATE_EVERY):
            if s > 0:  # refit on all samples whose target is known before this chunk starts
                known = np.arange(0, n_tr + s - H)
                model = rf(depth, leaf).fit(X[known], y[known])
            pred[s:s + UPDATE_EVERY] = model.predict(X[te[s:s + UPDATE_EVERY]])
        res["P"] = pred
    # ODE-only baseline (B5) and perturbed versions
    th, ts = fit_ode(d, np.isin(np.arange(len(d)), idx[tr]))
    Dd, Id, Ed = (d[COLS[k]].values for k in ("diet", "medication", "exercise"))
    def ode_pred(theta):
        j = idx[te]
        return d[COLS["glucose"]].values[j] + H * ts * (theta[0] * Dd[j] - theta[1] * Id[j] - theta[2] * Ed[j])
    res["B5"] = ode_pred(th); res["theta"] = th
    if with_extras:
        res["sens"] = {(c, s): ode_pred(th * np.where(np.arange(3) == c, 1 + s, 1.0)) for c in range(3) for s in (-0.2, 0.2)}
        # grouped permutation importance (MAE increase) and direction check on the personalized model
        base = np.abs(y[te] - m4.predict(X[te])).mean(); rng = np.random.default_rng(SEED)
        imp = {}
        for name, cols in groups.items():
            incs = []
            for _ in range(3):
                Xp = X[te].copy(); perm = rng.permutation(len(te)); Xp[:, cols] = Xp[perm][:, cols]
                incs.append(np.abs(y[te] - m4.predict(Xp)).mean() - base)
            imp[name] = float(np.mean(incs))
        res["imp"] = imp
        delta = m4.predict(X[te]) - g0[te]; sign = {}
        for name, cols, exp in (("Medication / insulin", W, -1), ("Diet intake", 2 * W, +1), ("Exercise", 3 * W, -1)):
            rho = spearmanr(X[te][:, cols], delta).correlation
            sign[name] = bool(np.isfinite(rho) and np.sign(rho) == exp)
        res["sign"] = sign
        res["model"] = m4; res["X_te"] = X[te]; res["ntr"] = len(tr)
    return res


def progression(t, series, days):
    """TAT, slope (mg/dL/day) and time to first crossing (days, censored at window end)."""
    tat = float(np.mean(series >= G_TH))
    slope = float(np.polyfit((t - t[0]) / days, series, 1)[0])
    above = np.where(series >= G_TH)[0]
    tau = float((t[above[0]] - t[0]) / days) if len(above) else float((t[-1] - t[0]) / days)
    return tat, slope, tau, bool(len(above))


# ----------------------------------------------------------------------------- output helpers
def write(out, name, rows):
    with open(os.path.join(out, name), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(" & ".join(r) + " \\\\\n\\hline\n")


def c(x):
    return "\\nt{" + x + "}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--cohortB", type=int, default=0)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--small", action="store_true", help="fewer patients (quick test)")
    # FIX 1: parse_known_args so Jupyter/Colab kernel args (e.g. "-f kernel.json") are ignored
    a, _ = ap.parse_known_args()

    # FIX 2: validate inputs so we never call pd.read_csv(None)
    if not a.demo:
        if a.csv is None:
            ap.error("either --demo or --csv <path> is required "
                     "(e.g. --csv T2D_Simulation_Data.csv, or --demo for the synthetic self-test)")
        if not os.path.isfile(a.csv):
            ap.error(f"--csv file not found: {a.csv!r}")

    os.makedirs(a.out, exist_ok=True)
    if a.demo:
        nA, nB = (4, 4) if a.small else (12, 12)
        raw = pd.concat([simulate_cohort(nA, False, 1), simulate_cohort(nB, True, 2, 100)], ignore_index=True)
        raw.to_csv(os.path.join(a.out, "demo_T2D_schema.csv"), index=False)
    else:
        raw = pd.read_csv(a.csv)
        if COLS["cohort"] not in raw:
            raw[COLS["cohort"]] = "A"
        if a.cohortB > 0:
            raw = pd.concat([raw, simulate_cohort(a.cohortB, True, 2, 1000)], ignore_index=True)

    df, pp = preprocess(raw)
    days = TIME_UNIT_PER_DAY
    pats = {pid: d for pid, d in df.groupby(COLS["patient"], sort=False)}

    # --- hyper-parameter selection on pooled validation data (last 20 % of every training segment)
    trs, vas = [], []
    for pid, d in pats.items():
        X, y, _, _, _ = build_xy(d); n = len(X); n_tr = int(TRAIN_FRAC * n)
        if n_tr < 60: continue
        cut = int(0.8 * (n_tr - H))
        trs.append((X[:cut], y[:cut])); vas.append((X[cut:n_tr - H], y[cut:n_tr - H]))
    Xt = np.vstack([t[0] for t in trs]); yt = np.concatenate([t[1] for t in trs])
    Xv = np.vstack([v[0] for v in vas]); yv = np.concatenate([v[1] for v in vas])
    best = min(GRID, key=lambda p: np.abs(yv - rf(*p).fit(Xt, yt).predict(Xv)).mean())
    depth, leaf = best

    # --- population model B3 (pooled training segments) ---
    Xp, yp = [], []
    for pid, d in pats.items():
        X, y, _, _, _ = build_xy(d); n = len(X); n_tr = int(TRAIN_FRAC * n)
        if n_tr >= 60: Xp.append(X[:n_tr - H]); yp.append(y[:n_tr - H])
    b3 = rf(depth, leaf).fit(np.vstack(Xp), np.concatenate(yp))

    R = {}
    for k, (pid, d) in enumerate(pats.items()):
        r = run_patient(d, best, with_extras=True)
        if r is None: continue
        X, y, g0, idx, _ = build_xy(d); n_tr = int(TRAIN_FRAC * len(X))
        r["B3"] = b3.predict(X[n_tr:]); r["cohort"] = d[COLS["cohort"]].iloc[0] if COLS["cohort"] in d else "A"
        for p in ("alpha_true", "beta_true", "gamma_true"):
            if p in d: r[p] = float(d[p].iloc[0])
        R[pid] = r
    ids = list(R)
    methods = [("B1", "B1 Persistence"), ("B2", "B2 Autoregressive (ridge, order %d)" % W), ("B3", "B3 Population Random Forest"),
               ("B4", "B4 Personalized RF, no online update"), ("B5", "B5 ODE only, Eq.~\\eqref{eq3}"), ("P", "Proposed (personalized, online)")]

    # --- Table: baselines ---
    mae = {m: np.array([reg_metrics(R[i]["y"], R[i][m])["mae"] for i in ids]) for m, _ in methods}
    rmse = {m: np.array([reg_metrics(R[i]["y"], R[i][m])["rmse"] for i in ids]) for m, _ in methods}
    r2 = {m: np.array([reg_metrics(R[i]["y"], R[i][m])["r2"] for i in ids]) for m, _ in methods}
    others = [m for m, _ in methods if m != "P"]
    pv = []
    for m in others:
        try: pv.append(wilcoxon(mae[m], mae["P"]).pvalue)
        except ValueError: pv.append(1.0)
    pv = dict(zip(others, holm(np.array(pv))))
    rows = []
    for m, label in methods:
        ss = 1 - mae[m].mean() / mae["B1"].mean()
        rows.append([c(label), c(pm(mae[m])), c(pm(rmse[m])), c(pm(r2[m])), c(fmt(ss)),
                     c("--" if m == "P" else ("$<$0.001" if pv[m] < 0.001 else fmt(pv[m], 3)))])
    write(a.out, "tab_baselines.tex", rows)

    # --- Table: detection ---
    rows = []
    for m, label in methods:
        if m not in ("B1", "B3", "B4", "P"): continue
        yt_all = np.concatenate([R[i]["y"] for i in ids]); pr_all = np.concatenate([R[i][m] for i in ids])
        pos = yt_all >= G_TH; pp_ = pr_all >= G_TH
        sens = (pp_ & pos).sum() / max(pos.sum(), 1); spec = (~pp_ & ~pos).sum() / max((~pos).sum(), 1)
        auc = roc_auc_score(pos, pr_all) if 0 < pos.sum() < len(pos) else np.nan
        ab = np.mean([clarke_zone(r_, p_) in ("A", "B") for r_, p_ in zip(yt_all, pr_all)]) * 100
        rows.append([c(label), c(fmt(sens)), c(fmt(spec)), c(fmt(auc)), c(fmt(ab, 1))])
    write(a.out, "tab_detect.tex", rows)

    # --- Table: progression (proposed) ---
    pr = {"TAT": [], "slope": [], "tau": []}; cens = 0
    for i in ids:
        t = R[i]["t"]; m_ = progression(t, R[i]["y"], days); p_ = progression(t, R[i]["P"], days)
        pr["TAT"].append((m_[0], p_[0])); pr["slope"].append((m_[1], p_[1])); pr["tau"].append((m_[2], p_[2])); cens += (not m_[3])
    rows = []
    for key, label in (("TAT", "Time above threshold TAT, Eq.~\\eqref{eq:tat}"), ("slope", "Trend $b_G$ (mg/dL/day), Eq.~\\eqref{eq:slope}"),
                       ("tau", "Time to crossing $\\tau$ (days), Eq.~\\eqref{eq:tau}")):
        arr = np.array(pr[key]); rows.append([c(label), c(fmt(arr[:, 0].mean())), c(fmt(arr[:, 1].mean())), c(fmt(np.abs(arr[:, 0] - arr[:, 1]).mean()))])
    write(a.out, "tab_prog.tex", rows)

    # --- Table: per phase (proposed vs persistence, evaluated inside each phase) ---
    rows = []
    if COLS["phase"] in df:
        for ph in ("diagnosis", "control", "follow-up"):
            res_ph = []
            for pid, d in pats.items():
                dp = d[d[COLS["phase"]] == ph].reset_index(drop=True)
                r = run_patient(dp, best, online=True) if len(dp) > 120 else None
                if r is not None: res_ph.append(r)
            if not res_ph: continue
            mp = np.array([reg_metrics(r["y"], r["P"])["mae"] for r in res_ph]); mb = np.array([reg_metrics(r["y"], r["B1"])["mae"] for r in res_ph])
            rp = np.array([reg_metrics(r["y"], r["P"])["rmse"] for r in res_ph]); r2p = np.array([reg_metrics(r["y"], r["P"])["r2"] for r in res_ph])
            yy = np.concatenate([r["y"] for r in res_ph]); pp_ = np.concatenate([r["P"] for r in res_ph])
            rows.append([c(ph.capitalize()), c(pm(mp)), c(pm(rp)), c(pm(r2p)), c(fmt(1 - mp.mean() / mb.mean())), c(fmt(yy.std(), 2))])
    if rows: write(a.out, "tab_phase.tex", rows)

    # --- Table: sensitivity of the ODE-only model and coefficient recovery ---
    rows = []
    names = ["$\\alpha$", "$\\beta$", "$\\gamma$"]; keys = ["alpha_true", "beta_true", "gamma_true"]
    for k in range(3):
        base = np.array([np.abs(R[i]["y"] - R[i]["B5"]).mean() for i in ids])
        d_m = np.array([np.abs(R[i]["y"] - R[i]["sens"][(k, -0.2)]).mean() for i in ids]) - base
        d_p = np.array([np.abs(R[i]["y"] - R[i]["sens"][(k, 0.2)]).mean() for i in ids]) - base
        rec = [abs(R[i]["theta"][k] - R[i][keys[k]]) / R[i][keys[k]] for i in ids if keys[k] in R[i]]
        rows.append([c(names[k]), c(fmt(d_m.mean(), 3)), c(fmt(d_p.mean(), 3)), c(fmt(float(np.median(rec)) * 100, 1) + "\\%" if rec else "n/a")])
    write(a.out, "tab_sens.tex", rows)

    # --- Table: importance ---
    groups = ["Glucose history", "Medication / insulin", "Diet intake", "Exercise", "Data quality flag"]
    rows = []
    for gname in groups:
        v = np.mean([R[i]["imp"][gname] for i in ids])
        ag = [R[i]["sign"][gname] for i in ids if gname in R[i]["sign"]]
        rows.append([c(gname), c(fmt(v, 3)), c(fmt(np.mean(ag) * 100, 0) + "\\%" if ag else "--")])
    write(a.out, "tab_imp.tex", rows)

    # --- Table: cost ---
    i0 = ids[0]; m0 = R[i0]["model"]; x1 = R[i0]["X_te"][:1]
    for _ in range(20): m0.predict(x1)
    tt = []
    for _ in range(1000):
        t0 = time.perf_counter(); m0.predict(x1); tt.append((time.perf_counter() - t0) * 1000)
    d0 = pats[i0]; X0, y0, _, _, _ = build_xy(d0)
    t0 = time.perf_counter(); rf(depth, leaf).fit(X0[:R[i0]["ntr"]], y0[:R[i0]["ntr"]]); t_up = (time.perf_counter() - t0) * 1000
    mem = len(pickle.dumps(m0)) / 1e6
    thr = []
    for npat in (10, 100, 1000):
        t0 = time.perf_counter()
        for _ in range(npat): m0.predict(x1)
        thr.append(npat / (time.perf_counter() - t0))
    write(a.out, "tab_cplx.tex", [
        [c("Prediction time (ms, mean $\\pm$ SD)"), c(fmt(np.mean(tt), 2) + " $\\pm$ " + fmt(np.std(tt), 2))],
        [c("Online update (refit) time (ms)"), c(fmt(t_up, 0))],
        [c("Memory per model (MB)"), c(fmt(mem, 2))],
        [c("Throughput, $N_p=10/100/1000$ (samples/s)"), c(" / ".join(f"{x:.0f}" for x in thr))]])

    # --- Table: dataset ---
    ns = [len(d) for d in pats.values()]; ts = np.median([np.median(np.diff(d[COLS["time"]].values)) for d in pats.values()])
    gall = df[COLS["glucose"]].values; coh = df.groupby(COLS["cohort"])[COLS["patient"]].nunique() if COLS["cohort"] in df else None
    rows = [[c("Number of patients"), c(str(len(pats)) + ("" if coh is None else " (" + ", ".join(f"cohort {k}: {v}" for k, v in coh.items()) + ")"))],
            [c("Samples per patient (median [min--max])"), c(f"{int(np.median(ns))} [{min(ns)}--{max(ns)}]")],
            [c("Sampling period (median, time units)"), c(fmt(float(ts), 1))],
            [c("Columns used"), c("patient, time, glucose, medication, diet, exercise" + (", phase" if COLS["phase"] in df else ""))],
            [c("Share of samples with $G\\geq g_{\\mathrm{th}}$"), c(fmt(float((gall >= G_TH).mean()) * 100, 1) + "\\%")],
            [c("Outliers flagged / gaps interpolated / rows dropped"), c(f"{pp['flagged']*100:.2f}\\% / {pp['interpolated']*100:.2f}\\% / {pp['dropped']*100:.2f}\\%")]]
    write(a.out, "tab_dataset.tex", rows)

    with open(os.path.join(a.out, "macros.tex"), "w") as f:
        f.write("\\newcommand{\\selDepth}{%s}\n\\newcommand{\\selLeaf}{%d}\n\\newcommand{\\nTest}{%d}\n"
                % ("none" if depth is None else depth, leaf, int(np.sum([len(R[i]['y']) for i in ids]))))
    print("done: results written to", a.out, "| selected (max_depth, min_leaf) =", best, "| patients used:", len(ids))


if __name__ == "__main__":
    main()