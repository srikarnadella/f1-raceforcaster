"""2026 season: how can race winners be predicted more confidently?

1. Signals: for every 2026 race, the rank each pre-race signal gave the eventual
   winner (1 = that signal's top pick).
2. Regimes (unsupervised): races clustered by their pre-race shape; how
   predictable the winner was in each regime.
3. Confidence: a walk-forward logistic model of "the favourite wins" from
   pre-race features (2023 onward; each race scored by a model trained on
   earlier races only), and how its confidence tiers did in 2026.
4. Teammates: inside the leading team, which pre-race signal picked the right
   teammate.

Every signal is known before the race. Writes reports/analysis_2026.json.
Usage: python scripts/analysis_2026.py
"""
import json
import warnings

import numpy as np
import pandas as pd

from f1 import improve, ingest, registry
from f1.analytics import build as B
from f1.config import REPORTS

SEASON = 2026


def signals(res, preds_v1, preds_v2):
    """Per driver per race (2023+): pre-race signal values, lower = better."""
    ol, lr, rp = B.read("one_lap"), B.read("long_runs"), B.read("race_pace")
    P = B.read("profiles")
    keys = ["season", "round", "driver_id"]
    d = res[res["season"] >= 2023][keys + ["code", "constructor_id", "circuit_id", "quali_pos", "position",
                                         "classified"]].copy()
    q = ol[ol.session == "Q"][keys + ["gap_pct"]].rename(columns={"gap_pct": "quali_gap"})
    fp = ol[ol.session.str.startswith("FP")].groupby(keys)["best_s"].min().reset_index()
    fp["fp_best"] = fp.groupby(["season", "round"])["best_s"].transform(lambda s: (s / s.min() - 1) * 100)
    d = d.merge(q, on=keys, how="left").merge(fp[keys + ["fp_best"]], on=keys, how="left")
    if len(lr):
        main = lr.groupby(["season", "round", "session"])["long_laps"].sum().reset_index().sort_values("long_laps")
        main = main.groupby(["season", "round"]).tail(1)[["season", "round", "session"]]
        m = lr.merge(main, on=["season", "round", "session"])[keys + ["long_run_gap_pct"]]
        d = d.merge(m.rename(columns={"long_run_gap_pct": "fp_longrun"}), on=keys, how="left")
    s = rp[rp.session == "S"][keys + ["race_pace_gap_pct"]].rename(columns={"race_pace_gap_pct": "sprint_pace"})
    d = d.merge(s, on=keys, how="left")
    if len(P):
        P = P.assign(profile_pace=P["car_race"] + P["race_skill"] / 2)
        d = d.merge(P[keys + ["profile_pace"]], on=keys, how="left")
    for name, pr in (("v1_model", preds_v1), ("v2_model", preds_v2)):
        x = pr[["season", "round", "driver_id", "p_win"]].rename(columns={"p_win": f"{name}_pwin"})
        d = d.merge(x, on=keys, how="left")
        d[name] = -d[f"{name}_pwin"]
    return d


SIGNALS = ["quali_pos", "quali_gap", "fp_best", "fp_longrun", "sprint_pace", "profile_pace", "v1_model", "v2_model"]
LABEL = {"quali_pos": "Qualifying position", "quali_gap": "Qualifying gap", "fp_best": "Practice best lap",
         "fp_longrun": "Practice long run", "sprint_pace": "Sprint race pace", "profile_pace": "Driver+car profile",
         "v1_model": "Model v1", "v2_model": "Model v2 (with simulator)"}


def race_shape(d):
    """Pre-race shape of each race: one row per race."""
    rows = []
    circ = B.read("circuits").set_index("circuit_id") if len(B.read("circuits")) else pd.DataFrame()
    est = B.read("eras_overtakes").set_index("circuit_id") if len(B.read("eras_overtakes")) else pd.DataFrame()
    for (s, r), g in d.groupby(["season", "round"]):
        g = g.sort_values("quali_pos")
        q = g.dropna(subset=["quali_gap"]).sort_values("quali_gap")
        team = g.dropna(subset=["quali_gap"]).groupby("constructor_id")["quali_gap"].min().sort_values()
        fav = g.sort_values("v2_model").iloc[0] if g["v2_model"].notna().any() else g.iloc[0]
        top6 = g.nsmallest(6, "quali_pos")
        agree = top6[["quali_pos", "fp_longrun"]].dropna()
        cid = g["circuit_id"].iat[0]
        rows.append({
            "season": s, "round": r, "circuit_id": cid,
            "pole_margin": float(q["quali_gap"].iat[1]) if len(q) > 1 else np.nan,
            "team_margin": float(team.iat[1] - team.iat[0]) if len(team) > 1 else np.nan,
            "fav_pwin": float(fav.get("v2_model_pwin", np.nan)), "fav_start": float(fav["quali_pos"]),
            "fav_code": fav["code"], "fav_driver": fav["driver_id"],
            "fav_team_mate_gap": float(abs(g[g.constructor_id == fav.constructor_id]["quali_gap"].diff().iloc[-1]))
            if (g.constructor_id == fav.constructor_id).sum() == 2 else np.nan,
            "practice_agrees": float(agree["quali_pos"].corr(agree["fp_longrun"], method="spearman")) if len(agree) >= 4 else np.nan,
            "passing": float(est["estimate"].get(cid, np.nan)) if len(est) else np.nan,
            "sc_rate": float(circ["sc_rate"].get(cid, np.nan)) if len(circ) and "sc_rate" in circ else np.nan,
            "winner": g[g.position == 1]["code"].iat[0] if (g.position == 1).any() else None,
            "winner_start": float(g[g.position == 1]["quali_pos"].iat[0]) if (g.position == 1).any() else np.nan,
            "fav_won": bool((g.position == 1).any() and g[g.position == 1]["driver_id"].iat[0] == fav["driver_id"]),
        })
    return pd.DataFrame(rows)


SHAPE = ["pole_margin", "team_margin", "fav_pwin", "fav_start", "fav_team_mate_gap", "practice_agrees", "passing", "sc_rate"]


def regimes(sh, k_range=range(2, 5), seed=0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    X = sh[SHAPE].fillna(sh[SHAPE].median())
    Z = ((X - X.mean()) / (X.std() + 1e-9)).to_numpy()
    best = None
    for k in k_range:
        km = KMeans(k, n_init=30, random_state=seed).fit(Z)
        sc = silhouette_score(Z, km.labels_)
        if best is None or sc > best[0]:
            best = (sc, k, km)
    sc, k, km = best
    sh = sh.assign(regime=km.labels_)
    cen = pd.DataFrame(km.cluster_centers_, columns=SHAPE)
    names = {}
    for c in range(k):
        r = cen.loc[c]
        if r["fav_pwin"] > 0.5 and r["pole_margin"] > 0:
            names[c] = "clear favourite"
        elif r["sc_rate"] > 0.5 or r["passing"] > 0.5:
            names[c] = "chaos-prone or easy passing"
        elif r["team_margin"] < -0.3:
            names[c] = "close fight between teams"
        else:
            names[c] = "open"
    # make names unique
    seen = {}
    for c in names:
        seen[names[c]] = seen.get(names[c], 0) + 1
        if seen[names[c]] > 1:
            names[c] += f" {seen[names[c]]}"
    sh["regime_name"] = sh["regime"].map(names)
    return sh, {"k": int(k), "silhouette": round(float(sc), 3)}


def confidence(sh):
    """Walk-forward logistic model of P(favourite wins) from the race shape."""
    from sklearn.linear_model import LogisticRegression
    sh = sh.sort_values(["season", "round"]).reset_index(drop=True)
    feats = ["fav_pwin", "pole_margin", "team_margin", "fav_start", "fav_team_mate_gap", "passing", "sc_rate"]
    out = []
    for i, r in sh.iterrows():
        tr = sh.iloc[:i]
        if len(tr) < 25 or tr["fav_won"].nunique() < 2:
            out.append(np.nan)
            continue
        X = tr[feats].fillna(tr[feats].median())
        mu, sd = X.mean(), X.std() + 1e-9
        m = LogisticRegression(C=0.5, max_iter=1000).fit((X - mu) / sd, tr["fav_won"])
        x = pd.DataFrame([r[feats]]).fillna(tr[feats].median())
        out.append(float(m.predict_proba((x - mu) / sd)[0, 1]))
    sh["p_fav_wins"] = out
    X = sh[feats].fillna(sh[feats].median())
    Z = (X - X.mean()) / (X.std() + 1e-9)
    m = LogisticRegression(C=0.5, max_iter=1000).fit(Z, sh["fav_won"])
    coef = dict(zip(feats, np.round(m.coef_[0], 3)))
    # exact SHAP values of a linear model (log-odds): coefficient x (standardised value - mean of 0)
    shap = pd.DataFrame(Z.to_numpy() * m.coef_[0], columns=feats, index=sh.index)
    sh = sh.join(shap.add_prefix("shap_"))
    return sh, coef


def main():
    warnings.simplefilter("ignore")
    res = ingest.load_results()
    reg = registry.ensure_initialized()
    e1 = improve._eval_cached(reg["versions"]["v1"]["spec"], res)
    e2 = improve._eval_cached(registry.champion_spec(reg), res)
    d = signals(res, e1["preds"], e2["preds"])
    out = {"season": SEASON, "races": int(d[d.season == SEASON][["round"]].drop_duplicates().shape[0])}

    # 1. signals
    sig = {}
    for era_name, mask in (("2026", d.season == SEASON), ("2023-2025", d.season < SEASON)):
        x = d[mask]
        rows = []
        for (s, r), g in x.groupby(["season", "round"]):
            w = g[g.position == 1]
            if w.empty:
                continue
            row = {"season": s, "round": r}
            for k in SIGNALS:
                if g[k].notna().sum() >= 8 and w[k].notna().all():
                    row[k] = int((g[k] < w[k].iat[0]).sum() + 1)
            rows.append(row)
        t = pd.DataFrame(rows)
        sig[era_name] = {k: {"top1": float((t[k] == 1).mean()), "top3": float((t[k] <= 3).mean()),
                             "mean_rank": float(t[k].mean()), "races": int(t[k].notna().sum())}
                         for k in SIGNALS if k in t and t[k].notna().sum() >= 5}
    out["signals"] = sig

    # winners and grid in 2026 vs before
    w = res[(res.position == 1) & (res.season >= 2022)]
    out["winners_2026"] = w[w.season == SEASON][["round", "race_name", "code", "constructor_id", "grid", "quali_pos"]].to_dict("records")
    out["pole_conversion"] = {str(s): float((g["quali_pos"] == 1).mean()) for s, g in w.groupby("season")}
    out["wins_by_team_2026"] = w[w.season == SEASON]["constructor_id"].value_counts().to_dict()

    # 2-3. regimes and confidence on 2023+ (2026 reported separately)
    sh = race_shape(d)
    sh, reg_info = regimes(sh)
    sh, coef = confidence(sh)
    out["regimes"] = reg_info
    by = sh.groupby("regime_name").agg(races=("fav_won", "size"), fav_won=("fav_won", "mean"),
                                       fav_pwin=("fav_pwin", "mean"), races_2026=("season", lambda s: int((s == SEASON).sum())))
    out["regime_table"] = by.reset_index().to_dict("records")
    out["confidence_coef_per_sd"] = coef
    c = sh.dropna(subset=["p_fav_wins"])
    c["tier"] = pd.cut(c["p_fav_wins"], [0, 0.4, 0.6, 0.75, 1.0], labels=["<40%", "40-60%", "60-75%", ">75%"])
    tiers = c.groupby("tier", observed=True).agg(races=("fav_won", "size"), predicted=("p_fav_wins", "mean"),
                                                 fav_won=("fav_won", "mean")).reset_index()
    tiers["tier"] = tiers["tier"].astype(str)
    out["confidence_tiers"] = tiers.to_dict("records")
    c26 = c[c.season == SEASON]
    out["confidence_2026"] = {
        "brier_conf_model": float(((c26["p_fav_wins"] - c26["fav_won"]) ** 2).mean()) if len(c26) else None,
        "brier_v2_pwin": float(((c26["fav_pwin"] - c26["fav_won"]) ** 2).mean()) if len(c26) else None,
        "fav_hit_rate": float(c26["fav_won"].mean()) if len(c26) else None,
        "high_conf_hit_rate": float(c26[c26["p_fav_wins"] >= 0.6]["fav_won"].mean()) if (c26["p_fav_wins"] >= 0.6).any() else None,
        "high_conf_races": int((c26["p_fav_wins"] >= 0.6).sum()),
    }
    out["races_2026"] = sh[sh.season == SEASON].round(3).to_dict("records")

    # 4. teammates in the 2026 leading team
    top_team = max(out["wins_by_team_2026"], key=out["wins_by_team_2026"].get)
    tm = d[(d.season == SEASON) & (d.constructor_id == top_team)]
    rows = []
    for r, g in tm.groupby("round"):
        if len(g) != 2 or g["position"].isna().any():
            continue
        g = g.sort_values("position")
        ahead = g.iloc[0]
        row = {"round": int(r), "ahead": ahead["code"]}
        for k in SIGNALS:
            if g[k].notna().all():
                row[k] = bool(g[k].iat[0] < g[k].iat[1])
        rows.append(row)
    t = pd.DataFrame(rows)
    out["teammates"] = {"team": top_team, "races": int(len(t)),
                        "signal_picked_right": {k: float(t[k].mean()) for k in SIGNALS if k in t and t[k].notna().sum() >= 5}}
    (REPORTS / "analysis_2026.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("signals", "pole_conversion", "wins_by_team_2026", "regimes", "regime_table",
                                          "confidence_coef_per_sd", "confidence_tiers", "confidence_2026", "teammates")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
