"""
Statistics for the TP / FP / FN event comparison (power_analysis.ipynb).

Design: events are nested in sessions and sessions in animals; every session contributes TP, FP
and FN events, so event class varies WITHIN session. Contrasts are therefore estimated within
sessions, and session/animal random effects absorb between-session differences.

- fit_nested_lmm: metric ~ fixed + (1|Animal) + (1|Animal:Session) [+ (0 + slope|Animal:Session)]
  (same parametrisation as Madrid_analysis daq/analysis/buffer_stats.py).
- pairwise_event_lmm: event-level LMM per metric and pair of classes, with a per-session random
  slope for the class effect; metric in SD units (for the forest plot) and raw units; Holm across
  metrics x pairs. Cliff's delta kept as descriptive.
- session_paired_wilcoxon: model-free check on session medians (paired by session), Holm.
- hierarchical_bootstrap_trace: CI of event-triggered mean traces (resample sessions, then events).
- loso_separability: leave-one-session-out classifier AUC for each pair of classes.
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import calculate_matched_pairs_effect_size_wilcoxon, cliffs_delta, holm_bonferroni

PAIRS = (("TP", "FP"), ("TP", "FN"), ("FP", "FN"))
PAIR_COLORS = {"TP − FP": "olive", "TP − FN": "teal", "FP − FN": "purple"}


# ---------------------------------------------------------------------------
# Animals
def load_animals(csv_path):
    """Session -> animal mapping from Animals.csv (columns: Animal, Session)."""
    df = pd.read_csv(csv_path)
    return dict(zip(df["Session"].astype(str).str.strip(), df["Animal"].astype(int)))


def add_animal(df, animals):
    out = df.copy()
    out["Animal"] = out["Session"].astype(str).map(animals)
    missing = sorted(out.loc[out["Animal"].isna(), "Session"].unique())
    if missing:
        print(f"Warning: no animal for sessions {missing}, dropped.")
        out = out.dropna(subset=["Animal"])
    out["Animal"] = out["Animal"].astype(int)
    return out


# ---------------------------------------------------------------------------
# Mixed models
def fit_nested_lmm(df, formula, slope=None, reml=True, verbose=False):
    """
    metric ~ fixed + (1|Animal) + (1|Animal:Session) [+ (0 + slope|Animal:Session)].

    statsmodels: groups=Animal with re_formula="1" (animal intercept) and variance components for
    the session intercept (and optional uncorrelated session slope). Falls back from lbfgs to powell.
    """
    df = df.reset_index(drop=True)
    vc = {"Session": "0 + C(Session)"}
    if slope is not None:
        vc["Session_slope"] = f"0 + C(Session):{slope}"
    model = smf.mixedlm(formula, df, groups=df["Animal"], re_formula="1", vc_formula=vc)
    res = None
    for method in ("lbfgs", "powell"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = model.fit(reml=reml, method=method)
            except Exception as e:
                if verbose:
                    print(f"  fit with {method} failed: {e}")
                continue
        if res.converged:
            break
    return res


def lmm_variances(res):
    """Animal, session (and session-slope) variances and residual variance."""
    out = {"Animal_var": float(res.cov_re.iloc[0, 0])}
    out.update({f"{n}_var": float(v) for n, v in zip(res.model.exog_vc.names, res.vcomp)})
    out["Residual_var"] = float(res.scale)
    return out


def pairwise_event_lmm(df, metrics, pairs=PAIRS, random_slope=True, alpha=0.05, verbose=True):
    """
    For each metric and pair (A, B): y ~ is_A + (1|Animal) + (1 + is_A || Animal:Session) on events
    of classes A and B. The coefficient of is_A is A - B. The metric is standardised over all events
    in `df` (SD units, comparable across metrics); the raw-unit estimate is Beta_SD * SD.
    If the random-slope model does not converge, the intercept-only model is used (column RE).
    """
    rows = []
    for metric in metrics:
        d_all = df.dropna(subset=[metric])
        mu, sd = d_all[metric].mean(), d_all[metric].std()
        for a, b in pairs:
            sub = d_all[d_all["Event_Type"].isin([a, b])].copy()
            sub["y"] = (sub[metric] - mu) / sd
            sub["is_A"] = (sub["Event_Type"] == a).astype(float)
            res, re_used = None, None
            if random_slope:
                res = fit_nested_lmm(sub, "y ~ is_A", slope="is_A")
                re_used = "intercepts + session slope"
            if res is None or not res.converged:
                res = fit_nested_lmm(sub, "y ~ is_A")
                re_used = "intercepts only"
            if res is None:
                print(f"  {metric} {a}-{b}: LMM failed")
                continue
            ci = res.conf_int().loc["is_A"]
            va, vb = sub.loc[sub["is_A"] == 1, metric], sub.loc[sub["is_A"] == 0, metric]
            u, _ = mannwhitneyu(va, vb, alternative="two-sided")
            delta, delta_label = cliffs_delta(u, len(va), len(vb))
            var = lmm_variances(res)
            rows.append({
                "Metric": metric,
                "Contrast": f"{a} − {b}",
                f"Median {a}": va.median(),
                f"Median {b}": vb.median(),
                "Beta": res.fe_params["is_A"] * sd,
                "CI_low": ci[0] * sd,
                "CI_high": ci[1] * sd,
                "Beta_SD": res.fe_params["is_A"],
                "CI_low_SD": ci[0],
                "CI_high_SD": ci[1],
                "z": res.tvalues["is_A"],
                "p": res.pvalues["is_A"],
                "Cliff_delta": delta,
                "Cliff_label": delta_label,
                "RE": re_used,
                "Converged": res.converged,
                "Session_slope_SD": np.sqrt(var["Session_slope_var"]) if "Session_slope_var" in var else np.nan,
                "n_A": len(va),
                "n_B": len(vb),
                "n_sessions": sub["Session"].nunique(),
                "n_animals": sub["Animal"].nunique(),
            })
    out = pd.DataFrame(rows)
    _, out["p_holm"] = holm_bonferroni(out["p"].values, alpha=alpha)
    out["Significant"] = out["p_holm"] < alpha
    if verbose:
        n_fallback = (out["RE"] == "intercepts only").sum()
        if random_slope and n_fallback:
            print(f"{n_fallback}/{len(out)} models fell back to random intercepts only.")
    return out


# ---------------------------------------------------------------------------
# Session-level, model-free check
def session_paired_wilcoxon(df, metrics, pairs=PAIRS, agg="median", alpha=0.05):
    """
    Wilcoxon signed-rank on per-session summaries (paired by session), Holm across metrics x pairs.
    Effect size: matched-pairs rank-biserial (positive when A > B).
    """
    rows = []
    for metric in metrics:
        tab = df.groupby(["Session", "Event_Type"])[metric].agg(agg).unstack()
        for a, b in pairs:
            paired = tab[[a, b]].dropna()
            try:
                _, p = wilcoxon(paired[a], paired[b])
            except ValueError:
                p = np.nan
            rc, rc_label = calculate_matched_pairs_effect_size_wilcoxon(paired[a], paired[b])
            rows.append({
                "Metric": metric,
                "Contrast": f"{a} − {b}",
                f"Session {agg} {a}": paired[a].median(),
                f"Session {agg} {b}": paired[b].median(),
                "Median paired diff": (paired[a] - paired[b]).median(),
                "Sessions A > B": f"{(paired[a] > paired[b]).sum()}/{len(paired)}",
                "p": p,
                "r_rb": rc,
                "r_rb_label": rc_label,
                "n_sessions": len(paired),
            })
    out = pd.DataFrame(rows)
    _, out["p_holm"] = holm_bonferroni(out["p"].fillna(1).values, alpha=alpha)
    out["Significant"] = out["p_holm"] < alpha
    return out


# ---------------------------------------------------------------------------
# Event-triggered traces
def hierarchical_bootstrap_trace(traces, sessions, n_boot=1000, ci=0.95, seed=0):
    """
    Mean trace and percentile CI, resampling sessions and then events within each session.
    traces: (n_events, n_time) linear values (e.g. power / background).
    Returns (mean, lower, upper), all linear (convert to dB afterwards: monotone, so CIs carry over).
    """
    rng = np.random.default_rng(seed)
    traces = np.asarray(traces)
    sessions = np.asarray(sessions)
    groups = [np.where(sessions == s)[0] for s in np.unique(sessions)]
    boots = np.empty((n_boot, traces.shape[1]))
    for i in range(n_boot):
        picked = rng.integers(len(groups), size=len(groups))
        idx = np.concatenate([rng.choice(groups[g], size=len(groups[g])) for g in picked])
        boots[i] = traces[idx].mean(axis=0)
    lower, upper = np.percentile(boots, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2], axis=0)
    return traces.mean(axis=0), lower, upper


def session_zscore_map(properties_df):
    """
    Per-session affine map from power / background (linear) to the legacy z-scored power
    (z-scored over the whole session): z = (P - mean) / std = (background / std) * (P / background) - mean / std.
    Recovered exactly from Peak_Power_Z vs Peak_Power_dB of the events of each session.
    Returns {session: (slope, intercept)}.
    """
    out = {}
    for session, d in properties_df.groupby("Session"):
        ratio = 10 ** (d["Peak_Power_dB"].values / 10)
        slope, intercept = np.polyfit(ratio, d["Peak_Power_Z"].values, 1)
        out[session] = (slope, intercept)
    return out


# ---------------------------------------------------------------------------
# Separability
def loso_separability(df, feature_sets, pairs=PAIRS, n_boot=1000, ci=0.95, seed=0, return_predictions=False):
    """
    Leave-one-session-out logistic regression (standardised features, balanced class weights) for
    each pair of classes and feature set. AUC and balanced accuracy on the pooled out-of-fold
    predictions; CI by bootstrapping sessions. AUC = 0.5 -> not separable, 1 -> perfectly separable.
    return_predictions: also return {(feature set, contrast): (y, prob)} for plot_roc_curves.
    """
    rng = np.random.default_rng(seed)
    rows = []
    predictions = {}
    for set_name, features in feature_sets.items():
        for a, b in pairs:
            sub = df[df["Event_Type"].isin([a, b])].dropna(subset=features).reset_index(drop=True)
            X, y, groups = sub[features].values, (sub["Event_Type"] == a).astype(int).values, sub["Session"].values
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
            prob = cross_val_predict(clf, X, y, groups=groups, cv=LeaveOneGroupOut(), method="predict_proba")[:, 1]
            auc = roc_auc_score(y, prob)
            bacc = balanced_accuracy_score(y, prob >= 0.5)
            session_idx = [np.where(groups == s)[0] for s in np.unique(groups)]
            boot = []
            for _ in range(n_boot):
                idx = np.concatenate([session_idx[i] for i in rng.integers(len(session_idx), size=len(session_idx))])
                if len(np.unique(y[idx])) == 2:
                    boot.append(roc_auc_score(y[idx], prob[idx]))
            lo, hi = np.percentile(boot, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
            predictions[(set_name, f"{a} vs {b}")] = (y, prob)
            rows.append({
                "Features": set_name,
                "Contrast": f"{a} vs {b}",
                "AUC": auc,
                "AUC_CI_low": lo,
                "AUC_CI_high": hi,
                "Balanced_accuracy": bacc,
                "n_A": int(y.sum()),
                "n_B": int(len(y) - y.sum()),
                "n_sessions": len(session_idx),
            })
    if return_predictions:
        return pd.DataFrame(rows), predictions
    return pd.DataFrame(rows)


def plot_roc_curves(predictions, sep_df=None, feature_sets=None, pairs=PAIRS, colors=None, figsize=(15, 5)):
    """
    ROC curves (TPR vs FPR) of the leave-one-session-out predictions, one panel per pair of classes
    and one curve per feature set. The first class of each pair is the positive class.
    sep_df (from loso_separability) adds the AUC and its session-bootstrap CI to the legend.
    """
    from sklearn.metrics import roc_curve

    if feature_sets is None:
        feature_sets = list(dict.fromkeys(k[0] for k in predictions))
    if colors is None:
        colors = dict(zip(feature_sets, ["black", "tab:orange", "tab:cyan", "tab:pink", "tab:brown"]))
    contrasts = [f"{a} vs {b}" for a, b in pairs]
    fig, axes = plt.subplots(1, len(contrasts), figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, contrast in zip(axes, contrasts):
        for set_name in feature_sets:
            if (set_name, contrast) not in predictions:
                continue
            y, prob = predictions[(set_name, contrast)]
            fpr, tpr, _ = roc_curve(y, prob)
            label = set_name
            if sep_df is not None:
                r = sep_df[(sep_df["Features"] == set_name) & (sep_df["Contrast"] == contrast)].iloc[0]
                label += f" (AUC {r['AUC']:.2f} [{r['AUC_CI_low']:.2f}, {r['AUC_CI_high']:.2f}])"
            ax.plot(fpr, tpr, color=colors.get(set_name), lw=2, label=label)
        ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--")
        a, b = contrast.split(" vs ")
        ax.set_title(f"{a} vs {b}", fontsize=16)
        ax.set_xlabel(f"False positive rate ({b} called {a})", fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=9, frameon=False, loc="lower right")
        ax.tick_params(axis="both", which="major", labelsize=12, width=2)
        ax.spines[["left", "bottom"]].set_linewidth(2)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("True positive rate", fontsize=14)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Forest plot
def p_to_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def plot_forest(lmm_df, metric_labels=None, pair_colors=PAIR_COLORS, ax=None, figsize=(6, 6),
                p_col="p_holm", alpha=0.05):
    """
    Standardised LMM effects (A - B, SD units) with 95% CI per metric and contrast.
    Filled markers: significant after Holm; open markers: not significant.
    """
    metrics = list(dict.fromkeys(lmm_df["Metric"]))
    contrasts = [c for c in pair_colors if c in set(lmm_df["Contrast"])]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    offsets = np.linspace(-0.25, 0.25, len(contrasts))
    x_max = lmm_df["CI_high_SD"].max()
    for j, contrast in enumerate(contrasts):
        for i, metric in enumerate(metrics):
            row = lmm_df[(lmm_df["Metric"] == metric) & (lmm_df["Contrast"] == contrast)]
            if row.empty:
                continue
            row = row.iloc[0]
            y = i + offsets[j]
            c = pair_colors[contrast]
            ax.plot([row["CI_low_SD"], row["CI_high_SD"]], [y, y], color=c, lw=2)
            ax.plot(row["Beta_SD"], y, "o", ms=7, color=c,
                    markerfacecolor=c if row[p_col] < alpha else "white", markeredgewidth=2,
                    label=contrast if i == 0 else None)
            ax.text(x_max + 0.08, y, p_to_stars(row[p_col]), va="center", fontsize=10, color=c)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([metric_labels.get(m, m) if metric_labels else m for m in metrics], fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel("Effect [SD units] (LMM, 95% CI)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=13, width=2)
    ax.spines[["left", "bottom"]].set_linewidth(2)
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(fontsize=12, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(contrasts))
    fig.tight_layout()
    return fig, ax
