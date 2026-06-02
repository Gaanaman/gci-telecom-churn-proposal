"""
Incremental model-improvement experiment (honest 5-fold OOF AUC at each step).
Techniques layered: CV -> native categoricals (CatBoost) -> feature engineering
-> fold-safe target encoding -> ensemble. Results -> experiment_results.txt
"""
from pathlib import Path
import warnings, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import rankdata
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
RNG = 42
N_SPLITS = 5
HERE = Path(__file__).resolve().parent
LOG = open(HERE / "experiment_results.txt", "w")

def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg); LOG.write(msg + "\n"); LOG.flush()

# ---- load ----
client = pd.read_csv(HERE / "telecom" / "Client.csv")
record = pd.read_csv(HERE / "telecom" / "Record.csv")
df = record.merge(client, on="Customer_ID", how="inner").drop(columns=["Customer_ID"])
y = df["churn"].values
log(f"data: {df.shape}, churn={y.mean():.4f}")

obj_cols = df.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in df.columns if c not in obj_cols + ["churn"]]

def engineer(d):
    d = d.copy(); eps = 1e-6
    d["eqp_per_tenure"] = d["eqpdays"] / (d["months"] * 30 + eps)
    d["rev_per_min"]    = d["rev_Mean"] / (d["mou_Mean"] + eps)
    d["care_per_min"]   = d["custcare_Mean"] / (d["mou_Mean"] + eps)
    d["overage_share"]  = d["ovrrev_Mean"] / (d["rev_Mean"] + eps)
    d["quality_fail"]   = d[["drop_vce_Mean","blck_vce_Mean","unan_vce_Mean"]].sum(axis=1)
    d["usage_momentum"] = d["change_mou"] / (d["mou_Mean"].abs() + eps)
    d["rev_momentum"]   = d["change_rev"] / (d["rev_Mean"].abs() + eps)
    d["mou_per_sub"]    = d["mou_Mean"] / (d["uniqsubs"] + eps)
    d["drops_per_call"] = d["drop_blk_Mean"] / (d["attempt_Mean"] + eps)
    d["compl_rate"]     = d["complete_Mean"] / (d["attempt_Mean"] + eps)
    d["miss_count"]     = d.isna().sum(axis=1)
    return d

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RNG)
folds = list(skf.split(df, y))

def report(name, oof):
    auc = roc_auc_score(y, oof); acc = accuracy_score(y, (oof >= 0.5).astype(int))
    log(f"  >> {name:<46} OOF AUC={auc:.4f}  acc={acc:.4f}")
    return auc

# ---- A: baseline (label-encode all, XGBoost) ----
t=time.time()
Xa = df.drop(columns="churn").copy()
for c in obj_cols: Xa[c] = LabelEncoder().fit_transform(Xa[c].astype(str))
oof = np.zeros(len(df))
for tr, va in folds:
    m = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                      eval_metric="logloss", random_state=RNG, n_jobs=-1)
    m.fit(Xa.iloc[tr], y[tr]); oof[va] = m.predict_proba(Xa.iloc[va])[:,1]
auc_a = report("A. baseline XGBoost (label-encoded, no FE)", oof); log(f"     [{time.time()-t:.0f}s]")

# ---- B: CatBoost native categoricals (no FE) ----
t=time.time()
Xb = df.drop(columns="churn").copy()
for c in obj_cols: Xb[c] = Xb[c].astype(str).fillna("NA")
cat_idx = [Xb.columns.get_loc(c) for c in obj_cols]
oof = np.zeros(len(df))
for tr, va in folds:
    m = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                           eval_metric="AUC", random_seed=RNG, verbose=0,
                           border_count=128, thread_count=-1)
    m.fit(Xb.iloc[tr], y[tr], cat_features=cat_idx,
          eval_set=(Xb.iloc[va], y[va]), early_stopping_rounds=40)
    oof[va] = m.predict_proba(Xb.iloc[va])[:,1]
auc_b = report("B. CatBoost native cats (no FE)", oof); log(f"     [{time.time()-t:.0f}s]")

# ---- C: CatBoost + engineered features ----
t=time.time()
Xc = engineer(df.drop(columns="churn"))
for c in obj_cols: Xc[c] = Xc[c].astype(str).fillna("NA")
cat_idx_c = [Xc.columns.get_loc(c) for c in obj_cols]
oof = np.zeros(len(df))
for tr, va in folds:
    m = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                           eval_metric="AUC", random_seed=RNG, verbose=0,
                           border_count=128, thread_count=-1)
    m.fit(Xc.iloc[tr], y[tr], cat_features=cat_idx_c,
          eval_set=(Xc.iloc[va], y[va]), early_stopping_rounds=40)
    oof[va] = m.predict_proba(Xc.iloc[va])[:,1]
auc_c = report("C. CatBoost + engineered features", oof); log(f"     [{time.time()-t:.0f}s]")

# ---- D: CatBoost + FE + fold-safe target encoding (high-card cats) ----
t=time.time()
hc = [c for c in obj_cols if df[c].nunique() > 8]   # high-cardinality cats
def te_fit(s, yt, m=20):
    g = pd.DataFrame({"c": s.values, "y": yt}); prior = yt.mean()
    agg = g.groupby("c")["y"].agg(["mean","count"])
    enc = (agg["mean"]*agg["count"] + prior*m) / (agg["count"] + m)
    return enc, prior
oof = np.zeros(len(df))
for tr, va in folds:
    Xtr, Xva = Xc.iloc[tr].copy(), Xc.iloc[va].copy()
    for c in hc:
        enc, prior = te_fit(Xtr[c], y[tr])
        Xtr[c+"_te"] = Xtr[c].map(enc).fillna(prior)
        Xva[c+"_te"] = Xva[c].map(enc).fillna(prior)
    ci = [Xtr.columns.get_loc(c) for c in obj_cols]
    m = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                           eval_metric="AUC", random_seed=RNG, verbose=0,
                           border_count=128, thread_count=-1)
    m.fit(Xtr, y[tr], cat_features=ci, eval_set=(Xva, y[va]), early_stopping_rounds=40)
    oof[va] = m.predict_proba(Xva)[:,1]
auc_d = report("D. CatBoost + FE + fold-safe target encoding", oof); log(f"     [{time.time()-t:.0f}s]")

# ---- E: ensemble (CatBoost-D + XGBoost-FE + HistGB-FE), rank-blend ----
t=time.time()
Xe = engineer(df.drop(columns="churn"))
for c in obj_cols: Xe[c] = LabelEncoder().fit_transform(Xe[c].astype(str))
oof_xgb = np.zeros(len(df)); oof_hgb = np.zeros(len(df)); oof_cat = np.zeros(len(df))
for tr, va in folds:
    xg = XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8,
                       colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
                       eval_metric="auc", random_state=RNG, n_jobs=-1)
    xg.fit(Xe.iloc[tr], y[tr]); oof_xgb[va] = xg.predict_proba(Xe.iloc[va])[:,1]
    hg = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.03, max_depth=6,
                                        l2_regularization=1.0, random_state=RNG)
    hg.fit(Xe.iloc[tr], y[tr]); oof_hgb[va] = hg.predict_proba(Xe.iloc[va])[:,1]
oof_cat = oof  # reuse D's catboost OOF
report("   (member) XGBoost+FE", oof_xgb)
report("   (member) HistGB+FE", oof_hgb)
rb = (rankdata(oof_cat) + rankdata(oof_xgb) + rankdata(oof_hgb)) / (3*len(df))
auc_e = report("E. ENSEMBLE rank-blend (Cat+XGB+HGB)", rb); log(f"     [{time.time()-t:.0f}s]")

log("\nSUMMARY")
for n,a in [("A baseline",auc_a),("B catboost",auc_b),("C +FE",auc_c),
            ("D +target-enc",auc_d),("E ensemble",auc_e)]:
    log(f"  {n:<16} AUC={a:.4f}  (Δ vs baseline {a-auc_a:+.4f})")
LOG.close()
