"""NSL-KDD binary intrusion detection: MI + MLP baseline vs LightGBM-importance + balanced LightGBM.

Split: official KDDTrain+ (train) / KDDTest+ (held-out test), used once for final scoring.
Tuning: none. Hyper-parameters are fixed, moderate and chosen a priori; no test-set selection.
"""
import io
import json
import os
import sys
import time

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEEDS = [42, 43, 44]          # randomised models are run with each seed
K = 20                        # features kept by each selector
MAX_TRAIN = 200_000           # subsample guard for the runtime budget
BASE = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
TRAIN_URL, TEST_URL = BASE + "KDDTrain+.txt", BASE + "KDDTest+.txt"
FEATS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
         "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
         "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
         "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
         "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
         "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
         "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
         "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
         "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]
CAT = ["protocol_type", "service", "flag"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch(url, tries=5):
    """Download text; retry with exponential backoff on HTTP 429 / 5xx and network errors."""
    err = "unknown"
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code != 429 and r.status_code < 500:
                r.raise_for_status()  # non-retryable 4xx errors propagate immediately
                return r.text
            err = f"HTTP {r.status_code}"
        except (requests.ConnectionError, requests.Timeout) as e:
            err = str(e)
        if i < tries - 1:
            log(f"download issue ({err}); retrying in {2 ** i}s")
            time.sleep(2 ** i)
    raise RuntimeError(f"could not download {url}: {err}")


def load(url):
    """Read a headerless NSL-KDD file and binarise the label (0 = normal, 1 = any attack)."""
    df = pd.read_csv(io.StringIO(fetch(url)), header=None, names=FEATS + ["label", "difficulty"])
    df["y"] = (df["label"] != "normal").astype(int)
    return df.drop(columns=["label", "difficulty"])


def preprocess(tr, te):
    """One-hot (fit on train only) for categoricals; standardise numerics (fit on train only)."""
    num = [c for c in FEATS if c not in CAT]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(tr[CAT])
    sc = StandardScaler().fit(tr[num])

    def transform(d):
        return np.hstack([sc.transform(d[num]), enc.transform(d[CAT])])

    names = num + list(enc.get_feature_names_out(CAT))
    is_discrete = np.arange(len(names)) >= len(num)  # one-hot columns are discrete
    return (transform(tr), tr["y"].values, transform(te), te["y"].values, names, is_discrete)


def evaluate(model, X_te, y_te, cols):
    """Shared evaluation: main metric macro-F1, plus accuracy, attack recall, attack F1."""
    pred = model.predict(X_te[:, cols])
    return {"macro_f1": f1_score(y_te, pred, average="macro"),
            "accuracy": accuracy_score(y_te, pred),
            "attack_recall": recall_score(y_te, pred),
            "attack_f1": f1_score(y_te, pred)}


def main():
    log("downloading NSL-KDD (raw text, official train/test split)")
    tr_full, te = load(TRAIN_URL), load(TEST_URL)
    n_rows = len(tr_full) + len(te)
    tr = tr_full.sample(MAX_TRAIN, random_state=42) if len(tr_full) > MAX_TRAIN else tr_full
    X_tr, y_tr, X_te, y_te, names, is_discrete = preprocess(tr, te)
    log(f"train {X_tr.shape}, test {X_te.shape}, test attack rate {y_te.mean():.3f}")

    res = {"baseline": [], "improved": []}
    for seed in SEEDS:
        # Baseline: top-K mutual information + MLP (no class weighting)
        log(f"seed {seed}: baseline = mutual information top-{K} + MLPClassifier")
        mi = mutual_info_classif(X_tr, y_tr, discrete_features=is_discrete, random_state=seed)
        cols_b = np.argsort(-mi)[:K]
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True,
                            random_state=seed)
        mlp.fit(X_tr[:, cols_b], y_tr)
        res["baseline"].append(evaluate(mlp, X_te, y_te, cols_b))

        # Improved: top-K LightGBM gain importance + class-balanced LightGBM
        log(f"seed {seed}: improved = LightGBM gain top-{K} + balanced LightGBM")
        sel = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31,
                                 importance_type="gain", random_state=seed, verbose=-1)
        sel.fit(X_tr, y_tr)
        cols_i = np.argsort(-sel.feature_importances_)[:K]
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 class_weight="balanced", random_state=seed, verbose=-1)
        clf.fit(X_tr[:, cols_i], y_tr)
        res["improved"].append(evaluate(clf, X_te, y_te, cols_i))
        log(f"seed {seed}: baseline macro-F1 {res['baseline'][-1]['macro_f1']:.4f}, "
            f"improved macro-F1 {res['improved'][-1]['macro_f1']:.4f}")

    # Aggregate over seeds
    m = {}
    for k in ("baseline", "improved"):
        scores = [r["macro_f1"] for r in res[k]]
        m[f"{k}_score"] = float(np.mean(scores))
        m[f"{k}_score_std"] = float(np.std(scores))
        m[f"{k}_accuracy"] = float(np.mean([r["accuracy"] for r in res[k]]))
        m[f"{k}_attack_recall"] = float(np.mean([r["attack_recall"] for r in res[k]]))
        m[f"{k}_f1"] = float(np.mean([r["attack_f1"] for r in res[k]]))
    m["improvement"] = m["improved_score"] - m["baseline_score"]
    if not all(np.isfinite(v) for v in m.values()):
        log("non-finite metrics; failing")
        sys.exit(1)

    # Chart: macro-F1 baseline vs improved (mean +/- std over seeds)
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["baseline\nMI + MLP", "improved\nLGBM sel. + bal. LGBM"],
           [m["baseline_score"], m["improved_score"]],
           yerr=[m["baseline_score_std"], m["improved_score_std"]], capsize=5,
           color=["#999999", "#3b7dd8"])
    ax.set_ylabel("test macro-F1")
    ax.set_title("NSL-KDD test macro-F1 (mean over seeds)")
    fig.tight_layout()
    fig.savefig("figures/macro_f1.png", dpi=120)
    plt.close(fig)

    results = {
        "question": "Can an embedded LightGBM-importance feature selection with a class-balanced "
                    "LightGBM classifier beat a standard mutual-information selection with an MLP "
                    "for binary intrusion detection on NSL-KDD?",
        "method": "Official NSL-KDD train/test files. Binary label (normal vs attack). One-hot "
                  "categoricals and standardised numerics fitted on train only. Baseline: top-K "
                  "mutual information + MLP. Improved: top-K LightGBM gain importance + class-"
                  "balanced LightGBM. Fixed hyper-parameters, no tuning. Test set used once. "
                  "Randomised models run with seeds 42, 43, 44.",
        "dataset": {
            "name": "NSL-KDD",
            "source_url": TRAIN_URL + " ; " + TEST_URL,
            "n_rows": int(n_rows),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "macro-F1 on the KDDTest+ test split",
        "higher_is_better": True,
        "baseline": f"Top-{K} features by mutual information, then scikit-learn MLPClassifier "
                    "(hidden layers 64 and 32, early stopping). Stands in for the paper's pipeline.",
        "improved": f"Top-{K} features by LightGBM gain importance (fit on train), then a "
                    "class-balanced LightGBMClassifier with fixed moderate hyper-parameters.",
        "metrics": m,
        "notes": "The test split is the official KDDTest+ file, not a random 20-30% split; it "
                 "contains attack types and distribution shift absent from training, so scores "
                 "reflect generalisation to novel attacks. Hyper-parameters were fixed a priori "
                 "and not tuned, so no cross-validation selection was performed. The MLP has no "
                 "class weighting. The paper's CircleSA optimiser and FNP-SE network were not "
                 "reproduced; the baseline is a generic stand-in. Only 3 seeds were run and "
                 "std is the population std over them. NSL-KDD is an old benchmark.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"baseline macro-F1 {m['baseline_score']:.4f} +/- {m['baseline_score_std']:.4f}; "
        f"improved {m['improved_score']:.4f} +/- {m['improved_score_std']:.4f}; "
        f"improvement {m['improvement']:+.4f}")
    log("wrote results.json and figures/macro_f1.png")


if __name__ == "__main__":
    main()
