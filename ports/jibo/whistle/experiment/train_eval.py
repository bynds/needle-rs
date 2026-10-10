"""Template classifier over Whistle encoder embeddings (needle_embed: one 512-wide row per 80 ms).

Train on synthetic sets, test on real speech. Reports forced-choice accuracy, open-set accuracy
(with a "none" class), and the accept/error/false-accept trade-off over confidence thresholds.

usage: train_eval.py TRAIN_SETS TEST_SETS [--clf logreg|mlp] [--out model.npz]
  sets are comma-separated names; each NAME.jsonl lives in gen/ or real/ and its clips' .emb beside them
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

W = "/tmp/claude-0/w"
D = 512


def load(name, limit=None):
    for base in ("gen", "real"):
        p = f"{W}/{base}/{name}.jsonl"
        if os.path.exists(p):
            break
    X, y, meta = [], [], []
    for line in open(p):
        r = json.loads(line)
        e = r["path"][:-4] + ".emb"
        if not os.path.exists(e):
            continue
        f = np.fromfile(e, dtype=np.float32).reshape(-1, D)
        X.append(features(f))
        y.append(r["label"])
        meta.append(r)
        if limit and len(X) >= limit:
            break
    return np.array(X), np.array(y), meta


def features(f):
    # mean, max and standard deviation over frames: a fixed-size summary of any clip length
    return np.concatenate([f.mean(0), f.max(0), f.std(0)])


def report(name, clf, sc, X, y, classes):
    P = clf.predict_proba(sc.transform(X))
    none = list(classes).index("none") if "none" in classes else None
    pred = classes[P.argmax(1)]
    out = {"set": name, "n": len(y)}
    pos = y != "none"
    if pos.any():
        Pc = P.copy()
        if none is not None:
            Pc[:, none] = -1
        forced = classes[Pc.argmax(1)]
        out["forced_acc"] = float((forced[pos] == y[pos]).mean())
        out["open_acc"] = float((pred[pos] == y[pos]).mean())
        out["rejected_as_none"] = float((pred[pos] == "none").mean())
    if (~pos).any():
        out["neg_false_accept"] = float((pred[~pos] != "none").mean())
    # threshold sweep on the top non-none probability
    conf = P.max(1)
    rows = []
    for t in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        acc_mask = (pred != "none") & (conf >= t)
        r = {"t": t}
        if pos.any():
            a = acc_mask[pos]
            r["accept"] = float(a.mean())
            r["err_accepted"] = float((pred[pos][a] != y[pos][a]).mean()) if a.any() else 0.0
        if (~pos).any():
            r["false_accept"] = float(acc_mask[~pos].mean())
        rows.append(r)
    out["sweep"] = rows
    print(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("test")
    ap.add_argument("--clf", default="logreg")
    ap.add_argument("--out")
    a = ap.parse_args()
    Xs, ys = [], []
    for n in a.train.split(","):
        X, y, _ = load(n)
        print(f"train {n}: {len(y)}", file=sys.stderr)
        Xs.append(X)
        ys.append(y)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    sc = StandardScaler().fit(X)
    if a.clf == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=300, early_stopping=True, random_state=0)
    else:
        clf = LogisticRegression(C=0.5, max_iter=3000)
    clf.fit(sc.transform(X), y)
    for n in a.test.split(","):
        Xt, yt, _ = load(n)
        report(n, clf, sc, Xt, yt, clf.classes_)
    if a.out:
        np.savez(a.out, classes=clf.classes_, mean=sc.mean_, scale=sc.scale_,
                 **({"coef": clf.coef_, "intercept": clf.intercept_} if a.clf == "logreg" else {}))


if __name__ == "__main__":
    main()
