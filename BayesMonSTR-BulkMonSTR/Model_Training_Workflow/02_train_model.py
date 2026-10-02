#!/usr/bin/env python3
"""Train a BulkMonSTR random forest (RF) classifier on labeled loci.

Same recipe as the published Illumina models and the Element models:
  Pipeline(SimpleImputer(mean) -> RandomForestClassifier(random_state=666)),
  grid search over n_estimators {50, 100, 200} x max_depth {None, 10, 20},
  10-fold cross-validation, accuracy as the selection score.
Train/test split and cross-validation folds are grouped by STR locus (default column STR_ID), so
that the same locus (e.g. from several samples or depths) is never in both training and test data.

Input
  --features  model_features_{indel,mismatch}.csv.gz from 01_build_model_features.py
              (columns sample_name, STR_ID, Label + model features, in model order)
  --out       output prefix, e.g. models/MyPlatform_RF_indel

Outputs
  <out>.pkl                 trained model (scikit-learn Pipeline), used with
                            BulkMonSTR_prediction.py -mid (indel) / -mm (mismatch)
  <out>_features.txt        model features, in order
  <out>_grid_search.tsv     cross-validation results
  <out>_test_metrics.tsv    precision / recall / F1 per class on the held-out loci
  <out>_test_confusion.tsv  confusion matrix on the held-out loci

By default the saved model is fitted on the training loci only (as for the Element models), and
the held-out loci are used for evaluation. With --refit-all, the best parameters are refitted on
all loci after evaluation.

Example
  python3 02_train_model.py --features features/model_features_indel.csv.gz --out models/My_RF_indel
"""
import argparse
import os
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline

META = ["sample_name", "STR_ID", "Label"]
CLASSES = ["Artifacts", "Germline Het", "Mosaic mutations"]


def make_pipeline(features):
    pre = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="mean"))]), features)])
    return Pipeline([("preprocessor", pre), ("classifier", RandomForestClassifier(random_state=666))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--group-col", default="STR_ID", help="column used to group loci (default STR_ID)")
    ap.add_argument("--test-size", type=float, default=0.2, help="fraction of held-out loci (default 0.2)")
    ap.add_argument("--folds", type=int, default=10, help="cross-validation folds (default 10)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--refit-all", action="store_true", help="refit the best model on all loci")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    d = pd.read_csv(a.features)
    feats = [c for c in d.columns if c not in META]
    y, groups = d["Label"], d[a.group_col]
    missing = set(CLASSES) - set(y)
    if missing:
        raise ValueError(f"no training loci for {missing}; all three classes are required")
    print("loci", len(d), y.value_counts().to_dict(), "features", len(feats))

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=a.test_size, random_state=42).split(d, y, groups))
    gs = GridSearchCV(make_pipeline(feats),
                      {"classifier__n_estimators": [50, 100, 200],
                       "classifier__max_depth": [None, 10, 20]},
                      cv=GroupKFold(n_splits=a.folds), scoring="accuracy", n_jobs=a.threads, refit=True)
    gs.fit(d.iloc[tr][feats], y.iloc[tr], groups=groups.iloc[tr])
    print("best", gs.best_params_, "CV accuracy", round(gs.best_score_, 4))
    pd.DataFrame(gs.cv_results_).to_csv(f"{a.out}_grid_search.tsv", sep="\t", index=False)

    model = gs.best_estimator_
    pred = model.predict(d.iloc[te][feats])
    pr, rc, f1, n = precision_recall_fscore_support(y.iloc[te], pred, labels=CLASSES, zero_division=0)
    met = pd.DataFrame({"class": CLASSES, "n": n, "precision": pr, "recall": rc, "F1": f1})
    met.to_csv(f"{a.out}_test_metrics.tsv", sep="\t", index=False)
    cm = pd.DataFrame(confusion_matrix(y.iloc[te], pred, labels=CLASSES), index=CLASSES, columns=CLASSES)
    cm.index.name = "true \\ predicted"
    cm.to_csv(f"{a.out}_test_confusion.tsv", sep="\t")
    print("held-out loci", len(te)); print(met.round(3).to_string(index=False))

    if a.refit_all:
        model = make_pipeline(feats).set_params(**gs.best_params_).fit(d[feats], y)
        print("refitted on all", len(d), "loci")
    with open(f"{a.out}.pkl", "wb") as fh:
        pickle.dump(model, fh)
    open(f"{a.out}_features.txt", "w").write("\n".join(feats) + "\n")
    print("saved", f"{a.out}.pkl")


if __name__ == "__main__":
    main()
