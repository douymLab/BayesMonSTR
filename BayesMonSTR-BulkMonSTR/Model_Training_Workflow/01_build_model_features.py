#!/usr/bin/env python3
"""Build random forest (RF) model features for labeled BulkMonSTR candidate loci.

The feature engineering is not re-implemented here: the relevant code blocks (column names,
derived features, VAF correction, process_input and the model feature orders) are read from
the BulkMonSTR prediction script (BulkMonSTR_prediction.py) and executed, so the features are
identical to those computed at prediction time. Each sample is processed separately, as in the
prediction step.

Inputs
  --src       path to BulkMonSTR_prediction.py of your BulkMonSTR installation
  --manifest  tab-delimited file with one row per sample and the columns
                sample_name   any sample identifier
                feature_file  feature output of this sample from extract_features.py
                              (headerless CSV, the input of the prediction step)
                label_file    tab-delimited file with the columns STR_ID and Label, where Label
                              is one of: Mosaic mutations, Germline Het, Artifacts
  --out-dir   output directory

Outputs (--out-dir)
  model_features_indel.csv.gz     sample_name, STR_ID, Label + indel model features
  model_features_mismatch.csv.gz  sample_name, STR_ID, Label + mismatch model features
  build_counts.tsv                labeled loci and loci with features per sample and type

Example
  python3 01_build_model_features.py --src ~/BayesMonSTR-BulkMonSTR/src/BulkMonSTR_prediction.py \
      --manifest manifest.tsv --out-dir features/
"""
import argparse
import io
import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
LABELS = ["Mosaic mutations", "Germline Het", "Artifacts"]
STR_ID_COL = 5          # 0-based column of the STR id in the headerless feature output


def block(lines, start_pat, end_pat, include_end=False):
    """Return the source lines from the first line matching start_pat up to end_pat."""
    s = next(i for i, l in enumerate(lines) if re.match(start_pat, l))
    e = next(i for i in range(s + 1, len(lines)) if re.match(end_pat, lines[i]))
    return "".join(lines[s:e + 1 if include_end else e])


def load_source(path):
    lines = open(path).readlines()
    return {
        "cate": block(lines, r"^MISMATCH_CATE = ", r"^\]$", include_end=True),
        "columns": block(lines, r"^germ_mut = \[", r"^fea_columns = ", include_end=True),
        "engineer": block(lines, r"^GIAB_fp = GIAB_fp\.drop_duplicates", r"^def process_input"),
        "process": block(lines, r"^def process_input", r"^# 太高的 stutter"),
        "orders": block(lines, r"^if MODEL_VAF_CORRECTION:", r"^\]$", include_end=True)
                  + block(lines, r"^mis_model_features_order = \[", r"^\]$", include_end=True),
        "vafcorr": block(lines, r"^def VAF_correction\(", r"^def check_nan"),
    }


def make_namespace(src):
    ns = {"pd": pd, "np": np, "stats": stats, "scipy": __import__("scipy"),
          "het_no_filter": False, "NORM1_AS_HET": True, "MODEL_VAF_CORRECTION": True,
          "ALLOW_NUMPY_MIN_VALUE": np.finfo(float).tiny}
    if not hasattr(stats, "binom_test"):          # scipy >= 1.12
        stats.binom_test = lambda k, n, p, alternative: stats.binomtest(int(k), int(n), p, alternative).pvalue
    for key in ("cate", "columns", "vafcorr", "process", "orders"):
        exec(src[key], ns)
    return ns


def features_for_sample(raw, src, ns):
    """raw: DataFrame of the raw feature output (strings) for the labeled loci of one sample."""
    buf = io.StringIO()
    raw.to_csv(buf, header=False, index=False)
    buf.seek(0)
    g = pd.read_csv(buf, sep=",", header=None)        # same dtype inference as the pipeline
    g.columns = ns["fea_columns"]
    local = dict(ns)
    local["GIAB_fp"] = g
    exec(src["engineer"], local)
    g = local["GIAB_fp"]
    vc = local["VAF_correction"]                     # VAF correction, as in the prediction step
    g["obs_vaf_correction"] = [vc(m, v, s, r, d) for m, v, s, r, d in zip(
        g["muttype"], g["observed_mosaic_allele_vaf_single_locus"], g["MBP"], g["read_length"], g["depth"])]
    g["mosaic_fraction_correction"] = [vc(m, f / 2, s, r, d) * 2 for m, f, s, r, d in zip(
        g["muttype"], g["MF_hom2het_het2het"], g["MBP"], g["read_length"], g["depth"])]
    local["GIAB_fp"] = g
    out = {}
    for kind, order in (("indel", "indel_model_features_order"), ("mismatch", "mis_model_features_order")):
        df = local["process_input"](kind, g)[local[order]].copy()
        if kind == "indel":   # the prediction step uses the corrected values under these names
            df.columns = ["MF_hom2het_het2het", *df.columns[1:3],
                          "observed_mosaic_allele_vaf_single_locus", *df.columns[4:]]
        out[kind] = df
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="BulkMonSTR_prediction.py")
    ap.add_argument("--manifest", required=True, help="sample_name, feature_file, label_file")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    src = load_source(a.src)
    ns = make_namespace(src)
    man = pd.read_csv(a.manifest, sep="\t")
    res = {"indel": [], "mismatch": []}
    counts = []
    for _, row in man.iterrows():
        lab = pd.read_csv(row["label_file"], sep="\t", dtype=str)[["STR_ID", "Label"]]
        bad = set(lab["Label"]) - set(LABELS)
        if bad:
            raise ValueError(f"{row['label_file']}: unknown labels {bad}; use {LABELS}")
        lab = lab.drop_duplicates("STR_ID").set_index("STR_ID")
        raw = pd.read_csv(row["feature_file"], header=None, dtype=str, keep_default_na=False)
        raw = raw[raw[STR_ID_COL].isin(lab.index)].drop_duplicates(STR_ID_COL)
        feats = features_for_sample(raw, src, ns)
        for kind, df in feats.items():
            meta = pd.DataFrame({"sample_name": row["sample_name"], "STR_ID": df.index,
                                 "Label": lab.loc[df.index, "Label"].values})
            res[kind].append(pd.concat([meta, df.reset_index(drop=True)], axis=1))
            counts.append((row["sample_name"], kind, len(df)))
        counts.append((row["sample_name"], "labeled", len(lab)))
        missing = len(lab) - len(raw)
        print(row["sample_name"], "labeled", len(lab), "not in feature file", missing,
              {k: len(v) for k, v in feats.items()})
    for kind, parts in res.items():
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(os.path.join(a.out_dir, f"model_features_{kind}.csv.gz"), index=False, compression="gzip")
        print(kind, df.shape, df["Label"].value_counts().to_dict())
    pd.DataFrame(counts, columns=["sample_name", "set", "n"]).to_csv(
        os.path.join(a.out_dir, "build_counts.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
