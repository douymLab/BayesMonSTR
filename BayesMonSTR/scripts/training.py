import os
import warnings

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
RF_N_ESTIMATORS  = 100

DATA_PATH = "supp_df.tsv"
MODEL_PATHS = {
    "phase":    "../demo/model/phase.joblib",
    "unphase":  "../demo/model/unphase.joblib",
}
OUTPUT_DIR = "output_models"
# ────────────────────────────────────────────────────────────────────────────


def get_feature_names(model) -> list[str]:
    """Extract feature names from a pre-trained sklearn model via feature_names_in_."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    raise ValueError(
        "The loaded model has no 'feature_names_in_' attribute. "
        "Ensure the model was trained on a named pandas DataFrame."
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, sep="\t")
    print(f"  Shape: {df.shape}")

    # ── Train one model per pre-trained joblib ────────────────────────────
    for model_name, model_path in MODEL_PATHS.items():
        print(f"\n{'='*50}")
        print(f"Model     : {model_name}")
        print(f"Loading   : {model_path}")

        pretrained = joblib.load(model_path)
        features   = get_feature_names(pretrained)
        print(f"  Features : {len(features)}")

        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"  Ignoring {len(missing)} missing feature(s): {missing[:10]}{'...' if len(missing) > 10 else ''}")
            features = [f for f in features if f in df.columns]

        X = df[features]
        y = df["label"]

        rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS)
        rf.fit(X, y)

        out_path = os.path.join(OUTPUT_DIR, f"{model_name}_model.joblib")
        joblib.dump(rf, out_path)
        print(f"  Saved → {out_path}")

    print(f"\nDone. Models saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()