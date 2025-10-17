import argparse
import os

import joblib
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process mutation data and make predictions"
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output"
    )
    parser.add_argument("-p", "--prefix", type=str, default="str_gt", help="prefix")
    parser.add_argument(
        "-u1", "--unphase_model1_path", type=str, default="model/unphase1.joblib", help="Path to unphase model 1"
    )
    parser.add_argument(
        "-u2", "--unphase_model2_path", type=str, default="model/unphase2.joblib", help="Path to unphase model 2"
    )
    parser.add_argument(
        "-ph", "--phase_model_path", type=str, default="model/phase.joblib", help="Path to phase model"
    )
    parser.add_argument(
        "-ind", "--individual", type=str, default="1", help="Individual code"
    )
    return parser.parse_args()


def load_models(unphase_model1_path, unphase_model2_path, phase_model_path):
    return {
        "unphase_model1": joblib.load(unphase_model1_path),
        "unphase_model2": joblib.load(unphase_model2_path),
        "phase_model": joblib.load(phase_model_path),
    }


def process_data(df, models, individual):
    # Rename columns except mappability, removing individual suffix
    rename_dict = {col: col.replace(f"_{individual}", "") for col in df.columns 
                  if col.endswith(f"_{individual}") and not col.startswith("mappability")}
    df = df.rename(columns=rename_dict)

    # Process phase data
    phase_df = df[(df["prop_sc_spn_read"] != 0) & (df["phasing_k_counts"] >= 2)].copy()
    if len(phase_df) > 0:
        X_phase = phase_df[models["phase_model"].feature_names_in_]
        phase_df["prediction"] = models["phase_model"].predict(X_phase)

        # Add phase filter
        phase_df["phase_filter"] = (
            (phase_df["vaf_mut_sc_mean_prob"] >= 0.1)
            & (phase_df["phasing_k_counts"] >= 4)
            & (phase_df["dis_prop_bulk"] < 0.1)
            & (phase_df["dis_prop_sc"] < 0.1)
            & (phase_df["dis_amp"] < 0.1)
            & (phase_df["dis_amp_avg"] < 0.1)
            & (phase_df["dis_prop_k"] < 0.1)
            & (phase_df["dis_prop_k_avg"] < 0.1)
        )

    # Process unphase data
    unphase_df = df.copy()
    if len(unphase_df) > 0:
        X_unphase = unphase_df[models["unphase_model1"].feature_names_in_]
        predictions = models["unphase_model1"].predict(X_unphase)

        # Apply model2 for mosaic predictions
        mask_mosaic = predictions == "mosaic"
        if any(mask_mosaic):
            X_mosaic = unphase_df[mask_mosaic][models["unphase_model2"].feature_names_in_]
            predictions[mask_mosaic] = models["unphase_model2"].predict(X_mosaic)

        unphase_df["prediction"] = predictions
        unphase_df["unphase_filter"] = (unphase_df["vaf_mut_sc_mean_prob"] > 0.25) & (
            unphase_df["min_mut_sc_counts"] >= 3
        )

    if len(phase_df) > 0:
        phase_df = phase_df[
            (phase_df["num_mut_cell"] > 0) &
            (phase_df["mo_posterior"] >= 0.9) &
            (phase_df["is_germ_hom"] == 1) & # whether germline genotype is reference homozygous
            (phase_df["prediction"] == "mosaic") &
            (phase_df["phase_filter"] == True)
        ]

    if len(unphase_df) > 0:
        unphase_df = unphase_df[
            (unphase_df["num_mut_cell"] > 0) &
            (unphase_df["mo_posterior"] >= 0.9) &
            (unphase_df["is_germ_hom"] == 1) & # whether germline genotype is reference homozygous
            (unphase_df["prediction"] == "mosaic") &
            (unphase_df["unphase_filter"] == True)
        ]

    return phase_df, unphase_df


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load input file
    print(f"Reading input file: {args.input}")
    combined_df = pd.read_csv(args.input, sep="\t", comment="#")

    # Load models
    print("Loading models...")
    models = load_models(
        args.unphase_model1_path, args.unphase_model2_path, args.phase_model_path
    )

    # Process data
    print("Processing data...")
    phase_df, unphase_df = process_data(combined_df, models, args.individual)

    # Save results
    print(f"Saving results to: {args.output}")
    output_phase = os.path.join(args.output, f"{args.prefix}_phase_results.csv")
    output_unphase = os.path.join(
        args.output, f"{args.prefix}_unphase_results.csv"
    )

    if len(phase_df) > 0:
        phase_df.to_csv(output_phase, index=False)
    else:
        print("No phase results to save")

    if len(unphase_df) > 0:
        unphase_df.to_csv(output_unphase, index=False)
    else:
        print("No unphase results to save")

    print("Processing complete!")


if __name__ == "__main__":
    main()

# python mosaic.py -i ../results/str_gt.vcf -o ../results
