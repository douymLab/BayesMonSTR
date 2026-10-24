"""CSV export for final filtering results."""


EXCLUDED_RESULT_COLUMNS = [
    "obs_mut_order",
    "obs_depth_string",
    "mle_mut_order",
    "mle_depth_string",
    "obs_phase_state",
    "obs_hap_state",
    "obs_hap_count",
    "mle_phase_state",
    "mle_hap_state",
    "mle_hap_count",
    "hard_filter_details",
    "hard_filter_category",
    "hard_filter_decision",
    "RF_Prediction",
    "Confidence",
    "Final Decision",
]


def write_result_csv(df, path):
    """Omit internal diagnostics without changing data used for filtering."""
    df.drop(columns=EXCLUDED_RESULT_COLUMNS, errors="ignore").to_csv(path, index=False)
