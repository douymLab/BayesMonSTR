"""Configurable, mutation-aware filtering for BayesMonSTR-ATAC results."""

import json
import math
import warnings

import numpy as np
import pandas as pd


OPERATORS = {">", ">=", "<", "<=", "==", "!=", "in", "not in"}
DEFAULT_MUTATION_TYPE_MAP = {
    "Insertion": "INDEL", "Deletion": "INDEL", "SNV": "Mismatch",
    "continuous_MNV": "Mismatch", "noncontinuous_MNV": "Mismatch",
}


def load_filter_config(path):
    """Load either a legacy list of filters or the mutation-aware v2 schema."""
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, (list, dict)):
        raise ValueError("Filter JSON must contain a list (legacy) or an object (v2).")
    return config


def _comparison_mask(series, operator, threshold):
    numeric_threshold = isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
    numeric_list = (
        isinstance(threshold, (list, tuple)) and threshold
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in threshold)
    )
    if numeric_threshold or numeric_list:
        series = pd.to_numeric(series, errors="coerce")
    if operator == ">": return series > threshold
    if operator == ">=": return series >= threshold
    if operator == "<": return series < threshold
    if operator == "<=": return series <= threshold
    if operator == "==": return series == threshold
    if operator == "!=": return series != threshold
    if operator == "in":
        return series.isin(threshold if isinstance(threshold, (list, tuple)) else [threshold])
    if operator == "not in":
        return ~series.isin(threshold if isinstance(threshold, (list, tuple)) else [threshold])
    raise ValueError(f"Unsupported operator: {operator}")


def _condition_label(condition):
    if "column" in condition:
        return condition.get("label", f"{condition.get('column')} {condition.get('operator')} {condition.get('threshold')}")
    for key in ("all", "any", "not"):
        if key in condition:
            return condition.get("label", key.upper())
    return "condition"


def condition_mask(df, condition, on_missing_column="error"):
    """Evaluate a leaf condition or a recursively nested all/any/not expression."""
    if not isinstance(condition, dict):
        raise ValueError(f"A filter condition must be an object, got: {condition!r}")
    if condition.get("enabled", True) is False:
        return pd.Series(True, index=df.index, dtype=bool)
    group_keys = [key for key in ("all", "any", "not") if key in condition]
    if group_keys:
        if len(group_keys) != 1:
            raise ValueError(f"Condition must use exactly one of all/any/not: {condition}")
        key = group_keys[0]
        children = condition[key]
        if key == "not":
            return ~condition_mask(df, children, on_missing_column)
        if not isinstance(children, list):
            raise ValueError(f"'{key}' must contain a list of conditions.")
        mask = pd.Series(key == "all", index=df.index, dtype=bool)
        for child in children:
            child_mask = condition_mask(df, child, on_missing_column)
            mask = (mask & child_mask) if key == "all" else (mask | child_mask)
        return mask

    required = {"column", "operator", "threshold"}
    if not required.issubset(condition):
        raise ValueError(f"Filter is missing required keys {sorted(required)}: {condition}")
    column, operator = condition["column"], condition["operator"]
    if operator not in OPERATORS:
        raise ValueError(f"Unsupported operator '{operator}' in filter: {condition}")
    if column not in df.columns:
        message = f"Column '{column}' required by filter was not found."
        if on_missing_column == "skip":
            warnings.warn(message + " Filter skipped.")
            return pd.Series(True, index=df.index, dtype=bool)
        raise KeyError(message)
    try:
        series = df[column]
        mask = _comparison_mask(series, operator, condition["threshold"]).fillna(False)
        if condition.get("missing_values", "fail") == "pass":
            threshold = condition["threshold"]
            numeric = isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
            numeric_list = (
                isinstance(threshold, (list, tuple)) and threshold
                and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in threshold)
            )
            missing = pd.to_numeric(series, errors="coerce").isna() if numeric or numeric_list else series.isna()
            mask |= missing
        return mask
    except TypeError as exc:
        raise TypeError(f"Cannot apply filter '{_condition_label(condition)}' to column '{column}' with dtype {df[column].dtype}.") from exc


def _normalise_conditions(filters):
    if filters is None: return []
    if isinstance(filters, list): return filters
    if isinstance(filters, dict) and "filters" in filters: return filters["filters"]
    if isinstance(filters, dict): return [filters]
    raise ValueError("Filters must be a list or condition object.")


def analyze_filters(df, filters, output_prefix=None, plot=False, title="Filtering Effect Analysis", on_missing_column="error"):
    """Apply top-level conditions sequentially; nested groups are evaluated atomically."""
    current = df.copy()
    counts = [{"step": "Original", "remaining": len(current)}]
    for condition in _normalise_conditions(filters):
        if condition.get("enabled", True) is False:
            continue
        current = current.loc[condition_mask(current, condition, on_missing_column)]
        counts.append({"step": _condition_label(condition), "remaining": len(current)})
    if plot and output_prefix:
        import matplotlib.pyplot as plt
        count_df = pd.DataFrame(counts)
        plt.figure(figsize=(10, 6))
        plt.bar(count_df["step"], count_df["remaining"], color="skyblue", edgecolor="navy")
        plt.title(f"{title}\nSequential Filtering: Sites Remaining")
        plt.ylabel("Number of Sites Remaining")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_sequential_filtering.png", dpi=300)
        plt.close()
    return current


def _binomial_noise_p_value(row):
    try:
        max_noise = max(float(row[name]) for name in ("inframe_ins_prob", "inframe_del_prob", "outframe_ins_prob", "outframe_del_prob") if pd.notna(row[name]))
        depth = int(round(float(row["depth"])))
        vaf = float(row["genotyping_mle_mosaic_allele_vaf_single_locus"])
        mosaic_depth = int(round(depth * vaf))
        if depth < 0 or mosaic_depth < 0 or mosaic_depth > depth or not 0 <= max_noise <= 1:
            return np.nan
        try:
            from scipy.stats import binomtest
            return binomtest(mosaic_depth, depth, max_noise, alternative="greater").pvalue
        except ImportError:
            # Keeps filtering usable in lightweight environments; production installs use SciPy.
            return min(1.0, sum(
                math.comb(depth, k) * max_noise ** k * (1 - max_noise) ** (depth - k)
                for k in range(mosaic_depth, depth + 1)
            ))
    except (KeyError, TypeError, ValueError):
        return np.nan


def add_filter_features(df, mutation_type_map=None):
    """Add the common classification and derived columns used by filter profiles."""
    result = df.copy()
    mapping = DEFAULT_MUTATION_TYPE_MAP.copy()
    if mutation_type_map:
        mapping.update(mutation_type_map)
    if "alleles_mut_type" not in result:
        raise KeyError("Column 'alleles_mut_type' is required to classify mutation types.")
    result["mut_type"] = result["alleles_mut_type"]
    result["mut_type_general"] = result["mut_type"].map(mapping)
    if "barcode_count_mosaic" in result:
        result["mut_type_share"] = pd.Series(pd.NA, index=result.index, dtype="object")
        result.loc[result["barcode_count_mosaic"] == 1, "mut_type_share"] = "Cell-specific"
        result.loc[result["barcode_count_mosaic"] > 1, "mut_type_share"] = "Share"
    if "AAD" in result:
        result["allele_count"] = result["AAD"].apply(
            lambda value: len(value.split(";"))
            if pd.notna(value) and isinstance(value, str) and value.strip()
            else 0
        )
    required = {"inframe_ins_prob", "inframe_del_prob", "outframe_ins_prob", "outframe_del_prob", "depth", "genotyping_mle_mosaic_allele_vaf_single_locus"}
    if "binomial_noise_p_value" not in result and required.issubset(result.columns):
        result["binomial_noise_p_value"] = result.apply(_binomial_noise_p_value, axis=1)
    return result


def apply_cohort_filter_config(df, config, output_prefix=None, plot=False, title="Cohort Filters"):
    """Apply filters that require combined samples, including recurrent-locus removal."""
    if not isinstance(config, dict):
        return df
    cohort = config.get("cohort")
    if not cohort or cohort.get("enabled", True) is False:
        return df

    result = add_filter_features(df, config.get("mutation_type_map"))
    result["__filter_order"] = np.arange(len(result))
    type_column = config.get("mutation_type_column", "mut_type_general")
    id_column = cohort.get("locus_id_column", "str_id")
    group_by = cohort.get("group_by", ["dataset", id_column])
    missing_group_columns = [column for column in group_by if column not in result]
    if id_column not in result:
        raise KeyError(f"Cohort locus ID column '{id_column}' was not found.")
    if missing_group_columns and cohort.get("on_missing_group_column", "error") != "skip_recurrence":
        raise KeyError(f"Cohort grouping columns were not found: {missing_group_columns}")
    if missing_group_columns:
        warnings.warn(
            f"Cohort recurrence filtering skipped because grouping columns were not found: "
            f"{missing_group_columns}. Row-level cohort filters are still applied."
        )

    parts = []
    configured_mask = pd.Series(False, index=result.index, dtype=bool)
    on_missing = config.get("on_missing_column", "error")
    for mutation_type, block in cohort.get("by_mutation_type", {}).items():
        type_mask = result[type_column] == mutation_type
        configured_mask |= type_mask
        part = result.loc[type_mask]
        if part.empty:
            continue
        if block.get("enabled", True) is False:
            parts.append(part)
            continue
        part = analyze_filters(
            part,
            block.get("filters", []),
            f"{output_prefix}_cohort_{mutation_type.lower()}" if output_prefix else None,
            plot,
            f"{title} - {mutation_type}",
            on_missing,
        )
        max_group_count = block.get("max_group_count")
        if max_group_count is not None and not missing_group_columns and not part.empty:
            group_counts = part.groupby(group_by, dropna=False).size().reset_index(name="count")
            recurrent_ids = group_counts.loc[group_counts["count"] > max_group_count, id_column].unique()
            part = part.loc[~part[id_column].isin(recurrent_ids)]
        parts.append(part)

    if cohort.get("unconfigured_mutation_types", "drop") == "keep":
        parts.append(result.loc[~configured_mask])
    result = pd.concat(parts, axis=0) if parts else result.iloc[0:0]
    return result.sort_values("__filter_order").drop(columns="__filter_order")


def apply_filter_config(df, config, output_prefix=None, plot=False, title="QC Filters"):
    """Apply a legacy flat config or the v2 basic/type-specific/final pipeline."""
    if isinstance(config, list):
        return analyze_filters(df, config, output_prefix, plot, title, on_missing_column="skip")
    if config.get("version", 2) != 2:
        raise ValueError(f"Unsupported filter configuration version: {config.get('version')}")
    on_missing = config.get("on_missing_column", "error")
    result = add_filter_features(df, config.get("mutation_type_map"))
    result["__filter_order"] = np.arange(len(result))
    result = analyze_filters(result, config.get("basic", []), f"{output_prefix}_basic" if output_prefix else None, plot, f"{title} - basic", on_missing)
    type_column = config.get("mutation_type_column", "mut_type_general")
    if type_column not in result:
        raise KeyError(f"Mutation type column '{type_column}' was not found.")
    by_type = config.get("by_mutation_type", {})
    if by_type:
        parts = []
        configured_mask = pd.Series(False, index=result.index, dtype=bool)
        for mutation_type, block in by_type.items():
            type_mask = result[type_column] == mutation_type
            configured_mask |= type_mask
            part = result.loc[type_mask]
            if part.empty:
                continue
            if isinstance(block, dict) and block.get("enabled", True) is False:
                parts.append(part)
            else:
                parts.append(analyze_filters(part, block, f"{output_prefix}_{mutation_type.lower()}" if output_prefix else None, plot, f"{title} - {mutation_type}", on_missing))
        if config.get("unconfigured_mutation_types", "drop") == "keep":
            parts.append(result.loc[~configured_mask])
        result = pd.concat(parts, axis=0) if parts else result.iloc[0:0]
    result = analyze_filters(result, config.get("final", []), f"{output_prefix}_final" if output_prefix else None, plot, f"{title} - final", on_missing)
    return result.sort_values("__filter_order").drop(columns="__filter_order")
