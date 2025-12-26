import pandas as pd
import numpy as np
import argparse
import json
import os
import sys
from pathlib import Path
import warnings


def determine_muttype2(df):
    """Determine zygosity transition type between GT and MGT."""
    def is_homo(geno):
        if pd.isna(geno) or '/' not in geno:
            return False
        parts = geno.split('/')
        return parts[0] == parts[1]


    df['muttype2'] = df.apply(lambda row:
        'homhet' if is_homo(row['GT']) and not is_homo(row['MGT']) else
        'hethom' if not is_homo(row['GT']) and is_homo(row['MGT']) else
        'hethet' if not is_homo(row['GT']) and not is_homo(row['MGT']) else
        None, axis=1)
    return df




def parse_allele_mosaic(row, CB_list):
    """Parse ALLELE_BARCODE_UMI to extract mosaic patterns and metrics."""
    if row['muttype2'] == 'homhet' or row['muttype2'] == 'hethet':
        key = row['MGT'].split('/')[1]
    elif row['muttype2'] == 'hethom':
        key = row['GT'].split('/')[1]
    else:
        return pd.Series([None] * 21, index=[
            'rc_total', 'barcode_rc_total_str', 'barcode_count_total', 'celltype_count_total',
            'rc_mosaic', 'vaf', 'barcode_rc_mosaic_str', 'barcode_count_mosaic', 'max_mosaic_barcode_rc',
            'max_mosaic_barcode_rc_total', 'max_mosaic_barcode_vaf', 'max_mosaic_barcode_rc_prop',
            'celltype_count_mosaic', 'na_celltype_rc_mosaic', 'max_celltype_rc_mosaic',
            'max_celltype_mosaic', 'mosaic_str',
            'max_vaf_mosaic_barcode', 'max_vaf_mosaic_barcode_rc', 'max_vaf_mosaic_barcode_vaf',
            'celltype_mosaic_vs_total'
        ])


    records = row['ALLELE_BARCODE_UMI'].split(";")
    result = []
    for record in records:
        if '|' not in record:
            continue
        allele_key, values = record.split("|", 1)
        sub_records = values.split("&")
        for sub_record in sub_records:
            sub_parts = sub_record.split("_")
            if len(sub_parts) == 2:
                result.append([allele_key, sub_parts[0], sub_parts[1]])


    if not result:
        return pd.Series([None] * 21, index=[...])  # same as above


    result_df = pd.DataFrame(result, columns=["Allele_index", "Read_name", "Barcode"])
    result_df["Read_name"] = result_df["Read_name"].str.replace("-", ":")

    if CB_list.empty:
        result_df['CellType'] = "Unknown"
    else:
        result_df = pd.merge(result_df, CB_list, on='Barcode', how='left')


    result_df_mosaic = result_df[result_df["Allele_index"] == key]
    if result_df_mosaic.empty:
        return pd.Series([None] * 21, index=[...])


    # Total counts
    rc_total = len(result_df)
    barcode_rc_total = result_df['Barcode'].value_counts()
    barcode_rc_total_str = ';'.join([f"{bc}:{ct}" for bc, ct in barcode_rc_total.items()])
    barcode_count_total = result_df['Barcode'].nunique()
    celltype_count_total = result_df['CellType'].nunique()


    # Mosaic counts
    rc_mosaic = len(result_df_mosaic)
    vaf = round(rc_mosaic / rc_total, 4)
    barcode_rc_mosaic = result_df_mosaic['Barcode'].value_counts()
    barcode_rc_mosaic_str = ';'.join([f"{bc}:{ct}" for bc, ct in barcode_rc_mosaic.items()])
    barcode_count_mosaic = result_df_mosaic['Barcode'].nunique()


    if not barcode_rc_mosaic.empty:
        max_barcode_mosaic = barcode_rc_mosaic.index[0]
        max_mosaic_barcode_rc = barcode_rc_mosaic.iloc[0]
        max_mosaic_barcode_rc_total = barcode_rc_total.get(max_barcode_mosaic, 0)
        max_mosaic_barcode_vaf = round(max_mosaic_barcode_rc / max_mosaic_barcode_rc_total, 4) if max_mosaic_barcode_rc_total > 0 else 0
        max_mosaic_barcode_rc_prop = round(max_mosaic_barcode_rc / rc_mosaic, 4)
    else:
        max_mosaic_barcode_rc = max_mosaic_barcode_rc_total = max_mosaic_barcode_vaf = max_mosaic_barcode_rc_prop = None
        max_barcode_mosaic = None


    celltype_rc_mosaic = result_df_mosaic['CellType'].value_counts()
    na_celltype_rc_mosaic = result_df_mosaic['CellType'].isna().sum() + (result_df_mosaic['CellType'] == 'NA').sum()
    celltype_count_mosaic = result_df_mosaic['CellType'].nunique()


    if not celltype_rc_mosaic.empty:
        max_celltype_rc_mosaic = celltype_rc_mosaic.iloc[0]
        max_celltype_mosaic = cell_type_mosaic = celltype_rc_mosaic.index[0]
    else:
        max_celltype_rc_mosaic = max_celltype_mosaic = None


    # Build mosaic_str
    str_parts = []
    for _, row2 in result_df_mosaic.iterrows():
        ct = row2['CellType'] if pd.notna(row2['CellType']) else 'NA'
        str_parts.append(f"{row2['Read_name']}_{row2['Barcode']}_{ct}")
    mosaic_str = f"{key}|{';'.join(str_parts)}"


    # Build celltype mosaic vs total
    celltype_str_parts = []
    all_barcode_ratios = []
    for celltype, group in result_df_mosaic.groupby('CellType'):
        bc_counts_mosaic = group['Barcode'].value_counts()
        ratio_list = []
        for bc, cnt_mosaic in bc_counts_mosaic.items():
            cnt_total = barcode_rc_total.get(bc, 0)
            ratio = cnt_mosaic / cnt_total if cnt_total > 0 else 0
            ratio_list.append((bc, cnt_mosaic, cnt_total, ratio))
            all_barcode_ratios.append((bc, cnt_mosaic, cnt_total, ratio))
        ratio_list.sort(key=lambda x: x[3], reverse=True)
        barcode_str = '&'.join([f"{bc}:{cm}/{ct}/{round(r, 2)}" for bc, cm, ct, r in ratio_list])
        celltype_str_parts.append(f"{celltype}|{barcode_str}")
    celltype_mosaic_vs_total = ';'.join(celltype_str_parts)


    # Max VAF barcode
    if all_barcode_ratios:
        best = max(all_barcode_ratios, key=lambda x: x[3])
        max_vaf_mosaic_barcode = best[0]
        max_vaf_mosaic_barcode_rc = best[1]
        max_vaf_mosaic_barcode_vaf = round(best[3], 4)
    else:
        max_vaf_mosaic_barcode = max_vaf_mosaic_barcode_rc = max_vaf_mosaic_barcode_vaf = None


    return pd.Series([
        rc_total, barcode_rc_total_str, barcode_count_total, celltype_count_total,
        rc_mosaic, vaf, barcode_rc_mosaic_str, barcode_count_mosaic,
        max_mosaic_barcode_rc, max_mosaic_barcode_rc_total, max_mosaic_barcode_vaf,
        max_mosaic_barcode_rc_prop, celltype_count_mosaic, na_celltype_rc_mosaic,
        max_celltype_rc_mosaic, max_celltype_mosaic, mosaic_str,
        max_vaf_mosaic_barcode, max_vaf_mosaic_barcode_rc, max_vaf_mosaic_barcode_vaf,
        celltype_mosaic_vs_total
    ], index=[
        'rc_total', 'barcode_rc_total_str', 'barcode_count_total', 'celltype_count_total',
        'rc_mosaic', 'vaf', 'barcode_rc_mosaic_str', 'barcode_count_mosaic',
        'max_mosaic_barcode_rc', 'max_mosaic_barcode_rc_total', 'max_mosaic_barcode_vaf',
        'max_mosaic_barcode_rc_prop', 'celltype_count_mosaic', 'na_celltype_rc_mosaic',
        'max_celltype_rc_mosaic', 'max_celltype_mosaic', 'mosaic_str',
        'max_vaf_mosaic_barcode', 'max_vaf_mosaic_barcode_rc', 'max_vaf_mosaic_barcode_vaf',
        'celltype_mosaic_vs_total'
    ])


def analyze_filters(df, filters, output_prefix, plot=True, title="Filtering Effect Analysis"):
    """
    Apply filters sequentially and individually, generate plots and return filtered DataFrame.
    Skips any filter if:
      - column not in df
      - column dtype is incompatible with threshold type (e.g., str vs float)
    """
    n_total = len(df)


    # Operator pretty mapping
    op_map = {
        '>': '>', '>=': '≥', '<': '<', '<=': '≤', '==': '==', '!=': '≠',
        'in': '∈', 'not in': '∉'
    }


    def is_compatible_type(series, thresh):
        """Check if series dtype is compatible with threshold type."""
        if isinstance(thresh, (int, float)):
            return pd.api.types.is_numeric_dtype(series)
        elif isinstance(thresh, str):
            return pd.api.types.is_string_dtype(series) or series.dtype == 'object'
        elif isinstance(thresh, (list, tuple)):
            if not thresh:
                return True
            elem_type = type(thresh[0])
            if elem_type in (int, float):
                return pd.api.types.is_numeric_dtype(series)
            elif elem_type == str:
                return pd.api.types.is_string_dtype(series) or series.dtype == 'object'
            else:
                return True
        else:
            return True  # fallback for other types


    # === Step 1: Validate and preprocess all filters once ===
    valid_filters = []
    for filt in filters:
        col = filt.get('column')
        op = filt.get('operator')
        thresh = filt.get('threshold')


        # Check required keys
        if not all(k in filt for k in ['column', 'operator', 'threshold']):
            warnings.warn(f"Invalid filter format (missing keys). Skipping: {filt}")
            continue


        # Check column exists
        if col not in df.columns:
            warnings.warn(f"Column '{col}' not found in DataFrame. Skipping filter: {filt}")
            continue


        # Check operator
        if op not in ['>', '>=', '<', '<=', '==', '!=', 'in', 'not in']:
            warnings.warn(f"Unsupported operator '{op}'. Skipping filter: {filt}")
            continue


        # Type compatibility check
        series = df[col]
        if op in ['>', '>=', '<', '<=', '==', '!=', 'in', 'not in']:
            if not is_compatible_type(series, thresh):
                t_type = type(thresh).__name__
                c_dtype = series.dtype
                warnings.warn(
                    f"Type mismatch: filter '{col} {op} {thresh}' skipped. "
                    f"Column '{col}' dtype is '{c_dtype}', but threshold is type '{t_type}'."
                )
                continue


        # Auto-generate label if not provided
        if 'label' not in filt or filt['label'] is None:
            pretty_op = op_map.get(op, op)
            if op in ['in', 'not in']:
                disp_list = thresh if isinstance(thresh, (list, tuple)) else [thresh]
                disp = ','.join(map(str, disp_list[:3])) + ('...' if len(disp_list) > 3 else '')
                filt['label'] = f"{col} {pretty_op} [{disp}]"
            else:
                filt['label'] = f"{col} {pretty_op} {thresh}"


        valid_filters.append(filt)

    # === Step 2: Sequential filtering ===
    current_df = df.copy()
    sequential_results = [{'step': 'Original', 'remaining': n_total}]
    for filt in valid_filters:
        col, op, thresh, label = filt['column'], filt['operator'], filt['threshold'], filt['label']
        try:
            if op == '>':
                current_df = current_df[current_df[col] > thresh]
            elif op == '>=':
                current_df = current_df[current_df[col] >= thresh]
            elif op == '<':
                current_df = current_df[current_df[col] < thresh]
            elif op == '<=':
                current_df = current_df[current_df[col] <= thresh]
            elif op == '==':
                current_df = current_df[current_df[col] == thresh]
            elif op == '!=':
                current_df = current_df[current_df[col] != thresh]
            elif op == 'in':
                current_df = current_df[current_df[col].isin(thresh)]
            elif op == 'not in':
                current_df = current_df[~current_df[col].isin(thresh)]
        except Exception as e:
            warnings.warn(f"Error in sequential filter {label}: {e}")
        sequential_results.append({'step': label, 'remaining': len(current_df)})


    sequential_df = pd.DataFrame(sequential_results)

    if plot:
        # === Step 3: Plotting ===
        # Sequential plot
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.bar(sequential_df['step'], sequential_df['remaining'], color='skyblue', edgecolor='navy', alpha=0.7)
        plt.title(f"{title}\nSequential Filtering: Sites Remaining", fontsize=12)
        plt.ylabel("Number of Sites Remaining")
        plt.xticks(rotation=45, ha='right')
        for i, row in sequential_df.iterrows():
            plt.text(i, row['remaining'] + n_total * 0.01, str(row['remaining']), ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_sequential_filtering.png', dpi=300)
        plt.close()

        # Individual plot
        individual_results = []
        for filt in filters:  # Iterate over original to report all, but skip invalid
            col, op, thresh = filt['column'], filt['operator'], filt['threshold']
            label = filt.get('label', f"{col} {op} {thresh}")


            # Skip if not in valid_filters
            if filt not in valid_filters:
                continue


            try:
                if op == '>':
                    passed = df[df[col] > thresh]
                elif op == '>=':
                    passed = df[df[col] >= thresh]
                elif op == '<':
                    passed = df[df[col] < thresh]
                elif op == '<=':
                    passed = df[df[col] <= thresh]
                elif op == '==':
                    passed = df[df[col] == thresh]
                elif op == '!=':
                    passed = df[df[col] != thresh]
                elif op == 'in':
                    passed = df[df[col].isin(thresh)]
                elif op == 'not in':
                    passed = df[~df[col].isin(thresh)]
                else:
                    continue  # should not reach here
                n_filtered_out = n_total - len(passed)
                individual_results.append({'filter': label, 'filtered_out': n_filtered_out})
            except Exception as e:
                warnings.warn(f"Error applying filter {label}: {e}")
        individual_df = pd.DataFrame(individual_results)

        if not individual_df.empty:
            plt.figure(figsize=(10, 6))
            plt.barh(individual_df['filter'], individual_df['filtered_out'], color='salmon', edgecolor='darkred', alpha=0.7)
            plt.title("Individual Filtering: Sites Filtered Out by Each Criterion Alone", fontsize=12)
            plt.xlabel("Number of Sites Filtered Out")
            for i, row in individual_df.iterrows():
                plt.text(row['filtered_out'] + n_total * 0.01, i, str(row['filtered_out']), va='center', fontsize=9)
            plt.tight_layout()
            plt.savefig(f'{output_prefix}_individual_filtering.png', dpi=300)
            plt.close()
        else:
            warnings.warn("No valid filters applied. Skipping individual filtering plot.")


    return current_df


def run_cell_level_filter(
    input_csv: str,
    sample_name: str,
    filters_json: str,
    output_file: str,
    cb_list_tsv: str = "None",
    mutation_type: str = "both",
    plot: bool = True,
    save_intermediate: bool = True
    
) -> pd.DataFrame:
    output_prefix=Path(output_file).with_suffix('')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    df_origin = pd.read_csv(input_csv)

    if cb_list_tsv is None:
        CB_list = pd.DataFrame(columns=['Barcode', 'CellType'])
    else:
        if not os.path.exists(cb_list_tsv):
            raise FileNotFoundError(f"Cell barcode list file not found: {cb_list_tsv}")
        CB_list = pd.read_csv(cb_list_tsv, header=None, sep="\t", names=['Barcode', 'CellType'])

    if not os.path.exists(filters_json):
        raise FileNotFoundError(f"Filters file not found: {filters_json}")
    with open(filters_json, 'r', encoding='utf-8') as f:
        filters = json.load(f)

    df_processed = determine_muttype2(df_origin)

    result_series = df_processed.apply(lambda row: parse_allele_mosaic(row, CB_list=CB_list), axis=1)
    df_processed = pd.concat([df_processed, result_series], axis=1)

    df_processed['MBP_abs'] = abs(df_processed['MBP'])
    df_processed['mut_mean_baseq'] = df_processed['mut_mean_baseq'].fillna(40)
    df_processed['sample'] = sample_name

    clean_df = analyze_filters(
        df_processed,
        filters,
        title=f"QC Filters - {sample_name}",
        output_prefix=output_prefix,
        plot=plot
    )

    if mutation_type == 'cell_specific':
        clean_df = clean_df[clean_df['barcode_count_mosaic'] == 1]
    elif mutation_type == 'share':
        clean_df = clean_df[clean_df['barcode_count_mosaic'] > 1]

    if save_intermediate:
        df_processed.to_csv(f'{output_prefix}_all.csv', index=False)
        clean_df.to_csv(f'{output_prefix}_clean.csv', index=False)
        print(f"Filtering completed. Final result saved to: {output_file}")        

    basic_cols = ['chrom','reference_start_coordinate_1_based_include','reference_end_coordinate_1_based_include',
                'mut_source_seq','mut_target_seq','GT','MGT','alleles_mut_type',
                'str_id','sample','barcode_count_mosaic','max_vaf_mosaic_barcode','max_celltype_mosaic',
                'used_read_num_in_genotyping','rc_mosaic','observed_mosaic_allele_vaf_single_locus','CMP']
    basic_cols_name = ['chr','start','end','ref','alt',
                'germline genotype','mosaic genotype','mutation type',
                'id','sample','mutant cell count','cell barcode','cell type',
                'depth','mosaic depth','VAF','posterior']
                
    clean_df_simp = clean_df[basic_cols]
    clean_df_simp.columns = basic_cols_name
    clean_path = f'{output_prefix}_clean_basicinfo.csv'
    clean_df_simp.to_csv(clean_path, index=False)

    print(f"Simplified final result saved to: {clean_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess and filter mosaic STR variants.")


    parser.add_argument('--input_csv', type=str, required=True, help='Input CSV file')
    parser.add_argument('--sample_name', type=str, required=True, help='Sample name')
    parser.add_argument('--filters_json', type=str, required=True, help='JSON file containing filter rules')
    parser.add_argument('--cb_list_tsv', type=str, default="None", help='Cell barcode to cell type mapping file (TSV)')
    parser.add_argument('--mutation_type', required=False, default="both", choices=["both", "cell_specific", "share"], help='Type of mutation')
    parser.add_argument('--output_file', type=str, required=True, help='Output file path')
    parser.add_argument('--plot', type=bool, required=False, default=True, help='Whether to plot count of loci during filtering.')


    args = parser.parse_args()


    try:
        run_cell_level_filter(
            input_csv=args.input_csv,
            sample_name=args.sample_name,
            filters_json=args.filters_json,
            cb_list_tsv=args.cb_list_tsv,
            mutation_type=args.mutation_type,
            output_file=args.output_file,
            plot=args.plot,
            save_intermediate=True
        )
        print(f"✅ Processing completed for sample '{args.sample}'.")
    except Exception as e:
        print(f"❌ Error during processing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()