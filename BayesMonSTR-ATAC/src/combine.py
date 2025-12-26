import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import json


def merge_csv_files(folder_path, output_file=None, add_filename=False, recursive=False, file_suffix='.csv'):
    df_list = []
    if recursive:
        files_iter = os.walk(folder_path)
    else:
        files_iter = [("", folder_path, os.listdir(folder_path))]

    for root, _, files in files_iter:
        for filename in files:
            if filename.endswith(file_suffix) and not filename.startswith('.'):  # 忽略隐藏文件
                file_path = os.path.join(root, filename)
                try:
                    df = pd.read_csv(file_path)
                    if add_filename:
                        df['sample'] = Path(file_path).parent.name
                    df_list.append(df)
                except pd.errors.EmptyDataError:
                    print(f"Skip empty file: {file_path}")
                except Exception as e:
                    print(f"Read fail: {file_path} - error: {e}")

    if not df_list:
        print("Warning: No valid files were found for merging.")
        return pd.DataFrame()


    # 合并
    merged_df = pd.concat(df_list, ignore_index=True)


    # 保存
    if output_file:
        merged_df.to_csv(output_file, index=False)
        print(f"Merge completed. Saved to: {output_file}")


    return merged_df
    
def analyze_filters(df, filters, output_prefix, plot=True, title="Filtering Effect Analysis"):

    n_total = len(df)

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
            return True 

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

def process_allele_df(df):
    all_records = []

    for _, row in df.iterrows():
        if row['ALLELE_BARCODE_UMI']!='.':
            records = row['ALLELE_BARCODE_UMI'].split(";")
            barcodes = []
            for record in records:
                if '|' not in record:
                    continue
                try:
                    _, values = record.split("|", 1)
                    sub_records = values.split("&")
                    for sub_record in sub_records:
                        sub_parts = sub_record.split("_")
                        if len(sub_parts) == 2:
                            barcode = sub_parts[1]  # 取下划线后的 Barcode
                            barcodes.append(barcode)
                except Exception as e:
                    continue

            if barcodes:
                barcode_counts = pd.Series(barcodes).value_counts()
                temp_df = pd.DataFrame({
                    'barcode': barcode_counts.index,
                    'count': barcode_counts.values,
                    'id': row['str_id'],
                    'length': row['length'],
                    'sample': row['sample']
                })
                all_records.append(temp_df)

    if not all_records:
        return pd.DataFrame(columns=['id', 'length', 'barcode', 'count', 'sample'])

    final_df = pd.concat(all_records, ignore_index=True)
    final_df = final_df[['id', 'length', 'barcode', 'count', 'sample']].reset_index(drop=True)
    return final_df

def run(input_dir, output_prefix, filters_json=None, mutation_type='both'):
    output_path = Path(output_prefix)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    all_path = f'{output_prefix}_all.csv'
    all_path = Path(all_path)

    if all_path.exists():
        df_processed = pd.read_csv(all_path)
        print(f"Result of all loci exits: {all_path}")
    else:
        df_processed = merge_csv_files(input_dir, output_file=None, recursive=True, file_suffix='_all.csv')
        df_processed.to_csv(all_path, index=False)
        print(f"Result of all loci saved to: {all_path}")

    if filters_json is None:
        filters_json = os.path.join(os.path.dirname(__file__), 'filters.json')
    with open(filters_json, 'r', encoding='utf-8') as f:
        filters = json.load(f)
    clean_df = analyze_filters(
        df_processed,
        filters,
        output_prefix=output_prefix
    )
    if mutation_type == 'cell_specific':
        clean_df = clean_df[clean_df['barcode_count_mosaic'] == 1]
    elif mutation_type == 'share':
        clean_df = clean_df[clean_df['barcode_count_mosaic'] > 1]
    clean_df.to_csv(f'{output_prefix}_clean.csv', index=False)
    print(f"Filtering completed. Final result saved to: {output_prefix}_clean.csv")        

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
    clean_df_simp.to_csv(f'{output_prefix}_clean_basicinfo.csv', index=False)
    print(f"Simplified final result saved to: {output_prefix}_clean_basicinfo.csv")


    df_all = merge_csv_files(input_dir, add_filename=True, recursive=True, file_suffix='_raw.bed')
    panel_38_final = pd.read_csv(f"/storage/douyanmeiLab/wangchunyi/reference/TR_catalog/HipSTR-references/human/hg38.hipstr_reference_removeslash_lt150_simp_annovar_gex_maf_gc_rank_zscore_hmm_gex2_addbasicinfo.csv")
    df_all = pd.merge(df_all, panel_38_final, left_on='str_id', right_on='id', how='left')

    final_df = process_allele_df(df_all)
    final_df_group = final_df.groupby(['sample','barcode']).agg(
        count=('length', 'size'),      # 每组有多少行
        length_sum=('length', 'sum')   # length 列的总和
    ).reset_index()
    final_df_group.to_csv(f"{output_prefix}_matrix.csv",index=False)
    print(f"Matirx saved to: {output_prefix}_matrix.csv")

    df_regulatory = df_all[df_all['max_freq_anno'].isin(['TssA','TssBiv','EnhA1','EnhA2'])]
    final_df_regulatory = process_allele_df(df_regulatory)
    final_df_regulatory_group = final_df_regulatory.groupby(['sample','barcode']).agg(
        count=('length', 'size'),      # 每组有多少行
        length_sum=('length', 'sum')   # length 列的总和
    ).reset_index()
    final_df_regulatory_group.to_csv(f"{output_prefix}_matrix_reg.csv",index=False)
    print(f"Matrix of regulatory loci saved to: {output_prefix}_matrix_reg.csv")

if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(description="Run the full mosaic filtering pipeline for a sample and region.")
    parser.add_argument("--input_dir", default="./04filter", help="Root input directory")
    parser.add_argument("--output_prefix", default="./04filter/results", help="Prefix of outputs")
    parser.add_argument("--filters_json", default=None, help="Json file for filtering thresholds. Default path is src/filters.json.")
    parser.add_argument('--mutation_type', required=False, default="both", choices=["both", "cell_specific", "share"], help='Type of mutation')

    args = parser.parse_args()


    run(
        input_dir=args.input_dir,
        output_prefix=args.output_prefix,
        filters_json=args.filters_json,
        mutation_type=args.mutation_type
    )