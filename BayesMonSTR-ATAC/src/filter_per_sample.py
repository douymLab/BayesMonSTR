import os
import sys
import subprocess
from pathlib import Path
import shutil

from initial_hard_filter import run_initial_hard_filter
from extract_features import run_extract_features
from final_hard_filter import run_final_hard_filter
from cell_level_filter import run_cell_level_filter
import warnings

warnings.filterwarnings("ignore")

def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def file_exists_and_not_empty(filepath):
    return os.path.isfile(filepath) and os.path.getsize(filepath) > 0


def run(sample, reference_genome, vcf, stutter_model, cell_barcode, 
         pop_info, recurrent_info, mappability, metadata, chrom=None, start=0, end=1000000000, 
         output_dir="./04filter", filters_json=None,
         mis_or_indel="both", mode="all", mutation_type="both",keep_temp=True,
         p=0.5, m=0, u=0.8, l=1e-8, d=1, ms=50, pgi=0.3, r=0.3, cf=0):
    
    if filters_json is None:
        filters_json = os.path.join(os.path.dirname(__file__), 'filters.json')
    if chrom is not None:
        output1 = f"{output_dir}/tmp_{chrom}_{start}_{end}/01initial_hard_filter/{sample}/{sample}_{chrom}_{start}_{end}.bed"
        output2 = f"{output_dir}/tmp_{chrom}_{start}_{end}/02extract_features/{sample}/{sample}_{chrom}_{start}_{end}.csv"
        input_bed_gz = f"{output_dir}/tmp_{chrom}_{start}_{end}/01initial_hard_filter/{sample}/{sample}_{chrom}_{start}_{end}_sorted.bed.gz"
        output3 = f"{output_dir}/tmp_{chrom}_{start}_{end}/03final_hard_filter/{sample}/{sample}_{chrom}_{start}_{end}.txt"
        output4 = f"{output_dir}/{sample}/{sample}_{chrom}_{start}_{end}.csv"
        tmp_dir = f"{output_dir}/tmp_{chrom}_{start}_{end}"
    else:
        output1 = f"{output_dir}/tmp/01initial_hard_filter/{sample}/{sample}.bed"
        output2 = f"{output_dir}/tmp/02extract_features/{sample}/{sample}.csv"
        input_bed_gz = f"{output_dir}/tmp/01initial_hard_filter/{sample}/{sample}_sorted.bed.gz"
        output3 = f"{output_dir}/tmp/03final_hard_filter/{sample}/{sample}.txt"
        output4 = f"{output_dir}/{sample}/{sample}.csv"
        tmp_dir = f"{output_dir}/tmp"

    # Step 1: Initial Hard Filter
    if file_exists_and_not_empty(output1):
        print(f"{output1} already exists.", file=sys.stderr)
    else:
        print("Running initial_hard_filter...", file=sys.stderr)
        ensure_dir(output1)
        run_initial_hard_filter(
            vcf=vcf,
            sample_name=sample,
            output_file=output1,
            pop_gi_file=pop_info,
            recurrent_filter_file=recurrent_info,
            posterior_filter=p, mutant_dp_filter=m, upper_vaf_filter=u, lower_vaf_filter=l, min_depth_filter=d, max_mutation_length=ms, pop_gi_filter=pgi, recurrent_filter=r, callable_filter=cf
        )
        print("initial_hard_filter DONE.", file=sys.stderr)


    # Step 2: Extract Features
    if file_exists_and_not_empty(output2):
        print(f"{output2} already exists.", file=sys.stderr)
    elif file_exists_and_not_empty(output1):
        print("Running extract_features...", file=sys.stderr)
        ensure_dir(output2)
        run_extract_features(
            metadata=metadata,
            reference_genome=reference_genome,
            bed_panel=input_bed_gz,
            output_dir=f"{tmp_dir}/02extract_features/{sample}",
            sample_name=sample,
            variant_info=vcf,
            stutter_model_in=stutter_model,
            mappability_annotation=mappability,
            chrom=chrom,
            start=start,
            end=end
        )
        print("extract_features DONE.", file=sys.stderr)
    else:
        print(f"Warning: No loci of {sample} in {chrom}:{start}-{end} passed the filter.", file=sys.stderr)
        return


    # Step 3: Final Hard Filter
    if file_exists_and_not_empty(output3):
        print(f"{output3} already exists.", file=sys.stderr)
    elif file_exists_and_not_empty(output2):
        print("Running final_hard_filter...", file=sys.stderr)
        ensure_dir(output3)
        run_final_hard_filter(
            input_file=output2,
            output_file=output3,
            reference_genome=reference_genome,
            mis_or_indel=mis_or_indel,
            mode=mode
        )
        print("Final hard filter DONE.", file=sys.stderr)
    else:
        print(f"Warning: No loci of {sample} in {chrom}:{start}-{end} passed the filter.", file=sys.stderr)
        return


    # Step 4: Cell Level Filter
    if file_exists_and_not_empty(output4):
        print(f"{output4} already exists.", file=sys.stderr)
    elif file_exists_and_not_empty(output3):
        print("Running cell_level_filter...", file=sys.stderr)
        ensure_dir(output4)
        run_cell_level_filter(
            input_csv=output3,
            sample_name=sample,
            cb_list_tsv=cell_barcode,
            filters_json=filters_json, 
            mutation_type=mutation_type,
            output_file=output4
        )
        print("Cell level filter DONE.", file=sys.stderr)
    else:
        print(f"Warning: No loci of {sample} in {chrom}:{start}-{end} passed the filter.", file=sys.stderr)

    if not file_exists_and_not_empty(output4):
        print(f"Warning: No loci of {sample} in {chrom}:{start}-{end} passed the filter.", file=sys.stderr)

    if not keep_temp:
        shutil.rmtree(tmp_dir)
        print(f"Temporary directory has been deleted: {tmp_dir}")



if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(description="Run the full mosaic filtering pipeline for a sample and region.")
    parser.add_argument("--sample", required=True, help="Sample name")
    parser.add_argument("--reference_genome", required=True, help="Path to reference genome (FASTA)")
    parser.add_argument("--vcf", required=True, help="Input VCF file (mosaic_fraction_estimation_results.vcf.gz)")
    parser.add_argument("--stutter_model", required=True, help="Stutter result BED file")
    parser.add_argument("--mappability", required=True, help="Mappability BED file")
    parser.add_argument("--metadata", required=True, help="Features metadata CSV")
    parser.add_argument("--cell_barcode", default=None, help="Cell barcode list file")
    parser.add_argument("--pop_info", required=False, help="Population info file (pop_infors_output.txt.gz)")
    parser.add_argument("--recurrent_info", required=False, help="Recurrent mosaic info file")
    parser.add_argument("--chrom", type=str, default="", help="Chromosome (e.g., chr6)")
    parser.add_argument("--start", type=int, default=0, help="Genomic region start")
    parser.add_argument("--end", type=int, default=1000000000, help="Genomic region end")
    parser.add_argument("--output_dir", default="./04filter", help="Root output directory")
    parser.add_argument("--filters_json", default=None, help="Json file for filtering thresholds. Default path is src/filters.json.")
    parser.add_argument("--mis_or_indel", default="both", choices=["both", "snv", "indel"])
    parser.add_argument("--mode", default="all", choices=["rf", "hard_filter", "both", "either", "all"])
    parser.add_argument('--mutation_type', required=False, default="both", choices=["both", "cell_specific", "share"], help='Type of mutation')
    parser.add_argument('--keep_temp', required=False, default=True, help='Whether to keep the temporary files.')
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--m", type=float, default=0)
    parser.add_argument("--u", type=float, default=0.8)
    parser.add_argument("--l", type=float, default=1e-8)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--ms", type=int, default=50)
    parser.add_argument("--pgi", type=float, default=0.3)
    parser.add_argument("--r", type=float, default=0.3)
    parser.add_argument("--cf", type=int, default=0)

    args = parser.parse_args()


    run(
        sample=args.sample,
        chrom=args.chrom,
        start=args.start,
        end=args.end,
        reference_genome=args.reference_genome,
        vcf=args.vcf,
        stutter_model=args.stutter_model,
        cell_barcode=args.cell_barcode,
        mappability=args.mappability,
        metadata=args.metadata,
        pop_info=args.pop_info,
        recurrent_info=args.recurrent_info,
        output_dir=args.output_dir,
        filters_json=args.filters_json,
        mis_or_indel=args.mis_or_indel,
        mode=args.mode,
        mutation_type=args.mutation_type,
        keep_temp=args.keep_temp,
        p=args.p, m=args.m, u=args.u, l=args.l, d=args.d, ms=args.ms,
        pgi=args.pgi, r=args.r, cf=args.cf
    )