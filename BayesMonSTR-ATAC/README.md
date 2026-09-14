## BayesMonSTR-ATAC

**BayesMonSTR-ATAC** is a computational method for detecting **mosaic mutations** in **short tandem repeat (STR) regions** from **single-nucleus ATAC-sequencing data**. It enables sensitive identification of repeat contractions, expansions, and interruption-type mosaic mutations within STR loci.
It provides an end-to-end workflow including read deduplication, stutter model estimation, mosaic genotyping, population information extraction, and post-calling filtering.

- [BayesMonSTR-ATAC](#bayesmonstr-atac)
- [Installation](#installation)
- [Demo](#demo)
- [Considerations](#considerations)
- [BayesMonSTR-ATAC workflow](#bayesmonstr-atac-workflow)
  - [1. Estimation of locus-based stutter model (all samples)](#1-estimation-of-locus-based-stutter-model-all-samples)
  - [2. Removing duplicate reads from snATAC-seq data (single sample)](#2-removing-duplicate-reads-from-snatac-seq-data-single-sample)
  - [3. Estimation of Mosaic Fraction and Mosaic Genotyping (all samples)](#3-estimation-of-mosaic-fraction-and-mosaic-genotyping-all-samples)
  - [4. Extraction of population information (all samples)](#4-extraction-of-population-information-all-samples)
  - [5. Filtering for Each Sample (single sample)](#5-filtering-for-each-sample-single-sample)
  - [6. Combination of chunked results (all sample)](#6-Combination-of-chunked-results-all-samples)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Installation
To set up the environment and install the necessary dependencies, follow the instructions below:
```bash
git clone https://github.com/douymLab/BayesMonSTR
conda env create -f BayesMonSTR-ATAC/environment.yml
conda activate bayesmonstr-atac
pip install -e BayesMonSTR-ATAC
```

---

## Demo
A demo dataset is provided in the `demo` directory for testing the tool and becoming familiar with the workflow. Before running the demo, download the reference genome file and ensure that `Homo_sapiens_assembly38.fasta` is present in the `demo/resources/` directory, either as a direct file or a symbolic link. To execute the demo, set the `demo` directory as the working directory and run the scripts directly from the `scripts` directory. These scripts can also serve as templates for processing your own data. Note that due to the small size of the demo dataset, population parameter estimation may be biased; hence, population-level filtering has been omitted. For real datasets with adequate sample sizes, it is recommended to apply population-level filtering.

---

## BayesMonSTR-ATAC workflow

Run BayesMonSTR-ATAC:

```bash
bayesmonstr-atac --help
```

You will see the available subcommands:

```text
stutter        Run stutter model estimation
deduplicate   Remove duplicate reads from snATAC-seq data
genotyping    Run mosaic genotyping
pop            Extract population information from BulkMonSTR VCF
filter         Filter genotyping results
```

Each subcommand has its own options and help message.

---


### 1. Estimation of locus-based stutter model (all samples)

**Notice:** It is important to emphasize that if the mosaic mutation is recurrent across different samples, the **stutter error rate** may be **overestimated**.
We recommend using more than **20 unrelated individuals** or samples (**avoid much recurrent mutations**) with a sequencing depth of at least **30×** to estimate the stutter error model accurately.

```bash
bayesmonstr-atac stutter \
  --metadata metadata.csv \
  --reference-genome reference.fa \
  --bed-panel str_panel.bed.gz \
  --output-dir output_dir
```

**Options**

* `--metadata, -i` : Metadata CSV file (Sample name and bam_path must be specific.) 

    | Ind              | Sample Name              | Sex     | Tissue  | sequencing_type | bam_path                                                                 | mosdepth_wgs_mean_depth | used_genotyping_str_mean_depth | seq_tech |
    |------------------|------------------|--------|---------|----------------|------------------------------------------------------------------------|-------------------------|--------------------------------|---------|
    | SRR13873087 | SRR13873087 | male | LCL | WGS            | ./resources/stutter_bam/SRR13873087_chr6_43243669_43243695.bam | 300                    | 300                           | illumina |

* `--reference-genome, -r` : Reference genome FASTA
* `--bed-panel, -b` : STR annotation BED file. 
    We use the **STR reference panel** from **HipSTR**, which can be downloaded from: [HipSTR Reference Panel](https://github.com/HipSTR-Tool/HipSTR-references). The HipSTR panel uses a **1-based** coordinate system. If you are using this panel, you must **convert it to a 0-based format** by subtracting **1** from the values in the second column. Alternatively, you can format your own STR loci following the **HipSTR BED file format**.  
* `--output-dir, -o` : Output directory.
    The stutter model results are saved in the following directory: `${output-dir}/results`,and the key columns are as following:

    | Key Columns | Description |
    |--------------|-------------|
    | `1th column`  | chromosome |
    | `2th column`   | zero-base STR start(close) |
    | `3th column`      | zero-base STR end(open) |
    | `4th column`   | Motif length |
    | `5th column`    | Period |
    | `6th column` | STR id |
    | `7th column` | Motif |
    | `23th column`  | In-frame insertion rate |
    | `24th column`   | In-frame deletion rate |
    | `25th column`      | In-frame step-size rate |
    | `26th column`   | Out-frame insertion rate |
    | `27th column`    | Out-frame deletion rate |
    | `28th column` | Out-frame step-size rate |
* `--loglevel, -ll` : Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)
* `--log-to-file, -lf` : Whether save log to file (default: `True`)
* `--chrom, -c` : Chromosome (default: empty)
* `--start, -s` : Genomic start position (default: `0`)
* `--end, -e` : Genomic end position (default: `1000000000`)
* `--threads, -t` : Number of threads to use (default: -1, use all available cores)

If the interval regions are processed in parallel, you should merge all the stutter results from the different intervals.

```sh
cat * >> stutter_result.txt
sort -k6,6 -u stutter_result.txt >> stutter_result_uniq.txt
sort -k1,1 -k2,2n -k3,3n stutter_result_uniq.txt >> stutter_result_uniq_sorted.bed
bgzip stutter_result_uniq_sorted.bed
tabix -p bed stutter_result_uniq_sorted.bed.gz
```
---

### 2. Removing duplicate reads from snATAC-seq data (single sample)

Remove duplicate reads from snATAC-seq BAM files.

```bash
bayesmonstr-atac deduplicate \
  --input-bam input.bam \
  --bed-panel str_panel.bed.gz \
  --output-bam dedup.sorted.bam
```

**Options**

* `--input-bam, -i` : Input BAM file
* `--bed-panel, -b` : STR BED file
* `--output-bam, -o` : Output sorted BAM file
* `--threads, -t` : Number of threads to use (default: -1, use all available cores)

---

### 3. Estimation of Mosaic Fraction and Mosaic Genotyping (all samples)

Perform mosaic STR mutation calling.

```bash
bayesmonstr-atac genotyping \
  --metadata metadata.csv \
  --reference-genome reference.fa \
  --bed-panel str_panel.bed.gz \
  --stutter-model stutter.bed.gz \
  --output-dir results/
```

**Options**

* `--metadata, -i` : Adapt the metadata CSV from the stutter error estimation step by updating the sample names and BAM paths to your input samples and deduplicated BAM files.
* `--reference-genome, -r` : Reference genome FASTA
* `--bed-panel, -b` : STR annotation BED
* `--output-dir, -o` : Output directory
* `--stutter-model, -s` : The stutter model generated from the previous step, which is used to help estimate the stutter errors in the STR regions. This file should be compressed with bgzip and indexed with tabix.
* `--loglevel, -ll` : Logging level
* `--log-to-file, -lf` : Whether save log to file (default: `True`)
* `--chrom, -c` : Chromosome (default: empty)
* `--start, -s` : Genomic start position (default: `0`)
* `--end, -e` : Genomic end position (default: `1000000000`)
* `--threads, -t` : Number of threads to use (default: -1, use all available cores)

---

### 4. Extraction of population information (all samples)

If you have more than **20 unrelated samples**, we recommend leveraging population data to filter BayesMonSTR-ATAC outputs. This helps **eliminate common germline variants and recurrent noise**, such as mapping errors and stutter errors.
However, if mosaic STR mutations **recur in your samples due to selective advantage of driver mutations** across individuals, population-based filtering may **not be appropriate**. Alternatively, you can apply a lenient threshold in the **filter step**.

```bash
bayesmonstr-atac pop \
  --vcf-file mosaic_fraction_estimation_results.vcf.gz \
  --output-file pop_infors_output.txt
```

**Options**

* `--vcf-file, -v` : The VCF file generated from genotyping step
* `--output-file, -o` : Output file path

---

### 5. Filtering for Each Sample (single sample)

Apply multiple filters to mosaic genotyping results for a given sample and region.

```bash
bayesmonstr-atac filter \
  --sample SAMPLE1 \
  --reference-genome reference.fa \
  --vcf mosaic_fraction_estimation_results.vcf.gz \
  --stutter-model stutter_model.bed.gz \
  --cell-barcode barcodes.txt \
  --mappability mappability.bed \
  --metadata features.csv \
  --pop-info population_info.txt \
  --recurrent-info recurrent.txt
```

**Options**

* `--sample, -sp` : Sample name
* `--reference-genome, -r` : Reference genome FASTA
* `--vcf, -v` : The VCF file generated from genotyping step
* `--stutter-model, -sm` : The stutter model generated from stutter model estimation step
* `--pop-info, -pi` : The population information file generated from population information extraction step
* `--recurrent-info, -ri` : The recurrent mosaic information file generated from population information extraction step
* `--mappability, -mp` : The BED file containing mappability information (K24 and K100) for regions in the reference genome. You can directly use `demo/resource/hg38.hipstr_reference_0based_Human_STR_1232500.bed.gz`.
* `--metadata, -i` :  The metadata CSV file used for genotyping can be reused.
* `--cell-barcode, -cb` : The TSV file containing two columns: `cell barcode` and `cell type`, where each row assigns a predicted or annotated cell type to a specific cell barcode. If you provide the cell barcode list, candidate mutations of barcodes that do not appear in this list will be automatically excluded. If you wish to retain these loci, modify filter named `na_celltype_rc_mosaic` in `src/filters.json`. (default: `None`)
* `--chrom, -c` : Chromosome (default: empty)
* `--start, -s` : Genomic start position (default: `0`)
* `--end, -e` : Genomic end position (default: `1000000000`)
* `--filters-json, -fj` : JSON filter profile. The default `src/filters.json` applies `basic` rules first, then the rules under `by_mutation_type`, and finally `final` rules. Supply another file to choose your own rules or thresholds.
* `--mutation-type, -mt` : Expected type of mutation. Select from cell specific, share and both (default: `both`).
* `--output-dir, -o` : Output directory (default: `./04filter`)
* `--plot, -p` : Add `--plot` in command line to plot count of loci during filtering.
* `--keep-temp, -k` : Add `--keep-temp` in command line to keep the temporary files.

The mutation-aware filter profile uses this structure:

```json
{
  "version": 2,
  "basic": [{"column": "CMP", "operator": ">", "threshold": 0.5}],
  "by_mutation_type": {
    "INDEL": {"filters": [{"column": "stutter_ratio", "operator": "<", "threshold": 0.1}]},
    "Mismatch": {"filters": [{"column": "mut_mean_baseq", "operator": ">=", "threshold": 30}]}
  },
  "final": [{"column": "celltype_count_mosaic", "operator": "==", "threshold": 1}],
  "cohort": {
    "group_by": ["dataset", "str_id"],
    "by_mutation_type": {
      "INDEL": {"filters": [{"column": "allele_count", "operator": "<", "threshold": 4}], "max_group_count": 2},
      "Mismatch": {"filters": [{"column": "allele_count", "operator": "==", "threshold": 2}], "max_group_count": 1}
    }
  }
}
```

Top-level filters are combined sequentially with AND. For compound rules, use nested
`{"all": [...]}`, `{"any": [...]}`, or `{"not": {...}}`. Set `"enabled": false` on a
rule or mutation-type block to disable it without deleting it. The legacy JSON format
(a top-level list of filters) remains supported. In v2 profiles, missing columns raise an
error by default to prevent silent under-filtering; set `"on_missing_column": "skip"`
only when that behavior is intentional. A rule can use `"missing_values": "pass"` when
the column is present but unavailable values such as `NA` or `.` should not be filtered.

The `cohort` stage runs only in `bayesmonstr-atac combine`, after all row-level filters.
It derives `allele_count` from `AAD`, applies mutation-specific allele-count rules, then
removes a locus globally when its count in any `group_by` group exceeds
`max_group_count`. If `dataset` is not present, the default profile still applies the
allele-count rules and skips only recurrence removal with a warning.

---

### 6. Combine chunked results

If you run the genotyping using chunked genomic intervals, use this command to combine all the results or results by sample from the different intervals. 

```bash
bayesmonstr-atac combine \
  --input-dir filter_dir \
  --output-prefix final_results \
  --dataset DATASET_NAME
```

**Options**

* `--input-dir, -i` : Root input directory (default: `./04filter`)
* `--output-prefix, -o` : Prefix of outputs (default: `./04filter/results`)
* `--filters-json, -fj` : Same as filter step
* `--mutation-type, -mt` : Same as filter step
* `--dataset, -ds` : Dataset label used by cohort-level `dataset + str_id` recurrence filtering. Omit it when input rows already contain a `dataset` column.

---

## Citation

If you use **BayesMonSTR-ATAC** in your research, please cite the corresponding manuscript (to be added).

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or issues, please open an issue or contact wangchunyi@westlake.edu.cn.

---
