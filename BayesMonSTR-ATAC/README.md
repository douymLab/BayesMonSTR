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
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Installation
To set up the environment and install the necessary dependencies, follow the instructions below:
```bash
git clone https://github.com/douymLab/BayesMonSTR
conda env create -f BayesMonSTR-ATAC/environment.yaml
conda activate BayesMonSTR-ATAC
pip install -e BayesMonSTR-ATAC
```

---

## Demo
A **demo dataset** is available in the `demo` directory, which can be used for testing the tool and familiarizing yourself with the workflow. Before running the demo, please download the reference genome file and ensure it is available in the  `demo/resources/` directory—either as a direct copy or via a symbolic link. To run the demo, navigate to the `demo` directory and execute the scripts directly from the `scripts` directory. In the demo, due to the small data size, estimating population parameters is prone to bias; therefore, population-related filtering has not been applied. For real datasets with a sufficiently large sample size, it is recommended to include population-level filtering.

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
* `--cell-barcode, -cb` : The TSV file containing two columns: `cell barcode` and `cell type`, where each row assigns a predicted or annotated cell type to a specific cell barcode.
* `--pop-info, -pi` : The population information file generated from population information extraction step
* `--recurrent-info, -ri` : The recurrent mosaic information file generated from population information extraction step
* `--mappability, -mp` : The BED file containing mappability information (K24 and K100) for regions in the reference genome. You can directly use `demo/resource/hg38.hipstr_reference_0based_Human_STR_1232500.bed.gz`.
* `--metadata, -i` :  The metadata CSV file used for genotyping can be reused.
* `--chrom, -c` : Chromosome (default: empty)
* `--start, -s` : Genomic start position (default: `0`)
* `--end, -e` : Genomic end position (default: `1000000000`)
* `--filters-json, -fj` : Json file for filtering thresholds. Default path is `src/filters.json`. In the provided JSON file, there are explanations for each filter, based on which you can modify the threshold of the filters.
* `--mutation-type, -mt` : Expected type of mutation. Select from cell specific, share and both (default: `both`).
* `--output-dir, -o` : Output directory (default: `./04filter`)

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
