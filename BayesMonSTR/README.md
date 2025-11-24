<h1 align="center">BayesMonSTR</h1>

<p align="center">A tool for mosaic STR mutation detection from single-cell data.</p>

## Table of contents

- [Table of contents](#table-of-contents)
- [Install](#install)
- [Resource](#resource)
- [Usage](#usage)
- [Demo](#demo)
- [Docker](#docker)
- [Q\&A](#qa)

## Install

BayesMonSTR can be installed directly by

```shell
git clone https://github.com/douymLab/BayesMonSTR.git

cd BayesMonSTR/

pip install python/
```

or to an new conda environment by

```shell
conda create -n BayesMonSTR -y -c conda-forge -c bioconda \
    biopython \
    duckdb \
    GPy \
    matplotlib \
    "numpy>=1.8" \
    pandas \
    psutil \
    "pysam>=0.21.0" \
    rich \
    scikit-learn \
    scipy \
    statsmodels \
    tqdm

conda activate BayesMonSTR

git clone https://github.com/douymLab/BayesMonSTR.git

cd BayesMonSTR/

pip install python/
```

to check installation, try

```shell
BayesMonSTR --version
```

see [Docker](#docker) for docker usage.

## Resource

- Download resources for human genome assembly GRCh37/hg19:

```shell
# human genome reference
wget ftp://gsapubftp-anonymous@ftp.broadinstitute.org/bundle/b37/human_g1k_v37_decoy.fasta.gz
wget ftp://gsapubftp-anonymous@ftp.broadinstitute.org/bundle/b37/human_g1k_v37_decoy.fasta.fai.gz
gunzip human_g1k_v37_decoy*.gz

# STR reference panel
wget https://github.com/wenx00/BayesMonSTR-resources/raw/refs/heads/master/human/reference_GRCh37.bed

# STR stutter profile
wget https://github.com/wenx00/BayesMonSTR-resources/raw/refs/heads/master/human/stutter_GRCh37.csv

# STR population allele frequency database
wget https://github.com/wenx00/BayesMonSTR-resources/raw/refs/heads/master/human/hap_freqs.db

```

- Download resources for mouse genome assembly GRCm38/mm10:

```shell
# mouse genome reference
wget https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz
gunzip mm10.fa.gz
samtools faidx mm10.fa

# STR reference panel
wget https://github.com/wenx00/STR-resources/raw/refs/heads/master/mouse/reference_mm10.bed

# STR stutter profile
wget https://github.com/wenx00/STR-resources/raw/refs/heads/master/mouse/stutter_mm10.csv
```

## Usage

```shell
$ BayesMonSTR --help

usage: BayesMonSTR [-h] [-D] [--nobulk] [--ra] [-n NUM_WORKERS] -r REF -i INFO [-a AB_INFO] [-p PHASING [PHASING ...]] [-g SC_INFO [SC_INFO ...]] [-f FREQ] -b REGION [-S STUTTER]
                   [-m MODE] [-C] [-Q] [-U UNPHASE] [-O VCF] [-V]

BayesMonSTR

options:
  -h, --help            show this help message and exit
  -D, --debug           Use debug mode (default: False)
  --nobulk              Ignore bulk data (default: False)
  --ra                  Not adding reference allele automatically (default: False)
  -n NUM_WORKERS, --num_workers NUM_WORKERS
                        Number of workers used for processing, use all workers if -1 (default: 1)
  -r REF, --ref REF     Reference genome FASTA/FA file (required)
  -i INFO, --info INFO  Metadata for BAM/CRAM/SAM files and sample infomation (required)
  -a AB_INFO, --ab_info AB_INFO
                        Allelic imbalance information for WGA samples for read-based phasing
  -p PHASING [PHASING ...], --phasing PHASING [PHASING ...]
                        Nearby germline hSNP information for bulk samples
  -g SC_INFO [SC_INFO ...], --sc_info SC_INFO [SC_INFO ...]
                        Nearby germline hSNP information for WGA samples
  -f FREQ, --freq FREQ  STR population allele frequency database.
  -b REGION, --region REGION
                        STR reference region BED file (required)
  -S STUTTER, --stutter STUTTER
                        STR stutter error model output filename (required)
  -m MODE, --mode MODE  STR mutation calling mode, do not consider mosaic allele in bulk samples (sp) or consider mosaic allele in bulk samples (se)
  -C, --coord           Do not use HMM segmentation
  -Q, --quiet           Avoid verbose output
  -U UNPHASE, --unphase UNPHASE
                        Unphase STR variant call output filename
  -O VCF, --vcf VCF     STR variant call output filename
  -V, --version         show program's version number and exit
```

### Required arguments

#### -r REF, --ref REF

Path to the reference genome file, e.g., human_g1k_v37_decoy.fasta.

#### -i INFO, --info INFO

Path to the sample information file, formatted as csv file like:

```csv
individual,cell,type,dp,path
BR1,BR1-bulk,bulk,30,/path/to/BR1-bulk.bam
BR1,BR1-mda-1,mda,30,/path/to/BR1-mda-1.bam
BR1,BR1-mda-2,mda,30,/path/to/BR1-mda-2.bam
BR1,BR1-pta-1,pta,30,/path/to/BR1-pta-1.bam
BR1,BR1-pta-2,pta,30,/path/to/BR1-pta-2.bam
BR2,BR2-bulk,bulk,30,/path/to/BR2-bulk.bam
BR2,BR2-mda-1,mda,30,/path/to/BR2-mda-1.bam
BR2,BR2-scc-1,scc,30,/path/to/BR2-scc-1.bam
```

`type` specifies the sample type: `'bulk'` for bulk samples, `'mda'` for MDA-amplified samples, `'pta'` for PTA-amplified samples, and `'scc'` for single-cell-derived colony data. `dp` indicates the average sequencing depth of the alignment file.

#### -b REGION, --region REGION

Path to the STR reference panel, e.g., reference_GRCh37.bed, formatted as tsv file like:

| chromosome | start    | end      | motif length | total length | name             | motif | mappability |
| ---------- | -------- | -------- | ------------ | ------------ | ---------------- | ----- | ----------- |
| 16         | 72671356 | 72671367 | 1            | 12.0         | Human_STR_545109 | A     | 0.897727    |

**without** header.

#### -S STUTTER, --stutter STUTTER

Path to the STR stutter error profile, e.g., stutter_GRCh37.csv. It is recommended to use the provided stutter profile if the input samples are from a small cohort. If this argument points to a non-existent file, STR stutter parameters will be calculated using the EM algorithm and output to that location.

### optional arguments

#### -O VCF, --vcf VCF

Path to STR variant calling output.

#### -f FREQ, --freq FREQ

Path to the STR population allele frequency database, e.g., hap_freqs.db. It is recommended to use the provided allele frequency database if the input samples are from a small cohort. If this argument is left empty, STR allele frequencies will be calculated using the EM algorithm instead of the provided database.

#### -p PHASING

By default, BayesMonSTR automatically searches for germline hSNPs. However, to specify hSNP loci, use this argument and provide a phased VCF file containing germline hSNPs for bulk data.

#### -g SC_INFO

Path to the phased VCF file containing germline hSNPs for single-cell data. This is used only for MDA-amplified samples; ignore this argument when using PTA-amplified or single-cell-derived colony data.

#### -a AB_INFO

Path to allele frequency (AF) values regressed using Gaussian process regression for MDA-amplified samples. This argument should be ignored when using PTA-amplified or single-cell-derived colony data.

## Demo

A demo for running BayesMonSTR is provided in the `demo/` directory, which includes sample input files. Users can refer to this example for the expected input format.

```shell
cd demo/

# run genotyping
BayesMonSTR -b GRCh37_reference_STR.bed \
    -i metadata.csv \
    -r human_g1k_v37_decoy.fasta \
    -S stutter.csv \
    --freq hap_freqs.db \
    -p BR1.phased.vcf.gz \
    -g BR1.phased.vcf.gz \
    -a merged_af.csv \
    -O str_gt

mkdir -p results

# run filtering for individual 1 (i.e., the first individual listed in metadata.csv)
python ../scripts/mosaic.py --individual 1 -i str_gt.vcf -o results
```

The final mosaic call set is provided in the `results/*.csv` files. Below is a description of key columns in these files:

| column                      | description                                                                                                                      |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| chr                         | Chromosome                                                                                                                       |
| pos                         | Genomic position                                                                                                                 |
| id                          | STR locus name                                                                                                                   |
| gt_mosaic                   | Complete mosaic genotype                                                                                                         |
| mo_posterior                | Mosaic posterior                                                                                                                 |
| mut_cell_list               | Mutant cell indicator (e.g., `(0, 1, 0, 0)` indicates that only the second cell is mutant, based on the order in `metadata.csv`) |
| mut_refalt                  | Mutation information formatted as `(chr, start, end, ref, alt)`                                                                  |
| phase_filter/unphase_filter | Indicates whether the locus passed filtering (`True` for true loci).                                                             |
| prediction                  | ML prediction results, (`mosaic` for true loci  )                                                                                |

## Docker

BayesMonSTR can be ran with Docker as well.

build docker image by

```shell
git clone https://github.com/douymLab/BayesMonSTR.git

cd BayesMonSTR/docker

sudo docker build -f docker/Dockerfile -t bayesmonstr .
```

run BayesMonSTR by

```shell
mkdir -p results

# run genotyping
sudo docker run -it --rm \
    -v ./demo/:/demo \
    -v ./results:/results \
    -w /demo \
    bayesmonstr \
    BayesMonSTR \
        -b GRCh37_reference_STR.bed \
        -i metadata.csv \
        -r human_g1k_v37_decoy.fasta \
        -S stutter.csv \
        --freq hap_freqs.db \
        -p BR1.phased.vcf.gz \
        -g BR1.phased.vcf.gz \
        -a merged_af.csv \
        -O /results/str_gt

# run filtering
sudo docker run -it --rm \
    -v ./demo/:/demo \
    -v ./results:/results \
    -w /demo \
    bayesmonstr \
    python /scripts/mosaic.py \
        -i /results/str_gt.vcf \
        -o /results
```

## Q&A

1. **Can I accelerate the genotyping?**
**Yes**, you can split STR reference panel file into *N* files and run them in parallel.

To be continued ...
