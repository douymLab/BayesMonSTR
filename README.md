# BayesMonSTR

A toolkit for mosaic STR mutation detection and visualization.

## Install

You can choose to run ./INSTALL.sh to install all four programs simultaneously, or install only the required program individually by following the instructions in each respective subdirectory.

## BayesMonSTR

A tool for mosaic STR mutation detection from **single-cell sequencing data**.

See [BayesMonSTR](BayesMonSTR/README.md) for details.

## BayesMonSTR-Bulk

A tool for mosaic STR mutation detection from **bulk sequencing data**.

See [BayesMonSTR-Bulk](BayesMonSTR-Bulk/README.md) for details.

## BayesMonSTR-ATAC

A tool for mosaic STR mutation detection from **single-nucleus ATAC-sequencing data**.

See [BayesMonSTR-ATAC](BayesMonSTR-ATAC/README.md) for details.

## INSIGHT

A tool for genome data **visualization**.

See [INSIGHT](INSIGHT/README.md) for details

## Docker

BayesMonSTR and BayesMonSTR-ATAC can be ran with Docker as well.

Build docker image by

```shell
git clone https://github.com/douymLab/BayesMonSTR.git

cd BayesMonSTR

docker build -t bayesmonstr:latest .
```

You can also download docker image from https://hub.docker.com/r/wenx00/bayesmonstr.

Load docker image by

```shell
docker load -i bayesmonstr_1.0.tar
```

Abtain help message by

```shell
docker run --rm bayesmonstr help
docker run --rm bayesmonstr bayesmonstr --help
docker run --rm bayesmonstr bayesmonstr-atac --help
docker run --rm -it bayesmonstr bash
```

Before running the demo, ensure that the reference genome files are downloaded. If not, visit https://github.com/broadinstitute/gatk/tree/master/src/test/resources/large to obtain them. Specifically, make sure the file `human_g1k_v37_decoy.fasta` is present in the `BayesMonSTR/demo/ directory` and `Homo_sapiens_assembly38.fasta` is available in the `BayesMonSTR-ATAC/demo/resources/` directory, either as a direct copy or a symbolic link.

Run BayesMonSTR demo by

```shell
cd {path_to_BayesMonSTR}

mkdir -p ./BayesMonSTR/results

# run genotyping
sudo docker run -it --rm \
    -v ./BayesMonSTR/demo/:/demo \
    -v ./BayesMonSTR/results:/results \
    -w /demo \
    bayesmonstr \
    bayesmonstr \
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
    -v ./BayesMonSTR/demo/:/demo \
    -v ./BayesMonSTR/results:/results \
    -v ./BayesMonSTR/scripts:/scripts \
    -w /demo \
    bayesmonstr \
    python /scripts/mosaic.py \
        -i /results/str_gt.vcf \
        -o /results
```

Run BayesMonSTR-ATAC demo by

```shell
cd {path_to_BayesMonSTR}

# Estimation of locus-based stutter model
docker run --rm \
    -v ./BayesMonSTR-ATAC/demo:/data \
    -w /data \
    bayesmonstr \
    bash /data/scripts/00stutter.sh

# Removing duplicate reads from snATAC-seq data
docker run --rm \
    -v ./BayesMonSTR-ATAC/demo:/data \
    -w /data \
    bayesmonstr \
    bash /data/scripts/01deduplicate.sh

# Estimation of Mosaic Fraction and Mosaic Genotyping
docker run --rm \
    -v ./BayesMonSTR-ATAC/demo:/data \
    -w /data \
    bayesmonstr \
    bash /data/scripts/02genotyping.sh

# Extraction of population information
docker run --rm \
    -v ./BayesMonSTR-ATAC/demo:/data \
    -w /data \
    bayesmonstr \
    bash /data/scripts/03population.sh

# Filtering for Each Sample
docker run --rm \
    -v ./BayesMonSTR-ATAC/demo:/data \
    -w /data \
    bayesmonstr \
    bash /data/scripts/04filter.sh
```

Because of permission restrictions in the Docker image, the warning
`[bgzip] Failed to set file specifications.` may occur, but since it does not affect the result, it can be safely ignored.