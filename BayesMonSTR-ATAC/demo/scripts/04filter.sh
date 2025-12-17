#!/bin/bash

for sample in demo1 demo2 demo3;do
  echo "Running $sample..." >&2
  bayesmonstr-atac filter \
    --sample $sample \
    --reference-genome ./resources/Homo_sapiens_assembly38.fasta \
    --vcf ./02genotyping/results/mosaic_fraction_estimation_results.vcf.gz \
    --stutter-model ./00stutter/stutter_result_uniq_sorted.bed.gz \
    --cell-barcode ./resources/cell_barcode/${sample}_cell_barcode.tsv \
    --mappability ./resources/hg38_k24_k100_mappability.bed.gz \
    --metadata ./resources/filter_metadata.csv \
    --keep-temp
    ### In the demo, due to the small data size, estimating population parameters is prone to bias; 
    ### therefore, population-related filtering has not been applied here. 
    ### For real datasets with a sufficiently large sample size, it is recommended to include population-level filtering.
    # --pop-info ./03population/pop_infors_output.txt.gz \
    # --recurrent-info ./03population/pop_infors_output_mosaic_recurrent_info.txt.gz \
  echo "Done." >&2
done