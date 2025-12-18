#!/bin/bash

bayesmonstr-atac stutter \
    --metadata ./resources/stutter_metadata.csv \
    --reference-genome ./resources/Homo_sapiens_assembly38.fasta \
    --bed-panel ./resources/hg38.hipstr_reference_0based_Human_STR_1232500.bed.gz \
    --output-dir ./00stutter \
    --chrom chr6 \
    --start 43243669 \
    --end 43243695 \

cd ./00stutter
cat results/* >> stutter_result.txt
sort -k6,6 -u stutter_result.txt >> stutter_result_uniq.txt
sort -k1,1 -k2,2n -k3,3n stutter_result_uniq.txt >> stutter_result_uniq_sorted.bed
bgzip stutter_result_uniq_sorted.bed
tabix -p bed -f stutter_result_uniq_sorted.bed.gz