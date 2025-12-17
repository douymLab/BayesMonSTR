#!/bin/bash
mkdir -p ./03population
bayesmonstr-atac pop \
  --vcf-file ./02genotyping/results/mosaic_fraction_estimation_results.vcf.gz \
  --output-file ./03population/pop_infors_output.txt
  
bgzip ./03population/pop_infors_output.txt
tabix -p bed ./03population/pop_infors_output.txt.gz
bgzip ./03population/pop_infors_output_mosaic_recurrent_info.txt
tabix -p bed ./03population/pop_infors_output_mosaic_recurrent_info.txt.gz