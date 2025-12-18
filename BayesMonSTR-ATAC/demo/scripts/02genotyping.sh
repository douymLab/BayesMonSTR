#!/bin/bash

bayesmonstr-atac genotyping \
    --metadata ./resources/genotyping_metadata.csv \
    --reference-genome ./resources/Homo_sapiens_assembly38.fasta \
    --bed-panel ./resources/hg38.hipstr_reference_0based_Human_STR_1232500.bed.gz \
    --stutter-model ./00stutter/stutter_result_uniq_sorted.bed.gz \
    --output-dir ./02genotyping \
    --chrom chr6 \
    --start 43243669 \
    --end 43243695 \

cd ./02genotyping/results
micromamba run -n bayesmonstr-atac bgzip hg38_chr6_43243669_43243695_mosaic_calling.vcf
micromamba run -n bayesmonstr-atac tabix -p vcf -f hg38_chr6_43243669_43243695_mosaic_calling.vcf.gz
micromamba run -n bayesmonstr-atac bcftools sort -Oz -o hg38_chr6_43243669_43243695_mosaic_calling.sorted.vcf.gz hg38_chr6_43243669_43243695_mosaic_calling.vcf.gz
micromamba run -n bayesmonstr-atac tabix -p vcf -f hg38_chr6_43243669_43243695_mosaic_calling.sorted.vcf.gz

first_file=true
> "mosaic_fraction_estimation_results.vcf"
for file in ./*_mosaic_calling.sorted.vcf.gz; do
    if [[ ! -f "$file" ]]; then
        echo "Warning: No result found"
        continue
    fi
    if $first_file; then
        zcat "$file" >> mosaic_fraction_estimation_results.vcf
        first_file=false
    else
        zcat "$file" | grep -v '^#' >> mosaic_fraction_estimation_results.vcf
    fi
done
micromamba run -n bayesmonstr-atac bgzip mosaic_fraction_estimation_results.vcf
micromamba run -n bayesmonstr-atac tabix -p vcf -f mosaic_fraction_estimation_results.vcf.gz