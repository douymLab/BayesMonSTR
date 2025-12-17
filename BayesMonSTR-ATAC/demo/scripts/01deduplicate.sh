#!/bin/bash

for sample in demo1 demo2 demo3;do
    mkdir -p ./01deduplicate/${sample}
    bayesmonstr-atac deduplicate \
        --input-bam ./resources/bam/${sample}.chr6-43243669-43243695.bam \
        --bed-panel ./resources/hg38.hipstr_reference_0based_Human_STR_1232500.bed.gz \
        --output-bam ./01deduplicate/${sample}/${sample}.chr6-43243669-43243695.deduplicated.bam
done