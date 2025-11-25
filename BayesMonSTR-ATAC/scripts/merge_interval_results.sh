#!/bin/bash

# 检查参数是否正确
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_directory> <output_directory> <file_prefix>"
    exit 1
fi

# 输入目录、输出目录和文件前缀
INPUT_DIR=$1
OUTPUT_DIR=$2
PREFIX=$3

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR does not exist."
    exit 1
fi

# 检查输出目录是否存在，不存在则创建
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# # 将输入目录下所有文件中的非注释行（不以 # 开头的行）合并到一个文件中
# zcat "$INPUT_DIR"/*sorted.vcf.gz | grep -v '^#' | sort -k1,1 -k2,2n  >> "$OUTPUT_DIR/${PREFIX}_no_header_sorted.vcf"

# # 去除文件中的重复行，并保存到没有头的文件
# sort -u -k3,3 "$OUTPUT_DIR/${PREFIX}.txt" > "$OUTPUT_DIR/${PREFIX}_no_header.vcf"

# # 按照位置对没有头的 VCF 文件进行排序
# sort -k1,1 -k2,2n "$OUTPUT_DIR/${PREFIX}.txt" > "$OUTPUT_DIR/${PREFIX}_no_header_sorted.vcf"

FIRST_FILE=$(ls "$INPUT_DIR"/*sorted.vcf.gz | head -n 1)
FIRST_FILE_PATH="$INPUT_DIR/$FIRST_FILE"
# 提取输入目录中的第一个文件的头部信息（假设所有文件的头部相同）
grep ^# ${FIRST_FILE_PATH} > "$OUTPUT_DIR/${PREFIX}_header_sorted.vcf"

# 将排序后的无头文件添加到头文件后面
zcat "$INPUT_DIR"/*sorted.vcf.gz | grep -v '^#' | sort -k1,1 -k2,2n  >> "$OUTPUT_DIR/${PREFIX}_header_sorted.vcf"

# 压缩最终的 VCF 文件
bgzip "$OUTPUT_DIR/${PREFIX}_header_sorted.vcf"

# 为压缩后的 VCF 文件生成索引
tabix -p vcf "$OUTPUT_DIR/${PREFIX}_header_sorted.vcf.gz"

echo "Processing completed. Output saved to $OUTPUT_DIR"
