import os
from in_sight.reads_manager import create_combined_read_manager
from in_sight.utils import parse_region, convert_pdf_to_png, get_memory_usage, HSNPInfo
from in_sight.utils import get_flanking_sequences, determine_file_type, get_file_mode
from in_sight.add_attr_info import hsnp_processor, str_data_processor, classify_allelic_reads_processor, str_data_processor_by_str_position
from in_sight.analysis.hmm_model import StrReads
from importlib import resources
import in_sight
import os
import subprocess
from typing import List, Tuple, Union, Dict
import time
import logging
import traceback
from typing import Optional
import pysam
from in_sight.str_utils import STRRegion
import pandas as pd
import ast

def run_r_visualization(base_df_path, read_df_path, reference_df_path, output_file="output.svg", 
                       mut_start=None, mut_end=None, mut_type=None, start_axis=None,
                       r_script_path=resources.files(in_sight).joinpath('r_script/plot_plain_with_full_features.R')):
    """
    Call Rscript for data visualization.

    Args:
        base_df_path (str): Path to base dataframe CSV file.
        read_df_path (str): Path to read dataframe CSV file.
        reference_df_path (str): Path to reference dataframe CSV file.
        output_file (str): Path to output image file (PNG).
        mut_start (int, optional): Mutation start position.
        mut_end (int, optional): Mutation end position.
        mut_type (str, optional): Mutation type.
        start_axis (int, optional): Start position of the reference sequence.

    Returns:
        None
    """
    # Ensure Rscript exists
    if not os.path.exists(r_script_path):
        raise FileNotFoundError(f"R script not found at {r_script_path}")

    # Prepare command arguments
    cmd = ["Rscript", r_script_path, 
           base_df_path, 
           read_df_path, 
           reference_df_path,
           output_file]
    
    # Add optional arguments if provided
    if mut_start is not None:
        cmd.append(str(mut_start))
        if mut_end is not None:
            cmd.append(str(mut_end))
            if mut_type is not None:
                cmd.append(str(mut_type))
                if start_axis is not None:
                    cmd.append(str(start_axis))

    # Call Rscript
    try:
        print(cmd)
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Rscript execution failed: {e}")

    print(f"Visualization saved to: {output_file}")

def str_visulization(bam_path: str,
                    region: Union[str, List[str], List[Tuple[str, str]]],
                    reference_fasta: str = None,
                    str_id: str = None,
                    str_region_info: STRRegion = None,
                    r_script_path: str = resources.files(in_sight).joinpath('r_script/plot_base.R'),
                    output_dir: str = "./",
                    prefix: str = "pileup_analysis",
                    run_visualization: bool = True,
                    mutation_region_label: str = None,  # 指定哪个区域标签用于突变范围
                    hsnp_region_label: str = None,     # 指定哪个区域应用hsnp处理器
                    hsnp_info: HSNPInfo = None,  # 添加hsnp信息参数
                    align_samples: bool = False,
                    start_axis: int = 1,
                    mut_type: str = "STR",
                    pro_type: str = None,
                    individual_code: str = None,
                    str_processor_type: str = 'str_position',
                    **kwargs):
    """
    Perform single base region analysis and generate visualizations
    
    Parameters:
        bam_path: Path to BAM/CRAM file
        region: Analysis region(s) in one of these formats:
               - Single string: "chrom:start-end" or "chrom:pos"
               - List of strings: ["chrom:start-end", "chrom:pos", ...]
               - List of tuples: [(region_string, label), ...] where each tuple contains a region string and a custom label
        reference_fasta: Path to reference genome
        str_id: STR ID
        str_region_info: Class for STR region
        r_script_path: Path to R script for visualization
        output_dir: Output directory (default current directory)
        prefix: Output file prefix (default 'pileup_analysis')
        run_visualization: Whether to execute R visualization (default True)
        mutation_region_label: Label of the region to use for mutation start/end coordinates (default None, which uses all regions)
        hsnp_region_label: Label of the region to apply hsnp processor (default None)
        hsnp_info: HSNP information dictionary for the hsnp processor (required if hsnp_region_label is specified)
        align_samples: Whether to align samples (default False)
        start_axis: Start position of the reference sequence (default 1)
        mut_type: Type of the mutation (default "STR")
        pro_type: Type of the sample (default None)
        individual_code: Individual code (default None)
        str_processor_type: Type of the str processor (default 'str_position')
        **kwargs: Additional parameters passed to sampler or read manager
    Returns:
        Dictionary containing generated DataFrames and file paths
    """
    # 处理不同类型的区域输入
    if isinstance(region, str):
        # 单个区域字符串
        regions_list = [(region, "analysis_region")]
    elif isinstance(region, list):
        if not region:
            raise ValueError("Region list cannot be empty")
        
        # 检查是否包含自定义标签（每个元素是元组）
        if isinstance(region[0], tuple):
            # 列表中的元组: [(region_string, label), ...]
            regions_list = region
        else:
            # 列表中的字符串: [region1, region2, ...]
            regions_list = [(reg, f"region_{i+1}") for i, reg in enumerate(region)]
    else:
        raise ValueError("Region must be a string, list of strings, or list of (string, label) tuples")
    
    # 解析所有区域
    parsed_regions = []
    chrom = None
    
    for reg_info in regions_list:
        reg_str, label = reg_info
        curr_chrom, start, end = parse_region(reg_str)
        
        # 确保所有区域都在同一条染色体上
        if chrom is None:
            chrom = curr_chrom
        elif chrom != curr_chrom:
            raise ValueError(f"All regions must be on the same chromosome. Found {chrom} and {curr_chrom}")
        
        # 添加到解析区域列表，使用提供的标签
        parsed_regions.append((start, end, label))
    
    # 检查hsnp相关参数
    if hsnp_region_label is not None:
        if hsnp_info is None:
            raise ValueError("hsnp_info must be provided when hsnp_region_label is specified")
            
        # 验证指定的hsnp区域标签存在
        if not any(label == hsnp_region_label for _, _, label in parsed_regions):
            raise ValueError(f"Region with label '{hsnp_region_label}' not found")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建分析管理器，使用解析后的多个区域
    manager = create_combined_read_manager(
        bam_path,
        chrom,
        parsed_regions,  # 现在传递多个区域及其自定义标签
        reference_fasta
    )

    # 处理数据
    manager.clear_processors()
    
    # 如果指定了hsnp_region_label，为对应区域添加hsnp处理器
    if hsnp_region_label is not None and hsnp_info is not None:
        # 找到指定标签的区域
        hsnp_regions = [r for r in parsed_regions if r[2] == hsnp_region_label]
        str_regions = [r for r in parsed_regions if r[2] == mutation_region_label]
        if hsnp_regions:
            hsnp_start, hsnp_end, _ = hsnp_regions[0]
            str_start, str_end, _ = str_regions[0]
            print(f"Adding HSNP processor for region '{hsnp_region_label}' ({hsnp_start}-{hsnp_end})")
            # 为该区域添加hsnp处理器
            manager.add_processor(hsnp_processor(hsnp_info, (str_start, str_end)))
            
    # 增加str_data
    ## protocol 1, use HMMSegger to get str_data
    if str_region_info is not None and str_id is not None:
        str_region = str_region_info
        if str_processor_type == 'hmms':
            str_id, str_reads = process_single_str_id(str_id, bam_path, reference_fasta, str_region)
            print(f"Adding STR data processor for region '{str_id}' by 'hmms'")
            manager.add_processor(str_data_processor(str_reads.str_data))
            # 增加classify_allelic_reads
            if str_id is not None and str_reads is not None and pro_type is not None and individual_code is not None:
                print(f"Adding classify_allelic_reads processor for region '{str_id}'")
                manager.add_processor(classify_allelic_reads_processor(str_reads.str_data, str_region_info, pro_type, individual_code = individual_code))
        elif str_processor_type == 'str_position':
            ## protocol 2, use str position to get str_data
            print(f"Adding STR data processor for region '{str_id}' by 'str_position'")
            manager.add_processor(str_data_processor_by_str_position(str_region_info))
        else:
            raise Warning(f"Unknown str processor type: {str_processor_type}, use 'hmms' or 'str_position'")
    
    manager.process_reads()
    
    manager.sort_reads('classify_allelic_reads.allele_source', 'snp.paired_hsnp_allele', 'str_info.STR_position', 'start', reverse=False)
    
    if align_samples:
        print(f"align_samples: {align_samples}")
        ref_start = min(r[0] for r in parsed_regions) - 150
        ref_end = max(r[1] for r in parsed_regions) + 150
        print(f"ref_start: {ref_start}, ref_end: {ref_end}")
    else:
        ref_start = None
        ref_end = None
    read_df, base_df, reference_df = manager.generate_dataframes(ref_start=ref_start, ref_end=ref_end)
    
    sampler_info = manager.sampler_info
    
    # 生成输出路径
    file_paths = None
    
    # 生成参考序列
    if not read_df.empty:  
        # 生成输出路径
        file_paths = {
            'base_df': os.path.join(output_dir, f"{prefix}_base.csv"),
            'read_df': os.path.join(output_dir, f"{prefix}_read.csv"),
            'reference_df': os.path.join(output_dir, f"{prefix}_reference.csv"),
            'plot': os.path.join(output_dir, f"{prefix}.pdf")
        }
        
        if align_samples:
            ## cut all dataframes to same reference position range
            base_df = base_df[(base_df['ref_pos'] >= ref_start) & (base_df['ref_pos'] <= ref_end)]
            reference_df = reference_df[(reference_df['ref_pos'] >= ref_start) & (reference_df['ref_pos'] <= ref_end)]

        # 保存文件
        base_df.to_csv(file_paths['base_df'], index=False)
        read_df.to_csv(file_paths['read_df'], index=False)
        reference_df.to_csv(file_paths['reference_df'], index=False)
    else:
        run_visualization = False
    
    # 执行可视化
    if run_visualization and file_paths:
        
        # 从 kwargs 中提取可能的可视化参数
        for key in ['mut_type', 'start_axis']:
            if key in kwargs:
                vis_kwargs[key] = kwargs[key]

        if str_region_info is not None:
            # Safely get mutation information
            mut_refalt_col = f'mut_refalt_{individual_code}'
            mut_info_col = f'mut_info_{individual_code}'
            additional_info_keys = str_region_info.additional_info.keys()

            if mut_refalt_col in additional_info_keys:
                mut_refalt = str_region_info.additional_info[mut_refalt_col]
                if pd.notna(mut_refalt) and mut_refalt != 'None':
                    try:
                        mut_refalt = ast.literal_eval(str(mut_refalt))
                        mut_start = mut_refalt[1]
                        if int(mut_start) is not None:
                            mut_start = int(mut_start) + 1
                        mut_end = mut_refalt[2]
                        if int(mut_end) is not None:
                            mut_end = int(mut_end) + 1
                    except (ValueError, SyntaxError, IndexError) as e:
                        logging.warning(f"Could not parse mut_refalt for {str_id}: {e}")
                        
            if mut_info_col in additional_info_keys:
                mut_info = str_region_info.additional_info[mut_info_col]
                if pd.notna(mut_info) and mut_info != 'None':
                    try:
                        mut_info = ast.literal_eval(str(mut_info))
                        mut_type = mut_info[2] if len(mut_info) > 2 else None
                    except (ValueError, SyntaxError, IndexError) as e:
                        logging.warning(f"Could not parse mut_info for {str_id}: {e}")
            print(f"Using mutation region with label '{mut_type}': {mut_start}-{mut_end}")
        else:
            # 使用所有区域的范围（默认行为）
            mut_start = min(region[0] for region in parsed_regions)
            mut_end = max(region[1] for region in parsed_regions)
            print(f"Using full range for mutation region: {mut_start}-{mut_end}")
        
        # 准备可视化参数
        vis_kwargs = {
            'base_df_path': file_paths['base_df'],
            'read_df_path': file_paths['read_df'],
            'reference_df_path': file_paths['reference_df'],
            'output_file': file_paths['plot'],
            'mut_start': mut_start,
            'mut_end': mut_end,
            'mut_type': mut_type,
            'r_script_path': r_script_path,
            'start_axis': start_axis
        }
                
        run_r_visualization(**vis_kwargs)

        # 将pdf转换为png
        pdf_path = file_paths['plot']
        png_path = pdf_path.replace('.pdf', '.png')
        convert_pdf_to_png(pdf_path, png_path)
    
    return {
        'file_paths': file_paths,
        'sampler_info': sampler_info
    }

## section for process str reads

def process_str_reads(bam_path: str, str_id: str, reference_fasta: str) -> Dict[str, StrReads]:
    """Process STR reads for a single STR site
    
    Args:
        bam_path: Path to BAM/CRAM file
        str_id: Single STR ID to process
        reference_fasta: Path to reference genome
        
    Returns:
        Dictionary containing processed STR reads for the given ID
    """
    str_reads_dict = {}
    
    # 直接处理单个str_id
    result = process_single_str_id(str_id, bam_path=bam_path, reference_fasta=reference_fasta)
    
    # 处理结果
    if result is not None:
        str_id, str_reads = result
        if str_reads is not None:
            str_reads_dict[str_id] = str_reads
    
    return str_reads_dict

def process_str_segment(
    bam_path: str,
    mode: str,
    ref_filename: str,
    single_region: Dict,
    flanking_seqs: Tuple[str, str]
) -> StrReads:
    """Process a single STR region"""
    left_flanking_three_bp, right_flanking_three_bp = flanking_seqs
    
    with pysam.AlignmentFile(bam_path, mode, reference_filename=ref_filename) as alignment_file:
        str_reads = StrReads()
        str_reads.compute_str_data(
            alignment_file,
            single_region['chrom'],
            single_region['start'],
            single_region['end']-1,
            single_region['str_id'],
            single_region['str_unit'],
            left_flanking_three_bp,
            right_flanking_three_bp
        )
        return str_reads

def process_single_str_id(str_id: str, bam_path: str, reference_fasta: str, str_region: STRRegion) -> Tuple[str, Optional[StrReads]]:
    """Process a single STR ID"""
    start_time = time.time()
    logging.info(f"Processing site: {str_id}")
    
    try:
        single_region = str_region
        
        flanking_seqs = get_flanking_sequences(reference_fasta, single_region)
        file_type = determine_file_type(bam_path)
        
        mode = get_file_mode(file_type)
        str_reads = process_str_segment(
            bam_path, mode, reference_fasta, single_region, flanking_seqs
        )
        
        elapsed_time = time.time() - start_time
        logging.info(f"Completed processing {str_id} in {elapsed_time:.2f}s")
        logging.info(f"Memory usage: {get_memory_usage()}")
        
        return str_id, str_reads
        
    except Exception as e:
        logging.error(f"Error processing {str_id}: {str(e)}")
        logging.error(traceback.format_exc())
        return str_id, None

