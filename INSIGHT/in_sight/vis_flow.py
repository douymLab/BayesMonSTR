import os
from in_sight.reads_manager import create_combined_read_manager
from in_sight.utils import parse_region, convert_pdf_to_png
from in_sight.samplers import MutationBasedSampler
from in_sight.add_attr_info import hsnp_processor
from importlib import resources
import in_sight
import os
import subprocess
from typing import List, Tuple, Union, Dict

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

def base_visualization(bam_path: str,
                    region: Union[str, List[str], List[Tuple[str, str]]],
                    reference_fasta: str = None,
                    ref_base: str = None,
                    r_script_path: str = resources.files(in_sight).joinpath('r_script/plot_base.R'),
                    output_dir: str = "./",
                    prefix: str = "pileup_analysis",
                    run_visualization: bool = True,
                    min_base_quality: int = 10,
                    target_count: int = 500,
                    filter_tags: dict = {},
                    mutation_region_label: str = None,  # 指定哪个区域标签用于突变范围
                    hsnp_region_label: str = None,     # 指定哪个区域应用hsnp处理器
                    hsnp_info: Dict[str, Dict[int, Tuple[str, str]]] = None,  # 添加hsnp信息参数
                    align_samples: bool = False,
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
        ref_base: Reference base (optional)
        r_script_path: Path to R script for visualization
        output_dir: Output directory (default current directory)
        prefix: Output file prefix (default 'pileup_analysis')
        run_visualization: Whether to execute R visualization (default True)
        min_base_quality: Minimum base quality for filtering (default 10)
        target_count: Target read count for sampling (default 500)
        filter_tags: Additional filter tags for read manager
        mutation_region_label: Label of the region to use for mutation start/end coordinates (default None, which uses all regions)
        hsnp_region_label: Label of the region to apply hsnp processor (default None)
        hsnp_info: HSNP information dictionary for the hsnp processor (required if hsnp_region_label is specified)
        align_samples: Whether to align samples (default False)
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
    
    # 准备 sampler 参数 - 使用第一个区域进行采样（如果提供了ref_base）
    sampler_kwargs = {
        'target_count': target_count,
        'chrom': chrom,
        'pos': parsed_regions[0][0]-1,  # 使用第一个区域的位置进行采样
        'min_base_quality': min_base_quality
    }
    
    # debug 
    print(f"sampler_kwargs: {sampler_kwargs}")
    
    # 仅当提供了 ref_base 时才添加该参数
    if ref_base is not None:
        sampler_kwargs['ref_base'] = ref_base
        
    print(f"sampler_kwargs: {sampler_kwargs}")
        
    # 从 kwargs 中提取可能的 sampler 参数
    for key in list(kwargs.keys()):
        if key in ['target_count', 'min_base_quality']:
            sampler_kwargs[key] = kwargs.pop(key)
            
    if ref_base is not None or 'ref_base' in kwargs:  # 修改判断条件
        sampler = MutationBasedSampler(**sampler_kwargs)
        print(f"sampler: {sampler}")
    else:
        sampler = None
        print(f"disable sampler")
        
    # 从 kwargs 中提取其他可能的过滤标签
    if 'filter_tags' in kwargs:
        # 合并现有的 filter_tags 与 kwargs 中的 filter_tags
        kwargs_filter_tags = kwargs.pop('filter_tags')
        if filter_tags:
            filter_tags.update(kwargs_filter_tags)
        else:
            filter_tags = kwargs_filter_tags
            
    # 增加调试信息
    print(f"Using filter_tags: {filter_tags}")
    print(f"Debug: filter_tags being sent to manager: {filter_tags}")
    
    # 创建分析管理器，使用解析后的多个区域
    manager = create_combined_read_manager(
        bam_path,
        chrom,
        parsed_regions,  # 现在传递多个区域及其自定义标签
        reference_fasta,
        filter_tags=filter_tags,
        sampler=sampler,
        **kwargs  # 传递剩余的参数给 read_manager
    )
    
    # 处理数据
    manager.clear_processors()
    
    # 如果指定了hsnp_region_label，为对应区域添加hsnp处理器
    if hsnp_region_label is not None and hsnp_info is not None:
        # 找到指定标签的区域
        hsnp_regions = [r for r in parsed_regions if r[2] == hsnp_region_label]
        if hsnp_regions:
            hsnp_start, hsnp_end, _ = hsnp_regions[0]
            print(f"Adding HSNP processor for region '{hsnp_region_label}' ({hsnp_start}-{hsnp_end})")
            # 为该区域添加hsnp处理器
            manager.add_processor(hsnp_processor(hsnp_info, (hsnp_start, hsnp_end)))
    
    manager.process_reads()
    manager.sort_reads('start')
    print(f"align_samples: {align_samples}")
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
        # 根据参数决定使用哪个区域的坐标
        if mutation_region_label is not None:
            # 查找指定标签的区域
            target_regions = [r for r in parsed_regions if r[2] == mutation_region_label]
            if not target_regions:
                raise ValueError(f"Region with label '{mutation_region_label}' not found")
            # 使用指定标签的区域坐标
            mut_start = target_regions[0][0]
            mut_end = target_regions[0][1]
            print(f"Using mutation region with label '{mutation_region_label}': {mut_start}-{mut_end}")
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
            'mut_type': "SNP",
            'r_script_path': r_script_path,
            'start_axis': 1
        }
        
        # 从 kwargs 中提取可能的可视化参数
        for key in ['mut_type', 'start_axis']:
            if key in kwargs:
                vis_kwargs[key] = kwargs[key]
                
        run_r_visualization(**vis_kwargs)

        # 将pdf转换为png
        pdf_path = file_paths['plot']
        png_path = pdf_path.replace('.pdf', '.png')
        convert_pdf_to_png(pdf_path, png_path)
    
    return {
        'file_paths': file_paths,
        'sampler_info': sampler_info
    }
