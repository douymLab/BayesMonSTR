#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CF2Insights - A tool for analysis and visualization of mutation data

This script extracts and visualizes mutation information from BAM files,
and generates region-organized HTML reports for easy viewing and sharing of results.
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import concurrent.futures
import traceback
import base64
import jinja2
import logging
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any
import time
import re
# 添加tqdm库用于显示进度条
from tqdm import tqdm

# Import in_sight modules
from ..vis_flow import base_visualization

# Default configuration file path
DEFAULT_CONFIG_PATH = 'cf2insights_config.yaml'

# Setup logging
def setup_logging(log_file=None):
    """Set up logging configuration"""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # 只有在指定日志文件时才添加文件处理器
    if log_file:
        handlers.append(logging.FileHandler(log_file))
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger("CF2Insights")

logger = setup_logging()

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze and visualize mutation data')
    
    # Configuration file option
    parser.add_argument('--config', '-c', default=DEFAULT_CONFIG_PATH,
                        help=f'Path to configuration file (default: {DEFAULT_CONFIG_PATH})')
    
    # 添加日志文件选项
    parser.add_argument('--log_file',
                        help='Path to log file (if not specified, logs will only be printed to console)')
    
    # Allow overriding config file with command line options
    parser.add_argument('--mutation_file', '-m',
                        help='Mutation information file (TSV format, cells as rows, mutations as columns)')
    parser.add_argument('--bam_path', '-b',
                        help='BAM file path')
    parser.add_argument('--reference_fasta', '-r',
                        help='Reference genome FASTA file path')
    parser.add_argument('--r_script',
                        help='R script path for plotting')
    parser.add_argument('--template_file',
                        help='HTML report template file path')
    parser.add_argument('--output_dir', '-o',
                        help='Base output directory path')
    parser.add_argument('--min_base_quality', type=int,
                        help='Minimum base quality value')
    parser.add_argument('--target_count', type=int,
                        help='Target read count')
    parser.add_argument('--max_workers', type=int,
                        help='Maximum number of worker threads for parallel processing')
    parser.add_argument('--align_samples', action='store_true',
                        help='Align samples')
    parser.add_argument('--mutation_parser', 
                        help='Rule for parsing mutation IDs (default, suffix, or regex:pattern)')
    
    # 添加新参数用于选择特定的mutation ID
    parser.add_argument('--selected_mutations', '-s', nargs='+',
                        help='Only process specified mutation IDs (space-separated list)')
    
    # 添加新参数用于高亮矩阵
    parser.add_argument('--all_pileup_file', '-l',
                        help='Optional all pileup sites for visualization (same format as mutation_file)')
    
    # 添加新参数用于伪批量处理模式
    parser.add_argument('--pseudo_bulk', action='store_true',
                        help='Enable pseudo-bulk mode: process all reads per mutation site instead of per cell.')
    
    return parser.parse_args()

def check_bam_index(bam_path: str):
    """Check if the BAM file has an index file"""
    if not os.path.exists(bam_path + '.bai'):
        logger.error(f"BAM file {bam_path} does not have an index file")
        sys.exit(1)

def get_config():
    """Get configuration from file and command line arguments"""
    args = parse_args()
    
    # Load configuration from file
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        # If no config file specified or doesn't exist, create empty config
        logger.warning(f"Configuration file {args.config} not found")
        config = {}
    
    # Override config with any command line arguments
    for key, value in vars(args).items():
        # Skip 'config' key and None values
        if key == 'config' or value is None:
            continue
        # For boolean flags, only override if explicitly provided on command line
        if key == 'align_samples' and not sys.argv.count('--align_samples'):
            continue
        # Handle pseudo_bulk flag similarly
        if key == 'pseudo_bulk' and not sys.argv.count('--pseudo_bulk'):
            continue
        config[key] = value
            
    # Check for required parameters
    required_params = ['mutation_file', 'bam_path', 'reference_fasta', 'r_script']
    missing_params = [param for param in required_params if param not in config]
    
    if missing_params:
        logger.error(f"Missing required parameters: {', '.join(missing_params)}")
        logger.error("Please provide them in the configuration file or as command line arguments")
        sys.exit(1)
        
    # Set defaults for optional parameters if not present
    if 'output_dir' not in config:
        config['output_dir'] = './output_base_visualization'
    if 'template_file' not in config:
        # 如果是伪批量模式，且用户未指定模板，则使用默认的批量模板
        if config.get('pseudo_bulk') and 'template_file' not in vars(args):
             config['template_file'] = 'template_bulk_summary.html'
        else:
             config['template_file'] = 'template_simple.html' # 默认单细胞模板
    if 'min_base_quality' not in config:
        config['min_base_quality'] = 10
    if 'target_count' not in config:
        config['target_count'] = 200
    if 'max_workers' not in config:
        config['max_workers'] = 10
    if 'align_samples' not in config:
        config['align_samples'] = True
    if 'mutation_parser' not in config:
        config['mutation_parser'] = 'default'
    # Set default for pseudo_bulk mode
    if 'pseudo_bulk' not in config:
        config['pseudo_bulk'] = False
    
    # 添加高亮颜色默认值
    if 'highlight_color' not in config:
        config['highlight_color'] = 'red'
    if 'highlight_bg_color' not in config:
        config['highlight_bg_color'] = '#fff0f0'
    
    return config

def parse_mutation_file(mut_file: str, parse_rule: str = "default", selected_mutations: List[str] = None) -> Dict[str, Any]:
    """
    Parse mutation file and return structured mutation information
    
    Args:
        mut_file: Path to mutation information file (TSV format)
        parse_rule: Rule for parsing mutation IDs
        selected_mutations: List of mutation IDs to process (if None, process all)
        
    Returns:
        Dict containing cell IDs and mutation information
        Format: {
            'cb_labels': [cell_id1, cell_id2,...],
            'mutations': {
                'chr22:22895504-22895504': {
                    'chrom': 'chr22',
                    'pos': 22895504,
                    'ref': 'C',
                    'alt': 'A',
                    'positive_cells': ['CAAGGCCAGACGACGT-1', ...],
                    'original_id': 'chr22_22895504_C_A' # Store the original ID
                },
                ...
            }
        }
    """
    logger.info(f"Parsing mutation file: {mut_file} with rule: {parse_rule}")
    # Read TSV file
    df = pd.read_csv(mut_file, sep='\t', index_col=0)
    
    result = {
        'cb_labels': df.index.tolist(),
        'mutations': {}
    }
    
    # 过滤mutation IDs如果指定了选择列表
    mut_ids = df.columns
    if selected_mutations:
        mut_ids = [mid for mid in df.columns if mid in selected_mutations]
        logger.info(f"Filtering to {len(mut_ids)} mutation IDs out of {len(df.columns)} total")
    
    # Parse each mutation site
    for mut_id in mut_ids:
        original_mut_id = mut_id # Store the original ID before parsing
        # Apply the specified parsing rule
        if parse_rule == "default":
            # Parse chromosome_position_ref_alt format
            parts = mut_id.split('_')
            if len(parts) != 4:
                logger.warning(f"Skipping invalid mutation ID format (default): {mut_id}")
                continue # Skip this mutation ID
            
            chrom, pos, ref, alt = parts
            
        elif parse_rule == "suffix":
            # Parse any_prefix_chromosome_position_ref_alt format
            # For example: P4_cSCC_chr1_1234_A_G -> last 4 parts are what we need
            parts = mut_id.split('_')
            if len(parts) < 4:
                logger.warning(f"Skipping invalid mutation ID format (suffix): {mut_id}")
                continue # Skip this mutation ID
            
            # Extract the last 4 parts
            chrom, pos, ref, alt = parts[-4], parts[-3], parts[-2], parts[-1]
            
            # Verify the chromosome format (should start with 'chr')
            if not chrom.startswith('chr'):
                logger.warning(f"Skipping invalid chromosome format in mutation ID: {mut_id}")
                continue # Skip this mutation ID
            
        elif parse_rule.startswith("regex:"):
            # Use custom regex pattern
            pattern = parse_rule[6:]  # Remove 'regex:' prefix
            match = re.match(pattern, mut_id)
            if not match:
                logger.warning(f"Mutation ID {mut_id} doesn't match pattern {pattern}, skipping.")
                continue # Skip this mutation ID
            
            try:
                chrom = match.group('chrom')
                pos = match.group('pos')
                ref = match.group('ref')
                alt = match.group('alt')
            except IndexError:
                 logger.warning(f"Regex pattern must contain named groups: chrom, pos, ref, alt. Skipping {mut_id}")
                 continue # Skip this mutation ID
            
        else:
            logger.error(f"Unsupported parse_rule: {parse_rule}")
            sys.exit(1) # Exit if rule is unsupported
        
        # Validate position
        try:
            pos_int = int(pos)
        except ValueError:
            logger.warning(f"Invalid position '{pos}' in mutation ID: {mut_id}, skipping.")
            continue
            
        region = f"{chrom}:{pos_int}-{pos_int}"  # Single base position
        
        # Record mutation information
        result['mutations'][region] = {
            'chrom': chrom,
            'pos': pos_int,
            'ref': ref,
            'alt': alt,
            'positive_cells': df.index[df[mut_id] == 1].tolist(),
            'original_id': original_mut_id # Store the original ID
        }
    
    logger.info(f"Successfully parsed {len(result['mutations'])} mutation sites.")
    return result

# # 默认规则处理 chr1_39034563_T_A 格式
# result = parse_mutation_file("mutations.tsv")

# # 处理前缀格式如 P4_cSCC_chr1_1234_A_G
# result = parse_mutation_file("mutations.tsv", parse_rule="suffix")

# # 使用自定义正则表达式处理复杂格式
# pattern = r"(?P<sample>\w+)_(?P<type>\w+)_(?P<chrom>chr\w+)_(?P<pos>\d+)_(?P<ref>[ACGTN])_(?P<alt>[ACGTN])"
# result = parse_mutation_file("mutations.tsv", parse_rule=f"regex:{pattern}")

def generate_report_by_region(region_data: Dict, images_dir: str, reports_dir: str, 
                             region_id: str, template_file: str, config: Dict = None) -> None:
    """
    Generate an independent HTML report for each region using structured directory images (Single-cell mode)
    
    Args:
        region_data: Dictionary containing region processing results for specific cells
        images_dir: Directory path for this region's images
        reports_dir: Report output directory
        region_id: Region's unique identifier (chrom_pos_ref_alt)
        template_file: HTML template file path (e.g., template_simple.html)
        config: Configuration dictionary
    """
    # Get basic region information from the first item
    if not region_data['items']:
        logger.warning(f"No items found for region {region_id}, skipping report generation.")
        return
        
    first_item = region_data['items'][0]
    
    # 设置默认高亮颜色
    highlight_color = 'red'
    highlight_bg_color = '#fff0f0'
    
    # 如果有配置，使用配置中的颜色
    if config:
        highlight_color = config.get('highlight_color', highlight_color)
        highlight_bg_color = config.get('highlight_bg_color', highlight_bg_color)
    
    # Prepare template data
    template_data = {
        'region_id': region_id,
        'chromosome': first_item['chrom'],
        'position': first_item['pos'],
        'reference': first_item['ref'],
        'alternate': first_item['alt'],
        'samples': [],
        'highlight_color': highlight_color,
        'highlight_bg_color': highlight_bg_color,
        'original_id': first_item.get('original_id', region_id) # Use original ID if available
    }
    
    # Add data for each cell associated with this region
    for item in region_data['items']:
        # Construct path to the image for this specific cell and region
        # Assumes images are stored like: images/by_region/region_id/cell_barcode.png
        plot_path = os.path.join(images_dir, f"{item['cb_label']}.png")
        
        # Read image and convert to base64
        image_base64 = ''
        plot_exists = os.path.exists(plot_path)
        if plot_exists:
            try:
                with open(plot_path, 'rb') as img_file:
                    image_bytes = img_file.read()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            except Exception as e:
                 logger.error(f"Error reading image {plot_path} for cell {item['cb_label']}: {e}")
                 plot_exists = False # Mark as not existing if reading failed

        # sampler_info should come from the results dictionary passed during report generation
        sampler_info = item.get('sampler_info', {'total': 0, 'sampled': 0})
        
        sample_data = {
            'cell_barcode': item['cb_label'],
            'image_base64': image_base64,
            'is_empty': not plot_exists,
            'note_message': 'Plot not generated or unreadable' if not plot_exists else '',
            'sampler_info': sampler_info,  # 添加采样器信息
            'highlight': item.get('highlight', False)  # 添加高亮标记
        }
        template_data['samples'].append(sample_data)
    
    # Generate report
    try:
        # Check if template file exists
        if not os.path.exists(template_file):
            logger.error(f"Template file not found: {template_file}")
            # Maybe use a default fallback template? For now, just error out.
            return 

        with open(template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        template = jinja2.Template(template_content)
        rendered_content = template.render(**template_data)
        
        # Write HTML file to reports directory
        # Ensure region_id is filesystem-safe (replace ':' with '_')
        safe_region_id = region_id.replace(':', '_')
        output_html = os.path.join(reports_dir, f"{safe_region_id}_report.html")
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(rendered_content)
            
        logger.info(f"Generated HTML report for region {region_id}")
        
    except jinja2.exceptions.TemplateNotFound:
         logger.error(f"Template file not found during rendering: {template_file}")
    except Exception as e:
        logger.error(f"Error generating report for region {region_id}: {str(e)}")
        logger.debug(traceback.format_exc())

def generate_bulk_summary_report(mutation_results: Dict, images_base_dir: str, reports_dir: str,
                                template_file: str, config: Dict = None) -> None:
    """
    Generate a single summary HTML report for all mutation sites in pseudo-bulk mode.

    Args:
        mutation_results: Dictionary containing results for each mutation site. 
                          Keys are mutation_ids (chrom_pos_ref_alt), values contain info like plot path, sampler_info.
        images_base_dir: Base directory where bulk images are stored (e.g., output/images/bulk_plots).
        reports_dir: Directory where the summary report will be saved.
        template_file: Path to the HTML template for the bulk summary (e.g., template_bulk_summary.html).
        config: Configuration dictionary.
    """
    logger.info("Generating pseudo-bulk summary report...")
    
    template_data = {
        'mutations': [],
        'report_title': f"Pseudo-Bulk Mutation Visualization Summary ({config.get('mutation_file', 'N/A')})"
    }

    # Iterate through the results collected from the bulk processing
    for mutation_id, result_info in mutation_results.items():
        # Construct the expected plot path based on mutation_id
        # Assumes images are stored like: images/bulk_plots/chrom_pos_ref_alt.png
        plot_filename = f"{mutation_id}.png"
        plot_path = os.path.join(images_base_dir, plot_filename)

        image_base64 = ''
        plot_exists = os.path.exists(plot_path)
        if plot_exists:
            try:
                with open(plot_path, 'rb') as img_file:
                    image_bytes = img_file.read()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            except Exception as e:
                logger.error(f"Error reading bulk image {plot_path} for mutation {mutation_id}: {e}")
                plot_exists = False

        # Extract details from mutation_id (assuming chrom_pos_ref_alt format)
        parts = mutation_id.split('_')
        chrom, pos, ref, alt = "Unknown", "Unknown", "Unknown", "Unknown"
        original_id = result_info.get('original_id', mutation_id) # Get original ID if stored
        if len(parts) == 4:
            chrom, pos, ref, alt = parts[0], parts[1], parts[2], parts[3]
        else:
             logger.warning(f"Could not parse mutation details from ID: {mutation_id}")


        mutation_data = {
            'mutation_id': mutation_id,
            'original_id': original_id, # Add original ID to report
            'chromosome': chrom,
            'position': pos,
            'reference': ref,
            'alternate': alt,
            'image_base64': image_base64,
            'is_empty': not plot_exists,
            'note_message': 'Plot not generated or unreadable' if not plot_exists else '',
            'sampler_info': result_info.get('sampler_info', {'total': 0, 'sampled': 0}) # Get sampler info if available
        }
        template_data['mutations'].append(mutation_data)
        
    # Sort mutations for consistent report order (optional, e.g., by chromosome then position)
    try:
        template_data['mutations'].sort(key=lambda m: (m['chromosome'], int(m['position']) if m['position'].isdigit() else float('inf')))
    except ValueError:
         logger.warning("Could not sort mutations numerically by position.")
         template_data['mutations'].sort(key=lambda m: (m['chromosome'], m['position']))


    # Generate the summary report
    try:
        if not os.path.exists(template_file):
            logger.error(f"Bulk summary template file not found: {template_file}. Cannot generate report.")
             # Create a dummy template file if it doesn't exist? Or exit?
            # For now, just return.
            return 

        with open(template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        template = jinja2.Template(template_content)
        rendered_content = template.render(**template_data)
        
        # Write the single HTML file to the main reports directory
        output_html = os.path.join(reports_dir, "bulk_summary_report.html")
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(rendered_content)
            
        logger.info(f"Generated bulk summary report: {output_html}")
        
    except jinja2.exceptions.TemplateNotFound:
         logger.error(f"Bulk summary template file not found during rendering: {template_file}")
    except Exception as e:
        logger.error(f"Error generating bulk summary report: {str(e)}")
        logger.debug(traceback.format_exc())


def process_visualization_tasks(config, processing_list, unique_mutations_map):
    """
    Process all visualization tasks in parallel, adapting for pseudo-bulk mode.
    
    Args:
        config: Configuration dictionary
        processing_list: List of tasks to process (per cell_barcode in normal mode)
        unique_mutations_map: Dictionary mapping region_id to mutation info including original_id
    """
    is_pseudo_bulk = config.get('pseudo_bulk', False)
    
    # --- Determine Output Directories based on mode ---
    if is_pseudo_bulk:
        reports_dir = os.path.join(config['output_dir'], 'reports') # Main reports dir for summary
        images_base_dir = os.path.join(config['output_dir'], 'images', 'bulk_plots')
        logger.info("Running in Pseudo-Bulk mode. Images will be stored in: " + images_base_dir)
    else:
        # Original single-cell structure
        reports_dir = os.path.join(config['output_dir'], 'reports', 'region_reports')
        images_base_dir = os.path.join(config['output_dir'], 'images', 'by_region')
        logger.info("Running in Single-Cell mode. Images will be stored in: " + images_base_dir)
        
    # Create necessary directories
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(images_base_dir, exist_ok=True)
    
    # --- Prepare Tasks based on mode ---
    all_tasks = []
    task_identifiers = set() # Keep track of tasks to avoid duplicates (esp. in bulk mode)

    if is_pseudo_bulk:
        logger.info("Preparing tasks for pseudo-bulk processing...")
        # Tasks are per unique mutation site
        for region_id, mutation_info in unique_mutations_map.items():
            # Use region_id (chrom_pos_ref_alt) as the unique identifier for the task
            mutation_task_id = f"{mutation_info['chrom']}_{mutation_info['pos']}_{mutation_info['ref']}_{mutation_info['alt']}"
            
            if mutation_task_id in task_identifiers:
                continue # Should not happen if unique_mutations_map is correct, but good practice
            
            # Define output path for the bulk plot
            output_filename = f"{mutation_task_id}.png"
            expected_output = os.path.join(images_base_dir, output_filename)
            
            # Prepare task if output doesn't exist
            if not os.path.exists(expected_output):
                task_args = {
                    'bam_path': config['bam_path'],
                    'region': f"{mutation_info['chrom']}:{mutation_info['pos']}-{mutation_info['pos']}", # Samtools region format
                    'reference_fasta': config['reference_fasta'],
                    'ref_base': mutation_info['ref'],
                    'r_script_path': config['r_script'],
                    'output_dir': images_base_dir, # Output directly to bulk dir
                    'prefix': mutation_task_id, # Prefix is the mutation ID
                    'run_visualization': True,
                    'min_base_quality': config['min_base_quality'],
                    'target_count': config['target_count'],
                    'filter_tags': None, # Process all reads for the region
                    'align_samples': config['align_samples'] # Align samples might still be relevant for visualization layout
                }
                # Store mutation info along with task args for later use (e.g., report generation)
                all_tasks.append((task_args, mutation_info)) 
                task_identifiers.add(mutation_task_id)
                logger.info(f"Prepared pseudo-bulk task for: {mutation_task_id} (Original ID: {mutation_info.get('original_id', 'N/A')})")
            else:
                logger.info(f"Skipped existing pseudo-bulk visualization: {mutation_task_id}")
                # Still need to collect info for the report even if skipped
                all_tasks.append((None, mutation_info)) # Mark task_args as None to indicate skipping

    else: # Single-cell mode (original logic)
        logger.info("Preparing tasks for single-cell processing...")
        # Pre-create region directories to avoid conflicts during concurrent creation
        all_regions = set()
        for item in processing_list:
            # region_id format: chrom_pos_ref_alt
            region_id = f"{item['chrom']}_{item['pos']}_{item['ref']}_{item['alt']}"
            all_regions.add(region_id)
        
        for region_id in all_regions:
            region_images_dir = os.path.join(images_base_dir, region_id)
            os.makedirs(region_images_dir, exist_ok=True)
            
        # Analyze all processing tasks for parallel execution
        for item in processing_list:
             # region_id format: chrom_pos_ref_alt
            region_id = f"{item['chrom']}_{item['pos']}_{item['ref']}_{item['alt']}"
            region_images_dir = os.path.join(images_base_dir, region_id)
            
            # Expected output file path for this cell
            output_filename = f"{item['cb_label']}.png"
            expected_output = os.path.join(region_images_dir, output_filename)
            
            # If file doesn't exist, add to task list
            task_identifier = f"{region_id}_{item['cb_label']}" # Unique identifier for cell-specific task
            if task_identifier not in task_identifiers:
                if not os.path.exists(expected_output):
                    task_args = {
                        'bam_path': config['bam_path'],
                        'region': item['region'], # Region format like chr:start-end
                        'reference_fasta': config['reference_fasta'],
                        'ref_base': item['ref'],
                        'r_script_path': config['r_script'],
                        'output_dir': region_images_dir, # Output to region-specific subdir
                        'prefix': item['cb_label'], # Prefix is cell barcode
                        'run_visualization': True,
                        'min_base_quality': config['min_base_quality'],
                        'target_count': config['target_count'],
                        'filter_tags': {'CB': item['cb_label']}, # Filter by cell barcode
                        'align_samples': config['align_samples']
                    }
                    # Store the original item data
                    all_tasks.append((task_args, item)) 
                    task_identifiers.add(task_identifier)
                    logger.info(f"Prepared task: {item['cb_label']} at {item['region']}")
                else:
                    logger.info(f"Skipped existing visualization: {item['cb_label']} at {item['region']}")
                    # Still include in results for report generation
                    all_tasks.append((None, item)) 
                    task_identifiers.add(task_identifier) # Add skipped task identifier too


    # --- Execute Tasks in Parallel ---
    tasks_to_run = [(args, info) for args, info in all_tasks if args is not None]
    skipped_tasks_info = [info for args, info in all_tasks if args is None]

    logger.info(f"Total tasks identified: {len(all_tasks)}")
    logger.info(f"Tasks to execute: {len(tasks_to_run)}")
    logger.info(f"Tasks skipped (already exist): {len(skipped_tasks_info)}")

    if not tasks_to_run:
        logger.info("No new visualization tasks to execute.")
        # Proceed directly to report generation with existing/skipped data
        if is_pseudo_bulk:
             # Need to collect results for skipped tasks too
             bulk_results = {f"{info['chrom']}_{info['pos']}_{info['ref']}_{info['alt']}": {'original_id': info.get('original_id', 'N/A'), 'sampler_info': {}} for info in skipped_tasks_info}
             generate_bulk_summary_report(bulk_results, images_base_dir, reports_dir, config['template_file'], config)
        else:
             # Pass skipped items to generate_all_reports
             generate_all_reports(skipped_tasks_info, images_base_dir, reports_dir, config['template_file'], {}, config)
        return # Exit processing function early

    # Define total tasks for progress bar
    total_tasks_for_progress = len(tasks_to_run)
    
    # Create process pool and submit tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=config['max_workers']) as executor:
        # Create a dictionary to map futures to their task info (item or mutation_info)
        future_to_task_info = {}
        
        # Submit all tasks that need running
        for task_args, task_info in tasks_to_run:
            identifier = f"{task_info['chrom']}_{task_info['pos']}_{task_info['ref']}_{task_info['alt']}"
            if not is_pseudo_bulk:
                identifier += f"_{task_info['cb_label']}" # Add cell barcode for single-cell mode
                
            logger.info(f"Submitting task: {identifier} at {time.strftime('%H:%M:%S')}")
            future = executor.submit(base_visualization, **task_args)
            future_to_task_info[future] = task_info # Store the corresponding info dict
        
        # Process results as they complete
        results = {} # Store results, structure depends on mode

        # Create progress bar
        with tqdm(total=total_tasks_for_progress, desc="Processing visualizations", unit="task") as progress_bar:
            for future in concurrent.futures.as_completed(future_to_task_info):
                task_info = future_to_task_info[future]
                
                # Determine the identifier based on mode
                if is_pseudo_bulk:
                    result_key = f"{task_info['chrom']}_{task_info['pos']}_{task_info['ref']}_{task_info['alt']}"
                    log_identifier = result_key + f" (Original: {task_info.get('original_id', 'N/A')})"
                else:
                    result_key_region = f"{task_info['chrom']}_{task_info['pos']}_{task_info['ref']}_{task_info['alt']}"
                    result_key_cell = task_info['cb_label']
                    log_identifier = f"{result_key_cell} at {task_info['region']}"

                try:
                    # Get the result from the completed future
                    task_result = future.result() 
                    
                    # Store the result (primarily sampler_info)
                    sampler_info = task_result.get('sampler_info', {'total': 0, 'sampled': 0})
                    
                    if is_pseudo_bulk:
                        # Store result per mutation_id
                        results[result_key] = {
                             'sampler_info': sampler_info,
                             'original_id': task_info.get('original_id', 'N/A') # Carry over original ID
                        }
                    else:
                        # Store result per region -> cell
                        if result_key_region not in results:
                            results[result_key_region] = {}
                        results[result_key_region][result_key_cell] = {
                            'sampler_info': sampler_info
                        }
                        
                    logger.info(f"Completed task: {log_identifier} at {time.strftime('%H:%M:%S')}")
                    
                except Exception as e:
                    logger.error(f"Visualization task error for {log_identifier}: {e}")
                    logger.debug(traceback.format_exc())
                    # Store error indication? For now, just log.
                    if is_pseudo_bulk:
                         results[result_key] = {'error': str(e), 'sampler_info': {}, 'original_id': task_info.get('original_id', 'N/A')}
                    else:
                        if result_key_region not in results:
                            results[result_key_region] = {}
                        results[result_key_region][result_key_cell] = {'error': str(e), 'sampler_info': {}}

                # Update progress bar
                progress_bar.update(1)
        
    # --- Generate Reports ---
    if is_pseudo_bulk:
         # Add info for skipped tasks to the results before generating report
         for info in skipped_tasks_info:
              mutation_id = f"{info['chrom']}_{info['pos']}_{info['ref']}_{info['alt']}"
              if mutation_id not in results: # Only add if not already processed (e.g., due to error)
                    results[mutation_id] = {
                         'sampler_info': {}, # No sampler info as it was skipped
                         'original_id': info.get('original_id', 'N/A'),
                         'skipped': True # Indicate it was skipped
                    }
         generate_bulk_summary_report(results, images_base_dir, reports_dir, config['template_file'], config)
    else:
        # Combine executed task results with skipped task info for single-cell reports
        full_processing_list_for_report = processing_list # Use the original list that contains all cell-mutation pairs
        generate_all_reports(full_processing_list_for_report, images_base_dir, reports_dir, config['template_file'], results, config)


def generate_all_reports(processing_list, images_base_dir, reports_dir, template_file, results=None, config=None):
    """
    Generate reports for all regions (Single-cell mode).
    Assumes config['pseudo_bulk'] is False.
    """
    if config and config.get('pseudo_bulk'):
         logger.error("generate_all_reports called in pseudo_bulk mode. This should not happen.")
         return # Should call generate_bulk_summary_report instead

    # Group by region for individual reports
    region_groups = defaultdict(lambda: {'items': []})
    processed_item_keys = set() # Track items already added to groups

    # First, process items that have results (executed or errored)
    if results:
        for region_id, cell_results in results.items():
            for cb_label, result_data in cell_results.items():
                 # Find the original item in processing_list matching region_id and cb_label
                 # This is inefficient, consider restructuring data earlier if performance is an issue
                 found_item = None
                 for item in processing_list:
                      item_region_id = f"{item['chrom']}_{item['pos']}_{item['ref']}_{item['alt']}"
                      if item_region_id == region_id and item['cb_label'] == cb_label:
                           found_item = item
                           break
                 
                 if found_item:
                      item_copy = found_item.copy()
                      item_copy.update(result_data) # Add sampler_info or error message
                      region_groups[region_id]['items'].append(item_copy)
                      processed_item_keys.add( (region_id, cb_label) ) # Mark as processed
                 else:
                      logger.warning(f"Could not find original item for result: {cb_label} in {region_id}")

    # Add items that were skipped (not in results dict)
    for item in processing_list:
         region_id = f"{item['chrom']}_{item['pos']}_{item['ref']}_{item['alt']}"
         cb_label = item['cb_label']
         if (region_id, cb_label) not in processed_item_keys:
              item_copy = item.copy()
              # Indicate it was skipped or image might exist but wasn't processed this run
              item_copy['sampler_info'] = item_copy.get('sampler_info', {'total': 'N/A', 'sampled': 'N/A (Skipped/Exists)'}) 
              region_groups[region_id]['items'].append(item_copy)


    logger.info(f"Starting to generate single-cell reports for {len(region_groups)} regions")
    
    # Generate report for each region
    for region_id, group_data in region_groups.items():
        # Construct the path to the image directory for this region
        # images_base_dir is like .../images/by_region
        region_images_dir = os.path.join(images_base_dir, region_id) 
        # reports_dir is like .../reports/region_reports
        generate_report_by_region(group_data, region_images_dir, reports_dir, region_id, template_file, config)


def print_statistics(processing_list, config, unique_mutations_map):
    """Print processing data statistics, adapting for mode."""
    is_pseudo_bulk = config.get('pseudo_bulk', False)
    
    if is_pseudo_bulk:
        # Statistics based on unique mutations
        total_tasks = len(unique_mutations_map)
        logger.info("--- Pseudo-Bulk Mode Statistics ---")
        logger.info(f"Total unique mutation sites to process: {total_tasks:,}")
        # Can add more stats here if needed, e.g., distribution by chromosome
        
    else:
        # Original single-cell statistics
        mutation_counter = defaultdict(int)  # Count processing times for each mutation site
        cell_counter = Counter()             # Count processing times for each cell
        unique_mutations_sc = set()             # Unique mutation site set (in single-cell context)
        unique_cells = set()                 # Unique cell set involved

        # Traverse processing list for statistics
        for item in processing_list:
            # Count mutation sites (using chromosome+position+ref+alt as unique identifier)
            mutation_id = f"{item['chrom']}_{item['pos']}_{item['ref']}_{item['alt']}"
            mutation_counter[mutation_id] += 1
            unique_mutations_sc.add(mutation_id)
            
            # Count cells
            cell_counter[item['cb_label']] += 1
            unique_cells.add(item['cb_label'])

        # Output statistics
        logger.info("--- Single-Cell Mode Statistics ---")
        logger.info(f"Total cell-mutation tasks: {len(processing_list):,}")
        logger.info(f"Unique mutation sites involved: {len(unique_mutations_sc):,}")
        logger.info(f"Unique cells involved: {len(unique_cells):,}\n")

        logger.info("Mutation processing frequency (Top 5):")
        # Use the unique_mutations_map to get original IDs for display if possible
        mut_freq_display = []
        for mut_id, count in Counter(mutation_counter).most_common(5):
             display_id = mut_id
             # Find original ID - requires iterating through map, might be slow if many mutations
             for region_key, info in unique_mutations_map.items():
                 if f"{info['chrom']}_{info['pos']}_{info['ref']}_{info['alt']}" == mut_id:
                      display_id = info.get('original_id', mut_id)
                      break
             mut_freq_display.append((display_id, count))

        for disp_id, count in mut_freq_display:
             logger.info(f"· {disp_id}: {count} cells")


        logger.info("\nCell task frequency (Top 5):")
        for cell, count in cell_counter.most_common(5):
            logger.info(f"· {cell}: {count} mutations")
        
        # Save statistics to CSV
        stats_dir = os.path.join(config['output_dir'], 'data', 'stats')
        os.makedirs(stats_dir, exist_ok=True)
        
        # Save mutation statistics (frequency across cells)
        mut_stats = pd.DataFrame(list(mutation_counter.items()), 
                                columns=['mutation_id', 'cell_count'])
        # Add original ID column if possible
        original_id_map = {f"{info['chrom']}_{info['pos']}_{info['ref']}_{info['alt']}": info.get('original_id', '') 
                           for info in unique_mutations_map.values()}
        mut_stats['original_id'] = mut_stats['mutation_id'].map(original_id_map)
        mut_stats = mut_stats[['mutation_id', 'original_id', 'cell_count']] # Reorder
        mut_stats.to_csv(os.path.join(stats_dir, 'mutation_frequency_stats.csv'), index=False)
        
        # Save cell statistics (number of mutations processed per cell)
        cell_stats = pd.DataFrame(list(cell_counter.items()), 
                                 columns=['cell_barcode', 'mutation_count'])
        cell_stats.to_csv(os.path.join(stats_dir, 'cell_task_stats.csv'), index=False)


def main():
    """Main function"""
    # Get configuration
    config = get_config()
    
    # 重新设置日志记录器，使用配置中的log_file参数（如果存在）
    global logger
    logger = setup_logging(config.get('log_file'))
    
    # Check BAM index
    check_bam_index(config['bam_path'])
    
    # debug 
    logger.debug(f"Effective configuration: {config}")
    
    # Create base output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # --- Parse Mutation Files ---
    selected_mutations = config.get('selected_mutations') # List of original IDs
    
    # File defining which mutation/cell pairs to potentially highlight (if not in bulk mode)
    highlight_mutation_data = parse_mutation_file(
        config['mutation_file'], 
        config['mutation_parser'],
        selected_mutations 
    )
    
    # Create highlight set for quick lookup (relevant for single-cell mode report)
    highlight_mutations = set()
    for region, info in highlight_mutation_data['mutations'].items():
        # region is like chr:start-end, info contains chrom, pos, ref, alt
        for cb_label in info['positive_cells']:
            highlight_key = f"{info['chrom']}_{info['pos']}_{info['ref']}_{info['alt']}_{cb_label}"
            highlight_mutations.add(highlight_key)
            
    # Determine the primary file to define *which* mutations/cells to process
    # If all_pileup_file is given, use that. Otherwise, use mutation_file.
    processing_mutation_file = config.get('all_pileup_file', config['mutation_file'])
    logger.info(f"Using mutation data for processing tasks from: {processing_mutation_file}")
    
    # Parse the file that defines the scope of work
    # We pass selected_mutations here again to ensure consistency if all_pileup_file is used
    processing_mutation_data = parse_mutation_file(
        processing_mutation_file, 
        config['mutation_parser'], 
        selected_mutations 
    )

    # Create data directory and save raw input paths/config? (Optional)
    data_dir = os.path.join(config['output_dir'], 'data', 'raw')
    os.makedirs(data_dir, exist_ok=True)
    # Consider saving the config used:
    config_save_path = os.path.join(data_dir, 'run_config.yaml')
    try:
        with open(config_save_path, 'w') as f:
             yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Saved run configuration to {config_save_path}")
    except Exception as e:
         logger.warning(f"Could not save run configuration: {e}")

    # --- Prepare Processing List and Unique Mutations ---
    # `processing_list` is primarily for single-cell mode (list of dicts per cell-mutation)
    # `unique_mutations_map` is useful for both modes, mapping unique chrom_pos_ref_alt to details
    processing_list = []
    unique_mutations_map = {} # Key: chrom_pos_ref_alt, Value: {'chrom': ..., 'pos':..., 'ref':..., 'alt':..., 'original_id':...}

    for region, info in processing_mutation_data['mutations'].items():
        # region here is like chr:start-end from the parser
        mutation_id_key = f"{info['chrom']}_{info['pos']}_{info['ref']}_{info['alt']}"
        
        # Store unique mutation info, including the original ID
        if mutation_id_key not in unique_mutations_map:
             unique_mutations_map[mutation_id_key] = {
                 'chrom': info['chrom'],
                 'pos': info['pos'],
                 'ref': info['ref'],
                 'alt': info['alt'],
                 'original_id': info['original_id'] # Get original ID from parser result
             }

        # Build the per-cell list (only used if not pseudo_bulk, but build anyway for stats)
        for cb_label in info['positive_cells']:
            # Check if this specific cell-mutation pair should be highlighted (using the highlight file data)
            highlight_key = f"{info['chrom']}_{info['pos']}_{info['ref']}_{info['alt']}_{cb_label}"
            is_highlighted = highlight_key in highlight_mutations
            
            processing_list.append({
                'region': region, # chr:start-end format
                'cb_label': cb_label,
                'chrom': info['chrom'],
                'pos': info['pos'],
                'ref': info['ref'],
                'alt': info['alt'],
                'highlight': is_highlighted,  # Add highlight flag (used in single-cell report)
                'original_id': info['original_id'] # Carry original ID
            })
            
    logger.info(f"Identified {len(unique_mutations_map)} unique mutation sites for processing.")
    if not config.get('pseudo_bulk'):
        logger.info(f"Prepared {len(processing_list)} total cell-mutation tasks for single-cell mode.")

    # Print statistics based on the mode
    print_statistics(processing_list, config, unique_mutations_map)
    
    # Process data using the appropriate parallel function
    process_visualization_tasks(config, processing_list, unique_mutations_map)
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    # Only execute when run as a script
    main()

