#!/usr/bin/env python3
# Import necessary modules
import os
import sys
import argparse
import logging
import duckdb
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import datetime
import jinja2
import base64
import ast  # For parsing complex values
import traceback  # For enhanced error reporting

# Import custom modules
from in_sight.str_vis_flow import str_visulization
from in_sight.str_utils import create_str_region_from_duckdb, get_str_hsnp_info

def check_file_exists(file_path: str) -> bool:
    """检查文件是否存在"""
    if not os.path.exists(file_path):
        return False
    return True

def check_output_file(file_path: str, force: bool = False) -> bool:
    """检查输出文件是否存在，如果存在且force为False，询问是否覆盖"""
    if os.path.exists(file_path) and not force:
        response = input(f"文件 {file_path} 已存在，是否覆盖? [y/N] ")
        return response.lower() in ['y', 'yes']
    return True

def detect_file_type(file_path: str) -> str:
    """使用magic number检测文件类型"""
    magic_dict = {
        b'\x1f\x8b\x08': 'gz',
        b'\x42\x5a\x68': 'bz2',
        b'\x50\x4b\x03\x04': 'zip',
        b'\x53\x51\x4c\x69': 'duckdb'  # DuckDB 文件的magic number
    }
    
    try:
        with open(file_path, 'rb') as f:
            file_start = f.read(4)  # 读取前4个字节来检测文件类型
        
        for magic, filetype in magic_dict.items():
            if file_start.startswith(magic):
                return filetype
        return 'plain'
    except (IOError, FileNotFoundError):
        logging.error(f"无法读取文件: {file_path}")
        return 'unknown'

def setup_logging(verbose: bool) -> None:
    """设置日志级别"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def resolve_input_path(input_path: Optional[str], stdin_format: str = 'text') -> str:
    """处理输入路径，支持标准输入"""
    if input_path == '-' or input_path is None:
        # 从标准输入读取
        temp_file = None
        if stdin_format == 'json':
            # 从标准输入读取JSON
            data = json.load(sys.stdin)
            temp_file = Path(os.path.expanduser('~/.cache/in_sight_temp.json'))
            with open(temp_file, 'w') as f:
                json.dump(data, f)
        else:
            # 从标准输入读取文本
            data = sys.stdin.read()
            temp_file = Path(os.path.expanduser('~/.cache/in_sight_temp.txt'))
            with open(temp_file, 'w') as f:
                f.write(data)
        return str(temp_file)
    return input_path

def resolve_output_path(output_path: Optional[str], default_path: str) -> str:
    """Handle output path, supporting standard output"""
    if output_path == '-' or output_path is None:
        return '-'  # Output to standard output
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    return output_path if output_path else default_path

def visualize_str(args: argparse.Namespace) -> None:
    """Main function for the visualize STR subcommand"""
    # Handle input and output paths
    bm_res_database = args.bm_res_database
    output_dir = args.output_dir
    
    # Check if input files exist
    required_files = [bm_res_database, args.bam_path, args.reference_fasta, args.r_script_path]
    for file_path in required_files:
        if not check_file_exists(file_path):
            logging.error(f"File does not exist: {file_path}")
            sys.exit(1)
    
    # Check file type
    db_file_type = detect_file_type(bm_res_database)
    if db_file_type != 'duckdb' and db_file_type != 'plain':
        logging.warning(f"Database file type may not be DuckDB: {db_file_type}")
    
    # Create output directory
    if output_dir != '-':
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if existing files should be overwritten
        output_file_path = os.path.join(output_dir, f"{args.prefix}" if args.prefix else f"{args.str_id}_{args.individual_code}.png")
        if os.path.exists(output_file_path) and not args.force:
            if not check_output_file(output_file_path, args.force):
                logging.info("Operation cancelled by user, not overwriting existing file")
                return
    
    # Display database information
    if args.verbose:
        logging.info(f"Viewing first few rows of database {bm_res_database}")
        con = duckdb.connect(bm_res_database, read_only=True)
        try:
            # Get table names
            tables = con.execute("SHOW TABLES").fetchall()
            if not tables:
                logging.error("No tables in database")
                sys.exit(1)
            
            table_name = tables[0][0]  # Use the first table
            logging.info(f"Using table: {table_name}")
            
            # Get database column information
            columns = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            column_names = [col[1] for col in columns]  # col[1] is the column name
            
            logging.info(f"Table {table_name} has columns: {', '.join(column_names)}")
            
            result = con.execute(f"SELECT * FROM '{table_name}' LIMIT 5").fetchall()
            for row in result:
                row_str = "\t".join(str(val) for val in row)
                logging.debug(row_str)
        except Exception as e:
            logging.error(f"Error querying database: {e}")
            sys.exit(1)
        finally:
            con.close()
    
    # Extract STR region information
    try:
        # Create column name mapping based on actual database structure
        # Assume database column structure is VCF format: chr, pos, id, ref, alt, qual, filter, info, format
        column_mapping = {
            'id': 'str_id',
            'chr': 'chrom',
            'start': 'start',
            'end': 'end',
            'motif': 'str_unit',
            'motif_len': 'str_unit_length',
            'len': 'average_length',
        }
        
        # Call create_str_region_from_duckdb with custom column mapping
        str_region_info = create_str_region_from_duckdb(
            bm_res_database, 
            args.str_id,
            column_mapping=column_mapping
        )
        
        if str_region_info is None:
            logging.error(f"Cannot find STR ID: {args.str_id}")
            sys.exit(1)
        
    except Exception as e:
        logging.error(f"Error getting STR region information: {e}")
        sys.exit(1)
    
    # Get SNP information
    try:
        SNP_info = get_str_hsnp_info(str_region_info, args.individual_code)
    except Exception as e:
        logging.error(f"Error getting SNP information: {e}")
        SNP_info = None
    
    # Prepare region information
    hsnp_region_label = None
    mutation_region_label = None
    str_id = None
    
    if str_region_info is not None:
        if SNP_info is not None:
            region_list = [(str_region_info.to_region_str(), 'STR'), (SNP_info.to_region_str(), 'hSNP')]
            mutation_region_label = 'STR'
            str_id = str_region_info['str_id']
            hsnp_region_label = 'hSNP'
        else:
            region_list = [(str_region_info.to_region_str(), 'STR')]
            mutation_region_label = 'STR'
            str_id = str_region_info['str_id']
    
    # Perform visualization
    prefix = f"{args.prefix}" if args.prefix else f"{args.str_id}_{args.individual_code}"
    
    try:
        str_visulization(
            bam_path=args.bam_path,
            region=region_list,
            reference_fasta=args.reference_fasta,
            r_script_path=args.r_script_path,
            str_id=str_id,
            str_region_info=str_region_info,
            mutation_region_label=mutation_region_label,
            hsnp_region_label=hsnp_region_label,
            hsnp_info=SNP_info,
            output_dir=output_dir,
            prefix=prefix,
            run_visualization=True,
            align_samples=True,
            start_axis=0,
            pro_type=args.pro_type,
            individual_code=args.individual_code
        )
        
        logging.info(f"Visualization results saved to: {output_dir}")
    except Exception as e:
        logging.error(f"Error during visualization process: {e}\n{traceback.format_exc()}")
        sys.exit(1)

def query_database(args: argparse.Namespace) -> None:
    """Main function for the query database subcommand"""
    # Handle input path
    bm_res_database = args.bm_res_database
    
    # Check if database file exists
    if not check_file_exists(bm_res_database):
        logging.error(f"Database file does not exist: {bm_res_database}")
        sys.exit(1)
    
    # Check file type
    db_file_type = detect_file_type(bm_res_database)
    if db_file_type != 'duckdb' and db_file_type != 'plain':
        logging.warning(f"Database file type may not be DuckDB: {db_file_type}")
    
    # Connect to the database
    try:
        con = duckdb.connect(bm_res_database, read_only=True)
    except Exception as e:
        logging.error(f"Error connecting to database: {e}")
        sys.exit(1)
    
    try:
        # First, get table names
        tables = con.execute("SHOW TABLES").fetchall()
        if not tables:
            logging.error("No tables in database")
            sys.exit(1)
        
        table_name = tables[0][0]  # Use the first table
        logging.info(f"Using table: {table_name}")
        
        # Get database column information
        columns = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        column_names = [col[1] for col in columns]  # col[1] is the column name
        
        logging.info(f"Table {table_name} has columns: {', '.join(column_names)}")
        
        # Execute different SQL based on query type
        if args.query_type == 'str_info':
            if not args.str_id:
                logging.error("STR ID must be specified for str_info query type")
                sys.exit(1)
            if 'id' in column_names:
                query = f"SELECT * FROM '{table_name}' WHERE id = '{args.str_id}' LIMIT {args.limit}"
            else:
                logging.error("No id column in database, cannot query by STR ID")
                sys.exit(1)
        else:
            # Default to returning the first few rows of all columns
            query = f"SELECT * FROM '{table_name}' LIMIT {args.limit}"
        
        # Execute query
        result = con.execute(query).fetchall()
        
        # Handle output
        if args.output_format == 'json':
            # Convert result to list of dictionaries
            result_list = []
            for row in result:
                result_dict = {column_names[i]: str(value) for i, value in enumerate(row) if i < len(column_names)}
                result_list.append(result_dict)
            
            # Output JSON
            if args.output == '-':
                json.dump(result_list, sys.stdout, indent=2)
            else:
                with open(args.output, 'w') as f:
                    json.dump(result_list, f, indent=2)
        else:
            # Text output
            if args.output == '-':
                # Output column names first
                print("\t".join(column_names))
                # Output data
                for row in result:
                    print("\t".join(str(value) for value in row))
            else:
                # Output to file
                with open(args.output, 'w') as f:
                    # Write column names first
                    f.write("\t".join(column_names) + "\n")
                    # Write data
                    for row in result:
                        f.write("\t".join(str(value) for value in row) + "\n")
    except Exception as e:
        logging.error(f"Error querying database: {e}\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        # Close connection
        con.close()

def visualize_single_str(bm_res_database: str, bam_path: str, reference_fasta: str, 
                         r_script_path: str, str_id: str, individual_code: str, 
                         pro_type: str, output_dir: str, prefix: str) -> Dict[str, Any]:
    """可视化单个STR的函数，被并行处理调用"""
    result = {
        'str_id': str_id,
        'individual_code': individual_code,
        'status': 'failed',
        'error': None
    }
    
    try:
        # 提取STR区域信息
        column_mapping = {
            'id': 'str_id',
            'chr': 'chrom',
            'start': 'start',
            'end': 'end',
            'motif': 'str_unit',
            'motif_len': 'str_unit_length',
            'len': 'average_length',
        }
        
        str_region_info = create_str_region_from_duckdb(
            bm_res_database, 
            str_id,
            column_mapping=column_mapping
        )
        
        if str_region_info is None:
            result['error'] = f"无法找到STR ID: {str_id}"
            return result
        
        # 获取SNP信息
        try:
            SNP_info = get_str_hsnp_info(str_region_info, individual_code)
        except Exception as e:
            SNP_info = None
        
        # 准备区域信息
        hsnp_region_label = None
        mutation_region_label = None
        
        if str_region_info is not None:
            if SNP_info is not None:
                region_list = [(str_region_info.to_region_str(), 'STR'), (SNP_info.to_region_str(), 'hSNP')]
                mutation_region_label = 'STR'
                str_id_val = str_region_info['str_id']
                hsnp_region_label = 'hSNP'
            else:
                region_list = [(str_region_info.to_region_str(), 'STR')]
                mutation_region_label = 'STR'
                str_id_val = str_region_info['str_id']
        
        # 执行可视化
        str_visulization(
            bam_path=bam_path,
            region=region_list,
            reference_fasta=reference_fasta,
            r_script_path=r_script_path,
            str_id=str_id_val,
            str_region_info=str_region_info,
            mutation_region_label=mutation_region_label,
            hsnp_region_label=hsnp_region_label,
            hsnp_info=SNP_info,
            output_dir=output_dir,
            prefix=prefix,
            run_visualization=True,
            align_samples=True,
            start_axis=0,
            pro_type=pro_type,
            individual_code=individual_code
        )
        result['status'] = 'success'
        
    except Exception as e:
        result['error'] = f"{str(e)}\n{traceback.format_exc()}"
    
    return result

def encode_image_to_base64(image_path: str) -> Optional[str]:
    """将图像文件转换为base64编码

    Args:
        image_path: 图像文件路径

    Returns:
        base64编码的字符串，如果图像不存在或无法读取则返回None
    """
    if not os.path.exists(image_path):
        logging.warning(f"图像文件不存在: {image_path}")
        return None
    
    try:
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            base64_data = base64.b64encode(img_data).decode('utf-8')
            return base64_data
    except Exception as e:
        logging.error(f"无法读取或编码图像: {image_path}, 错误: {e}")
        return None

def extract_database_info(db_path, str_id, column_list=None):
    """Query the database to extract information for a specific STR ID
    
    Args:
        db_path: Path to the database file
        str_id: STR ID to query
        column_list: List of columns to extract, or None for all columns
        
    Returns:
        Dictionary with column name as key and value from database
    """
    try:
        con = duckdb.connect(db_path, read_only=True)
        
        # Get table name
        tables = con.execute("SHOW TABLES").fetchall()
        if not tables:
            logging.error("No tables found in database")
            return {}
        
        table_name = tables[0][0]  # Use the first table
        
        # Get columns
        columns = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        column_names = [col[1] for col in columns]
        
        # Always fetch all columns if column_list is None
        columns_str = "*"
        
        # Execute the query using parameterized query to prevent SQL injection
        # Use double quotes around table_name to handle special characters
        query = f'SELECT {columns_str} FROM "{table_name}" WHERE id = ?'
        result = con.execute(query, [str_id]).fetchone()
        
        if not result:
            logging.error(f"No data found for STR ID: {str_id}")
            return {}
        
        # Convert result to dictionary
        data = {}
        for i, col in enumerate(column_names):
            if i < len(result):
                data[col] = result[i]
        
        return data
    except Exception as e:
        logging.error(f"Error querying database: {e}")
        return {}
    finally:
        if 'con' in locals():
            con.close()

def parse_complex_value(value):
    """Parse complex values like tuples from string representation
    
    Args:
        value: String representation of a complex value
        
    Returns:
        Python object or original string if parsing fails
    """
    if not isinstance(value, str):
        return value
    
    if value is None:
        return None
        
    # Try to parse as Python literal
    try:
        if value.startswith('(') and value.endswith(')'):
            return ast.literal_eval(value)
        elif value.startswith('[') and value.endswith(']'):
            return ast.literal_eval(value)
        elif value.startswith('{') and value.endswith('}'):
            return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        pass
    
    return value

def distribute_values_to_samples(value, tasks, metadata, index_col=None):
    """Distribute values to samples based on their index
    
    Args:
        value: A value that might be a list or tuple to distribute
        tasks: List of task dictionaries representing samples
        metadata: Dictionary mapping individual_code+sample_name to metadata
        index_col: Column name in metadata to use as index
        
    Returns:
        Dictionary mapping task index to appropriate value
    """
    # Parse the value if it's a string representation
    parsed_value = parse_complex_value(value)
    
    # If it's not a sequence or is a string, repeat for all samples
    if not isinstance(parsed_value, (list, tuple)) or isinstance(parsed_value, str):
        return {i: parsed_value for i in range(len(tasks))}
    
    # If it's a sequence but we don't have an index column, can't distribute
    if not index_col:
        logging.warning(f"Can't distribute sequence value without index column")
        return {i: parsed_value for i in range(len(tasks))}
    
    # Try to distribute values based on index column
    result = {}
    for i, task in enumerate(tasks):
        ind_code = task.get('individual_code', '')
        sample_name = task.get('sample_name', '')
        key = (ind_code, sample_name)
        
        if key in metadata and index_col in metadata[key]:
            try:
                index = int(metadata[key][index_col])
                if 0 <= index < len(parsed_value):
                    result[i] = parsed_value[index]
                else:
                    result[i] = f"Index out of range: {index} in value {parsed_value}"
            except (ValueError, TypeError):
                result[i] = "Invalid index"
        else:
            result[i] = "No metadata"
    
    return result

def read_column_config_file(file_path):
    """从配置文件中读取列信息和说明
    
    配置文件格式为CSV，支持以下列（不一定都存在）：
    - column_name: 数据库中的列名（必须）
    - Name: 可选的显示名称
    - Description: 列说明，将显示在HTML报告中
    - Range: 可选的值范围信息
    - Quality Indicator: 可选的质量指标
    - Formula: 可选的计算公式
    - Group: 分组名称，用于在报告中组织列
    - Level: 可选的层级信息
    - Type: 列类型，global(全局)或sample(样本特定)
    
    Args:
        file_path: 配置文件路径
        
    Returns:
        (global_columns, sample_columns, column_descriptions, column_groups, column_metadata)元组
          column_metadata是一个字典，包含每列的所有配置信息
    """
    if not os.path.exists(file_path):
        logging.error(f"列配置文件不存在: {file_path}")
        return [], [], {}, {}, {}
    
    global_columns = []
    sample_columns = []
    column_descriptions = {}
    column_groups = {}
    column_metadata = {}  # 新增: 保存每列的所有配置信息
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # 读取第一行作为标题
            header = next(reader, None)
            if not header:
                logging.error(f"列配置文件为空或格式不正确: {file_path}")
                return [], [], {}, {}, {}
            
            # 标准化标题名称
            header = [h.strip() for h in header]
            
            # 查找必要列的索引
            col_idx = -1
            type_idx = -1
            group_idx = -1
            desc_idx = -1
            
            for i, h in enumerate(header):
                h_lower = h.lower()
                if h_lower == "column_name":
                    col_idx = i
                elif h_lower == "type":
                    type_idx = i
                elif h_lower == "group":
                    group_idx = i
                elif h_lower == "description":
                    desc_idx = i
            
            # 验证必要的列是否存在
            if col_idx == -1:
                logging.error(f"列配置文件缺少必要的'column_name'列: {file_path}")
                return [], [], {}, {}, {}
            
            # 读取每一行
            for row in reader:
                if not row or len(row) <= col_idx or not row[col_idx].strip():
                    continue  # 跳过空行或列名为空的行
                
                column_name = row[col_idx].strip()
                if column_name.startswith('#'):
                    continue  # 跳过注释行
                
                # 获取类型，默认为global
                col_type = "global"
                if type_idx != -1 and len(row) > type_idx:
                    type_value = row[type_idx].strip().lower()
                    if type_value:
                        col_type = type_value
                
                # 获取分组（如果有）
                group = "其他"
                if group_idx != -1 and len(row) > group_idx:
                    group_val = row[group_idx].strip()
                    if group_val:
                        group = group_val
                
                # 获取描述（如果有）
                description = ""
                if desc_idx != -1 and len(row) > desc_idx:
                    description = row[desc_idx].strip()
                
                # 根据类型添加到相应列表
                if col_type == "sample":
                    sample_columns.append(column_name)
                else:
                    global_columns.append(column_name)
                
                # 保存描述
                if description:
                    column_descriptions[column_name] = description
                
                # 保存分组信息
                column_groups[column_name] = group
                
                # 保存该列的所有配置信息
                column_info = {}
                for i, header_name in enumerate(header):
                    if i < len(row) and row[i].strip():
                        column_info[header_name] = row[i].strip()
                column_metadata[column_name] = column_info
        
        return global_columns, sample_columns, column_descriptions, column_groups, column_metadata
    except Exception as e:
        logging.error(f"读取列配置文件时出错: {e}")
        return [], [], {}, {}, {}

def generate_html_report(
    output_dir: str, 
    str_id: str, 
    metadata_file: str, 
    database_file: str,
    sort_by: Optional[str],
    processed_tasks: List[Dict[str, Any]],
    all_metadata_samples: List[Dict[str, Any]],
    embed_images: bool = False,
    global_columns: Optional[List[str]] = None,
    sample_columns: Optional[List[str]] = None,
    index_column: Optional[str] = None,
    column_descriptions: Optional[Dict[str, str]] = None,
    column_groups: Optional[Dict[str, str]] = None,
    column_metadata: Optional[Dict[str, Dict[str, str]]] = None,
    ic_for_replacement: Optional[str] = None,
    report_filename: Optional[str] = None,
    template_file_path: Optional[str] = None,
    exclude_sample_by: Optional[str] = None
) -> str:
    """Generate HTML report for batch visualization results with database information using Table.
    
    Args:
        output_dir: Output directory
        str_id: STR ID
        metadata_file: Metadata file path
        database_file: Database file path
        sort_by: Sort by column
        processed_tasks: List of task results from the processing step.
        all_metadata_samples: List of all sample dictionaries derived from the metadata file, potentially sorted.
        embed_images: Whether to embed images as base64 in HTML
        global_columns: List of database columns to show globally
        sample_columns: List of database columns to distribute to samples
        index_column: Column in metadata to use as index for distributing values
        column_descriptions: Dictionary mapping original column names to descriptions.
        column_groups: Dictionary mapping original column names to group names.
        column_metadata: Dictionary mapping original column names to their full configuration metadata.
        ic_for_replacement: Individual code to replace {ic} in column names
        report_filename: Optional filename for the HTML report. If None, defaults to {str_id}_report.html.
        template_file_path: Optional path to a custom Jinja2 template file.
        exclude_sample_by: Column name in metadata file to exclude samples if value is TRUE (case-insensitive)
        
    Returns:
        HTML file path
    """
    # Determine the filename for the HTML report
    if report_filename:
        # Ensure the filename ends with .html
        if not report_filename.lower().endswith('.html'):
            report_filename += '.html'
        html_file = os.path.join(output_dir, report_filename)
    else:
        html_file = os.path.join(output_dir, f"{ic_for_replacement}_{str_id}_report.html")
    
    # If no column description dictionary is provided, create an empty one
    if column_descriptions is None:
        column_descriptions = {}
    
    # If no column group dictionary is provided, create an empty one
    if column_groups is None:
        column_groups = {}
    
    # If no column metadata dictionary is provided, create an empty one
    if column_metadata is None:
        column_metadata = {}
    
    # Create a column name mapping for handling {ic} replacement in column names
    column_metadata_map = {}
    if column_metadata:
        # First copy the original mapping
        for col, metadata in column_metadata.items():
            column_metadata_map[col] = metadata
            
        # Create a mapping for replaced column names
        if ic_for_replacement:
            for col, metadata in column_metadata.items():
                if '{ic}' in col:
                    # Create replaced column name
                    replaced_col = col.replace('{ic}', ic_for_replacement)
                    column_metadata_map[replaced_col] = metadata
                    
        # Also create a reverse mapping for replaced column names to original metadata
        for original_col, replaced_col in zip(global_columns + sample_columns, 
                                             [c.replace('{ic}', ic_for_replacement) if '{ic}' in c and ic_for_replacement else c 
                                              for c in global_columns + sample_columns]):
            if original_col != replaced_col and original_col in column_metadata:
                column_metadata_map[replaced_col] = column_metadata[original_col]
    
    # Extract all database information for the STR ID
    str_data = extract_database_info(database_file, str_id)
    
    # If global_columns and sample_columns are empty, use all columns in the database as global_columns
    # if (global_columns is None or len(global_columns) == 0) and (sample_columns is None or len(sample_columns) == 0):
    #     global_columns = list(str_data.keys())
    #     sample_columns = []

    if 'gt_germline' in global_columns:
        phy_str = str_data.get(f'phy_{ic_for_replacement}', None)
        if phy_str:
            try:
                phy_list = ast.literal_eval(phy_str)
                # Extract gt_germline (from third-to-last row)
                if isinstance(phy_list, (list, tuple)) and len(phy_list) >= 3:
                    gt_germline = phy_list[-3]
                    if gt_germline is not None:
                        str_data['gt_germline'] = str(gt_germline)
                        logging.debug(f"Successfully extracted gt_germline from phy_{ic_for_replacement}")
            except (ValueError, SyntaxError) as e:
                logging.warning(f"Could not parse phy_{ic_for_replacement} for gt_germline: {e}")

    if 'alt_dp' in sample_columns:
        phy_str = str_data.get(f'phy_{ic_for_replacement}', None)
        if phy_str:
            try:
                phy_list = ast.literal_eval(phy_str)
                if isinstance(phy_list, (list, tuple)) and len(phy_list) >= 12:
                    genotypes_str = phy_list[11]
                    genotypes = ast.literal_eval(str(genotypes_str))
                    if isinstance(genotypes, (list, tuple)):
                        str_data['alt_dp'] = str(genotypes)
                        logging.debug(f"Successfully extracted alt_dp from phy_{ic_for_replacement}")
            except (ValueError, SyntaxError) as e:
                logging.warning(f"Could not parse phy_{ic_for_replacement} for alt_dp: {e}")
    
    # Ensure all configured columns are displayed even if they don't exist in the database
    all_configured_columns = (global_columns or []) + (sample_columns or [])
    for col in all_configured_columns:
        if col not in str_data:
            str_data[col] = "N/A"  # Use N/A as default value for non-existing columns
    
    # --- Prepare Global Data for Table --- 
    global_data_rows_flat = []
    for col in global_columns + sample_columns or []:
        # Get original column name if replacement happened, to fetch metadata
        original_col = col
        if ic_for_replacement and f'{{{ic_for_replacement}}}' in col:
            original_col = col.replace(ic_for_replacement, '{ic}')
        
        row_data = {
            "attribute": col,
            "value": str_data.get(col, "N/A"),
            "description": column_descriptions.get(original_col, ""),
            "metadata": column_metadata.get(original_col, {})
        }

        global_data_rows_flat.append(row_data)
    
    # --- Prepare Sample-Specific Data for Table --- 
    sample_data_distributed = {}
    if sample_columns and processed_tasks:
        # 给未被 exclude 的样本分配 valid_index
        valid_index = 0
        n_valid = 0
        for sample in all_metadata_samples:
            if not sample.get('excluded', False):
                sample['valid_index'] = valid_index
                valid_index += 1
            else:
                sample['valid_index'] = None
        n_valid = valid_index
        for col in sample_columns:
            value_from_db = str_data.get(col, "N/A")
            parsed_value = parse_complex_value(value_from_db)
            # 判断是否需要跳过第一个样本
            skip_first_sample = False
            if isinstance(parsed_value, (list, tuple)):
                if len(parsed_value) == n_valid - 1:
                    skip_first_sample = True
            distributed_for_col = {}
            for i, sample_meta in enumerate(all_metadata_samples):
                if sample_meta.get('excluded', False):
                    distributed_for_col[i] = "N/A"
                    continue
                valid_index = sample_meta['valid_index']
                if parsed_value is None:
                    distributed_for_col[i] = "N/A"
                elif isinstance(parsed_value, (list, tuple)):
                    if skip_first_sample:
                        if valid_index == 0:
                            distributed_for_col[i] = "N/A"
                        elif 0 <= valid_index - 1 < len(parsed_value):
                            v = parsed_value[valid_index - 1]
                            distributed_for_col[i] = v if v is not None and v != '' else "N/A"
                        else:
                            distributed_for_col[i] = "[Index out of bounds]"
                    else:
                        if 0 <= valid_index < len(parsed_value):
                            v = parsed_value[valid_index]
                            distributed_for_col[i] = v if v is not None and v != '' else "N/A"
                        else:
                            distributed_for_col[i] = "[Index out of bounds]"
                else:
                    distributed_for_col[i] = parsed_value
            sample_data_distributed[col] = distributed_for_col
    
    print(f"sample_data_distributed: {sample_data_distributed}")
    
    # Prepare template data
    template_data = {
        'str_id': str_id,
        'metadata_file': metadata_file,
        'database_file': database_file,
        'generation_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'is_sorted': sort_by, # Use the arg to indicate sorting intention
        'embed_images': embed_images,
        'global_data_rows_flat': global_data_rows_flat,
        'tasks': [],
        'has_sample_data': bool(sample_columns)
    }
    
    # Create a lookup for processed task results by (ic, sample_name)
    processed_lookup = {
        (task.get('individual_code'), task.get('sample_name')): task 
        for task in processed_tasks
    }
    
    # Iterate through the potentially sorted list of all samples from metadata
    for i, sample_meta in enumerate(all_metadata_samples):
        ic = sample_meta.get('individual_code', 'N/A')
        sample_name = sample_meta.get('sample_name', 'N/A')
        key = (ic, sample_name)
        
        # Find corresponding processed task result or use metadata
        processed_task = processed_lookup.get(key) 
        
        excluded = sample_meta.get('excluded', False)
        excluded_reason = sample_meta.get('excluded_reason', None)
        
        task_data = {
            'status': processed_task.get('status', 'missing') if processed_task else sample_meta.get('status', 'missing'),
            'individual_code': ic,
            'bam_path': processed_task.get('bam_path', sample_meta.get('bam_path', 'N/A')) if processed_task else sample_meta.get('bam_path', 'N/A'),
            'pro_type': processed_task.get('pro_type', sample_meta.get('pro_type', 'N/A')) if processed_task else sample_meta.get('pro_type', 'N/A'),
            'sample_name': sample_name,
            'sort_value': sample_meta.get('sort_value', 'N/A'),
            'excluded': excluded,
            'excluded_reason': excluded_reason,
            # Table ready data for this sample
            'sample_data_rows_grouped': {},
            'sample_data_rows_flat': []
        }
        
        # Populate sample-specific Table data for this task using the index 'i'
        if sample_columns and not excluded:
            for col in sample_columns:
                # Get original column name for metadata lookup
                original_col = col
                if ic_for_replacement and f'{{{ic_for_replacement}}}' in col:
                    original_col = col.replace(ic_for_replacement, '{ic}')
                
                value_for_sample = sample_data_distributed.get(col, {}).get(i, "N/A")
                group = column_groups.get(col, "其他")
                
                if group not in task_data['sample_data_rows_grouped']:
                    task_data['sample_data_rows_grouped'][group] = []
                
                row_data = {
                    "attribute": col,
                    "value": value_for_sample,
                    "description": column_descriptions.get(original_col, ""),
                    "metadata": column_metadata.get(original_col, {})
                }
                task_data['sample_data_rows_grouped'][group].append(row_data)
                task_data['sample_data_rows_flat'].append(row_data)
        
        # Process image path
        image_path = None
        if processed_task and processed_task.get('status') == 'success':
            image_path = processed_task.get('image_path')
        elif sample_meta.get('status') == 'success': # Handle previously existing images
             prefix = f"{sample_meta.get('pro_type')}_{ic}_{sample_name}"
             potential_path = os.path.join(output_dir, f"{prefix}.png")
             if os.path.exists(potential_path):
                 image_path = potential_path
                 task_data['status'] = 'success' # Update status if image found
        
        if task_data['status'] == 'success' and image_path and os.path.exists(image_path):
            task_data['abs_image_path'] = os.path.abspath(image_path)
            task_data['image_path'] = image_path # Store relative/original path too
            
            # Embed image if needed
            if embed_images:
                base64_image = encode_image_to_base64(image_path)
                if base64_image:
                    task_data['base64_image'] = base64_image
        
        template_data['tasks'].append(task_data)
    
    # Load and render the template
    try:
        template_name = 'str_report.html' # Default template name
        
        # Determine template directory and name
        if template_file_path and os.path.exists(template_file_path):
            # Use custom template file
            template_dir = os.path.dirname(template_file_path)
            template_name = os.path.basename(template_file_path)
            logging.info(f"Using custom template: {template_file_path}")
        else:
            if template_file_path:
                logging.warning(f"Custom template file not found: {template_file_path}. Using default template.")
            # Use default template from package
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            template_dir = os.path.join(os.path.dirname(current_script_dir), 'templates')
            logging.info(f"Using default template from: {template_dir}")
        
        # Create Jinja2 environment and load the template
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # 处理特殊字符，确保公式等能够正确显示
        def safe_formula(formula):
            if not formula or not isinstance(formula, str):
                return ""
            # 将HTML特殊字符转义，但保留基本格式
            return formula.replace('<', '&lt;').replace('>', '&gt;')
        
        env.filters['safe_formula'] = safe_formula
        
        try:
            template = env.get_template(template_name)
        except jinja2.TemplateNotFound:
            logging.error(f"Template {template_name} not found in directory {template_dir}.")
            # Fallback to writing a very basic error report
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(f"<html><body><h1>STR Visualization Report - {str_id}</h1><p>Error: Template file {template_name} not found in {template_dir}.</p></body></html>")
            return html_file

        # Ensure each task has sample_data_rows_flat
        for task in template_data['tasks']:
            if 'sample_data_rows_flat' not in task and 'sample_data_rows_grouped' in task:
                # 如果有分组数据但没有平铺数据，从分组数据中重建平铺数据
                task['sample_data_rows_flat'] = []
                for group, rows in task['sample_data_rows_grouped'].items():
                    task['sample_data_rows_flat'].extend(rows)
            elif 'sample_data_rows_flat' not in task:
                # 确保至少有一个空数组
                task['sample_data_rows_flat'] = []
        
        # 添加全局数据标志，便于模板中检查
        template_data['global_data'] = len(template_data.get('global_data_rows_flat', [])) > 0
        
        # --- 新增：保存 task_data 到 debug 文件 ---
        try:
            debug_dir = output_dir if output_dir != '-' else '.'
            debug_file = os.path.join(debug_dir, 'task_data_debug.jsonl')
            with open(debug_file, 'a', encoding='utf-8') as f_debug:
                f_debug.write(json.dumps(task_data, ensure_ascii=False) + '\n')
        except Exception as e:
            logging.warning(f"写入 task_data_debug.jsonl 失败: {e}")
    
        # Render the template
        html_content = template.render(**template_data)
        
        # Write the HTML file
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_file
    except Exception as e:
        logging.error(f"Error rendering template: {e}\n{traceback.format_exc()}")
        # If there's an error, try to generate a simple report
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(f"<html><body><h1>STR Visualization Report - {str_id}</h1><p>Error generating template report: {e}</p></body></html>")
        return html_file

def visualize_batch_str(args: argparse.Namespace) -> None:
    """Main function for batch visualization of STR subcommand, using parallel processing"""
    # Check if input file exists
    if not check_file_exists(args.metadata_file):
        logging.error(f"Metadata file does not exist: {args.metadata_file}")
        sys.exit(1)
    
    # Check other required files
    required_files = [args.bm_res_database, args.reference_fasta, args.r_script_path]
    for file_path in required_files:
        if not check_file_exists(file_path):
            logging.error(f"File does not exist: {file_path}")
            sys.exit(1)

    # Create output directory
    if args.output_dir != '-':
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse column mapping
    column_map = {}
    if args.column_mapping:
        try:
            for pair in args.column_mapping.split(','):
                src, dst = pair.split(':')
                column_map[src.strip()] = dst.strip()
        except Exception as e:
            logging.error(f"Error parsing column mapping: {e}")
            sys.exit(1)
    
    # Default column names
    ic_col = column_map.get('ic', 'individual_code')
    path_col = column_map.get('p', 'path')
    type_col = column_map.get('t', 'type')
    
    # Get global columns and sample-specific columns
    global_columns = args.global_columns if hasattr(args, 'global_columns') else ['ref', 'phasable_{ic}']
    sample_columns = args.sample_columns if hasattr(args, 'sample_columns') else ['vaf_list_eq_{ic}']
    column_descriptions = {}
    column_groups = {}
    
    # If column configuration file is provided, read column information from file
    if hasattr(args, 'column_config') and args.column_config:
        file_global_cols, file_sample_cols, file_descriptions, file_groups, column_metadata = read_column_config_file(args.column_config)
        
        # If column configuration file exists and is not empty, always use it to define columns
        if file_global_cols or file_sample_cols:
            global_columns = file_global_cols
            sample_columns = file_sample_cols
            column_descriptions = file_descriptions
            column_groups = file_groups
            
            # Log which columns are being used
            logging.info(f"Read {len(global_columns)} global columns, {len(sample_columns)} sample columns from config file")
            logging.info(f"Read {len(column_descriptions)} column descriptions, {len(column_groups)} column groups from config file")
        else:
            column_metadata = {}
            logging.warning(f"Column configuration file {args.column_config} did not provide valid column definitions")
    else:
        column_metadata = {}
        logging.info(f"Using default column configuration: global columns={global_columns}, sample columns={sample_columns}")
    
    # Collect all individual_codes from samples to replace {ic} placeholder
    all_individual_codes = set()
    
    # 1. Task preparation phase: collect all tasks to be executed
    all_tasks = []
    task_identifiers = set()  # For deduplication
    
    # Create a mapping for finding BAM paths when processing skipped tasks
    metadata_info = {}  # Format: {(individual_code, sample_identifier): {"bam_path": path, "pro_type": type}}
    
    # Store the target individual code once to avoid redundant checks
    target_individual_code = args.individual_code if hasattr(args, 'individual_code') and args.individual_code else None
    
    exclude_sample_by = getattr(args, 'exclude_sample_by', None)
    remove_sample_by = getattr(args, 'remove_sample_by', None)
    
    excluded_samples = set()
    
    with open(args.metadata_file, 'r', newline='') as f: # Use newline='' for csv module
        # Let csv.DictReader handle the header automatically
        # It correctly parses quoted fields.
        reader = csv.DictReader(f, delimiter=',') 
        
        # Get the headers parsed by DictReader
        headers = reader.fieldnames
        if not headers:
             logging.error(f"Could not read headers from metadata file: {args.metadata_file}")
             sys.exit(1)

        # Check if required columns exist
        missing_cols = []
        for col, default_name in [(ic_col, 'individual_code'), (path_col, 'path'), (type_col, 'type')]:
            if col not in headers:
                missing_cols.append(f"{col}(mapped from {default_name})")
        
        if missing_cols:
            logging.error(f"Metadata file is missing required columns: {', '.join(missing_cols)}")
            sys.exit(1)
        
        # Process each row, preparing tasks
        # No need to skip the header row manually anymore
        for row in reader:
            # Remove by column if specified
            if remove_sample_by and remove_sample_by in row and row[remove_sample_by].strip().lower() != 'true':
                continue
            # Skip rows that don't include
            if 'include' in row and row['include'].lower() not in ['1', 'true', 'yes', 'y']:
                continue
            # Skip rows that don't match the specified individual_code
            if target_individual_code and row[ic_col] != target_individual_code:
                continue
            # Exclude by column if specified
            excluded_reason = None
            if exclude_sample_by and exclude_sample_by in row and row[exclude_sample_by].strip().lower() == 'true':
                excluded_samples.add((row[ic_col], row[args.sample_name] if args.sample_name and args.sample_name in row else os.path.basename(row[path_col]).replace('.bam', '').replace('.cram', '')))
                excluded_reason = f"{exclude_sample_by}=TRUE"
            try:
                individual_code = row[ic_col]
                bam_path = row[path_col]
                pro_type = row[type_col]
                # Add individual ID to set for replacing {ic} placeholder
                all_individual_codes.add(individual_code)
                # Check if BAM file exists
                if not check_file_exists(bam_path):
                    logging.error(f"BAM file does not exist: {bam_path}, skipping this sample")
                    continue
                # Build output prefix and file path
                if args.sample_name and args.sample_name in row:
                    sample_identifier = row[args.sample_name]
                elif 'sn' in column_map and column_map['sn'] in row:
                    sample_identifier = row[column_map['sn']]
                else:
                    sample_identifier = os.path.basename(bam_path).replace('.bam', '').replace('.cram', '')
                # Store metadata information in mapping for later lookup
                metadata_info[(individual_code, sample_identifier)] = {
                    "bam_path": bam_path,
                    "pro_type": pro_type
                }
                if args.sort_by and args.sort_by in row:
                    try:
                        sort_value = float(row[args.sort_by])
                        metadata_info[(individual_code, sample_identifier)]["sort_value"] = sort_value
                    except (ValueError, TypeError):
                        logging.warning(f"Sort column {args.sort_by} value '{row[args.sort_by]}' is not a valid number, using default sorting")
                # 标记排除原因
                if excluded_reason:
                    metadata_info[(individual_code, sample_identifier)]["excluded"] = True
                    metadata_info[(individual_code, sample_identifier)]["excluded_reason"] = excluded_reason
                prefix = f"{pro_type}_{individual_code}_{sample_identifier}"
                output_file = os.path.join(args.output_dir, f"{prefix}.png")
                task_id = f"{args.str_id}_{individual_code}_{sample_identifier}"
                if task_id in task_identifiers:
                    continue
                task_identifiers.add(task_id)
                if os.path.exists(output_file) and not args.force:
                    logging.debug(f"Skipping existing output file: {output_file}")
                    all_tasks.append((None, {
                        'str_id': args.str_id,
                        'individual_code': individual_code,
                        'status': 'skipped',
                        'file': output_file
                    }))
                    continue
                task_args = {
                    'bm_res_database': args.bm_res_database,
                    'bam_path': bam_path,
                    'reference_fasta': args.reference_fasta,
                    'r_script_path': args.r_script_path,
                    'str_id': args.str_id,
                    'individual_code': individual_code,
                    'pro_type': pro_type,
                    'output_dir': args.output_dir,
                    'prefix': prefix
                }
                task_info = {
                    'str_id': args.str_id,
                    'individual_code': individual_code,
                    'bam_path': bam_path,
                    'prefix': prefix,
                    'pro_type': pro_type,
                    'sample_name': sample_identifier,
                    'output_file': output_file
                }
                all_tasks.append((task_args, task_info))
            except Exception as e:
                logging.error(f"Error preparing task: {e}, row content: {row}")
    
    # If a single individual_code is specified, use it to replace all {ic} placeholders
    if target_individual_code:
        ic_for_replacement = target_individual_code
    # If there is only one individual_code, use it
    elif len(all_individual_codes) == 1:
        ic_for_replacement = list(all_individual_codes)[0]
    # Otherwise, keep placeholder
    else:
        ic_for_replacement = None
        if '{ic}' in ' '.join(global_columns + sample_columns):
            logging.warning(f"Found multiple individual_codes: {all_individual_codes}, but no individual_code specified. Use --individual-code parameter to specify one.")
    
    # Replace {ic} placeholder in column names, and update column descriptions and metadata keys
    if ic_for_replacement:
        # Create updated column descriptions and groups dictionaries
        updated_descriptions = {}
        updated_groups = {}
        updated_metadata = {}  # Add this line, for updating metadata dictionary
        
        # Replace global column names and description keys
        updated_global_columns = []
        for col in global_columns:
            new_col = col.replace('{ic}', ic_for_replacement)
            updated_global_columns.append(new_col)
            # Update description dictionary keys
            if col in column_descriptions:
                updated_descriptions[new_col] = column_descriptions[col]
            # Update metadata dictionary keys
            if col in column_metadata:
                updated_metadata[new_col] = column_metadata[col]
        
        # Replace sample column names and description keys
        updated_sample_columns = []
        for col in sample_columns:
            new_col = col.replace('{ic}', ic_for_replacement)
            updated_sample_columns.append(new_col)
            # Update description dictionary keys
            if col in column_descriptions:
                updated_descriptions[new_col] = column_descriptions[col]
            # Update metadata dictionary keys
            if col in column_metadata:
                updated_metadata[new_col] = column_metadata[col]
        
        # Update groups dictionary keys
        for col, group in column_groups.items():
            new_col = col.replace('{ic}', ic_for_replacement)
            updated_groups[new_col] = group
        
        # Use updated columns and descriptions
        global_columns = updated_global_columns
        sample_columns = updated_sample_columns
        column_descriptions = updated_descriptions
        column_groups = updated_groups
        column_metadata = updated_metadata  # Add this line, update column metadata
    
    # 2. Task execution phase: use ProcessPoolExecutor to process tasks in parallel
    tasks_to_run = [(args, info) for args, info in all_tasks if args is not None]
    total_tasks = len(tasks_to_run)
    completed_tasks = 0
    processed_count = 0
    error_count = 0
    skipped_count = sum(1 for args, _ in all_tasks if args is None)
    
    processed_results = []
    
    logging.info(f"Found {total_tasks} tasks to process, {skipped_count} tasks skipped")
    
    # Set number of parallel processes, default to half of CPU cores or 4, whichever is smaller
    max_workers = min(args.workers if hasattr(args, 'workers') else 4, os.cpu_count() or 4)
    logging.info(f"Using {max_workers} processes to process tasks in parallel")
    
    # Use progress bar to track progress
    with tqdm(total=total_tasks, desc="Processing progress") as progress_bar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task_args, task_info in tasks_to_run:
                future = executor.submit(visualize_single_str, **task_args)
                future_to_task[future] = task_info
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    task_info = future_to_task[future]
                    
                    if result['status'] == 'success':
                        processed_count += 1
                        # Build task result directly, without relying on file system check
                        task_result = {
                            'status': 'success',
                            'str_id': result['str_id'],
                            'individual_code': result['individual_code'],
                            'bam_path': task_info['bam_path'],
                            'pro_type': task_info['pro_type'],
                            'sample_name': task_info['sample_name'],
                            'image_path': task_info['output_file']  # Use expected output file path
                        }
                        processed_results.append(task_result)
                    else:
                        error_count += 1
                        logging.error(f"Error processing sample {result['individual_code']}: {result['error']}")
                except Exception as e:
                    error_count += 1
                    task_info = future_to_task[future]
                    logging.error(f"Error processing sample {task_info['individual_code']}: {e}")
                
                completed_tasks += 1
                progress_bar.update(1)
    
    # 3. Result statistics
    total_count = processed_count + error_count + skipped_count
    logging.info(f"Batch processing completed: success={processed_count}, failed={error_count}, skipped={skipped_count}, total={total_count}")
    
    # 4. Generate HTML report
    if not args.no_html:
        # Check if there are any tasks (processed or from metadata) to show
        if all_tasks:
            # Collect all successful task information
            processed_tasks = []
            
            # Add successful tasks from current run
            processed_tasks.extend(processed_results)
            
            # Add skipped tasks
            for task_args, task_info in all_tasks:
                if task_args is None:  # This is a skipped task
                    output_file = task_info.get('file')
                    if output_file and os.path.exists(output_file):
                        # Parse prefix from file name
                        prefix = os.path.basename(output_file).replace('.png', '')
                        name_parts = prefix.split('_')
                        if len(name_parts) >= 3:
                            pro_type = name_parts[0]
                            individual_code = name_parts[1]
                            sample_name = '_'.join(name_parts[2:])
                            
                            # Try to find BAM path from metadata
                            bam_path = 'Unknown (skipped task)'
                            if (individual_code, sample_name) in metadata_info:
                                info = metadata_info[(individual_code, sample_name)]
                                bam_path = info["bam_path"]
                            
                            task_result = {
                                'status': 'success',
                                'str_id': task_info['str_id'],
                                'individual_code': individual_code,
                                'bam_path': bam_path,
                                'pro_type': pro_type,
                                'sample_name': sample_name,
                                'image_path': output_file
                            }
                            processed_tasks.append(task_result)
            
            # Create a set to quickly check processed files
            processed_files = {os.path.basename(task.get('image_path', '')) for task in processed_tasks}
            
            # Check if there are any other PNG files that were not collected
            for png_file in processed_files:
                if png_file not in processed_files:
                    # Try to parse information from file name
                    try:
                        # Assume format is: {type}_{individual_code}_{sample_name}.png
                        prefix = png_file.replace('.png', '')
                        name_parts = prefix.split('_')
                        if len(name_parts) >= 3:
                            pro_type = name_parts[0]
                            individual_code = name_parts[1]
                            sample_name = '_'.join(name_parts[2:])
                            
                            # Try to find BAM path from metadata
                            bam_path = 'Unknown (from previous run)'
                            if (individual_code, sample_name) in metadata_info:
                                info = metadata_info[(individual_code, sample_name)]
                                bam_path = info["bam_path"]
                            
                            task_result = {
                                'status': 'success',
                                'str_id': args.str_id,
                                'individual_code': individual_code,
                                'bam_path': bam_path,
                                'pro_type': pro_type,
                                'sample_name': sample_name,
                                'image_path': os.path.join(args.output_dir, png_file)
                            }
                            processed_tasks.append(task_result)
                    except Exception as e:
                        logging.warning(f"Cannot parse information from file name {png_file}: {e}")
            
            # processed_tasks list now contains results from current run
            # all_tasks contains metadata about all potential tasks
            
            # Generate HTML report
            if processed_tasks:
                try:
                    # The list `all_metadata_samples` is already sorted if --sort_by was used
                    if args.sort_by:
                        # Create a lookup for processed task results by (ic, sample_name)
                        processed_lookup = {
                            (task.get('individual_code'), task.get('sample_name')): task 
                            for task in processed_tasks
                        }
                        
                        # Create all_metadata_samples from metadata_info keys
                        all_metadata_samples = []
                        for i, (key, info) in enumerate(metadata_info.items()):
                            individual_code, sample_name = key
                            sample_info = {
                                'individual_code': individual_code,
                                'sample_name': sample_name,
                                'bam_path': info.get('bam_path', 'Unknown'),
                                'pro_type': info.get('pro_type', 'Unknown'),
                                'sort_value': info.get('sort_value', float('inf')),
                                'status': 'missing',
                                'index': i,
                                'excluded': info.get('excluded', False),
                                'excluded_reason': info.get('excluded_reason', None)
                            }
                            all_metadata_samples.append(sample_info)
                        
                        # Sort all_metadata_samples
                        all_metadata_samples.sort(key=lambda x: x.get('sort_value', float('inf')))
                        
                        # Assign index after sorting
                        for i, sample in enumerate(all_metadata_samples):
                            sample['index'] = i
                    
                    # print(f"Total samples for report (sorted: {args.sort_by is not None}): {len(all_metadata_samples)}")
                    # print(f"Global columns for report: {global_columns}")
                    # print(f"Sample columns for report: {sample_columns}")
                    # print(f"Processed results: {processed_tasks}")
                    
                    html_file = generate_html_report(
                        args.output_dir,
                        args.str_id,
                        args.metadata_file,
                        args.bm_res_database,
                        args.sort_by,
                        processed_tasks,
                        all_metadata_samples,
                        embed_images=args.embed_images,
                        global_columns=global_columns,
                        sample_columns=sample_columns,
                        index_column='index',
                        column_descriptions=column_descriptions,
                        column_groups=column_groups,
                        column_metadata=column_metadata,
                        ic_for_replacement=ic_for_replacement,
                        report_filename=args.report_name,
                        template_file_path=args.template_file,
                        exclude_sample_by=exclude_sample_by
                    )
                    logging.info(f"HTML report generated: {html_file}")
                except Exception as e:
                    logging.error(f"Error generating HTML report: {e}")

def main():
    """Main function to handle command line arguments and execute corresponding functions"""
    # Create the main parser
    parser = argparse.ArgumentParser(
        description='BayesMonstr Result Conversion and STR Visualization Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Display detailed log information')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    
    # Create sub-command parsers
    subparsers = parser.add_subparsers(dest='command', help='Sub-commands')
    
    # Visualization sub-command
    visualize_parser = subparsers.add_parser('visualize', help='Visualize STR regions')
    visualize_parser.add_argument('-i', '--bm_res_database', required=True, help='Path to BayesMonstr result database')
    visualize_parser.add_argument('-b', '--bam_path', required=True, help='Path to BAM file')
    visualize_parser.add_argument('-r', '--reference_fasta', required=True, help='Path to reference genome FASTA file')
    visualize_parser.add_argument('-s', '--r_script_path', required=True, help='Path to R script')
    visualize_parser.add_argument('-id', '--str_id', required=True, help='STR ID')
    visualize_parser.add_argument('-ic', '--individual_code', required=True, help='Individual code')
    visualize_parser.add_argument('-pt', '--pro_type', default="MDA", help='Processing type')
    visualize_parser.add_argument('-o', '--output_dir', required=True, help='Output directory, use "-" for standard output')
    visualize_parser.add_argument('-p', '--prefix', help='Output file prefix')
    visualize_parser.add_argument('-f', '--force', action='store_true', help='Overwrite existing output files')
    visualize_parser.set_defaults(func=visualize_str)
    
    # Query database sub-command
    query_parser = subparsers.add_parser('query', help='Query BayesMonstr result database')
    query_parser.add_argument('-i', '--bm_res_database', required=True, help='Path to BayesMonstr result database')
    query_parser.add_argument('-t', '--query_type', choices=['str_info', 'all'], 
                            default='all', help='Query type')
    query_parser.add_argument('-id', '--str_id', help='STR ID for str_info query type')
    query_parser.add_argument('-ic', '--individual_code', help='Individual code for individual_info query type')
    query_parser.add_argument('-l', '--limit', type=int, default=10, help='Limit the number of returned rows')
    query_parser.add_argument('-o', '--output', default='-', help='Output file path, default is standard output ("-")')
    query_parser.add_argument('-f', '--output_format', choices=['text', 'json'], default='text', help='Output format')
    query_parser.set_defaults(func=query_database)
    
    # Batch visualization sub-command
    batch_parser = subparsers.add_parser('visualize_batch', help='Batch visualize multiple STR samples')
    batch_parser.add_argument('-m', '--metadata_file', required=True, 
                             help='Metadata file containing sample information (CSV format)')
    batch_parser.add_argument('-i', '--bm_res_database', required=True, 
                             help='Path to BayesMonstr result database')
    batch_parser.add_argument('-r', '--reference_fasta', required=True, 
                             help='Path to reference genome FASTA file')
    batch_parser.add_argument('-s', '--r_script_path', required=True, 
                             help='Path to R script')
    batch_parser.add_argument('-id', '--str_id', required=True, 
                             help='STR ID to process')
    batch_parser.add_argument('-sn', '--sample_name', 
                             help='Column name in metadata file for file naming, default is BAM/CRAM file name')
    batch_parser.add_argument('-cm', '--column_mapping', 
                             help='Column name mapping, e.g., "ic:individual_code,p:path,t:type"')
    batch_parser.add_argument('-o', '--output_dir', required=True, 
                             help='Output directory')
    batch_parser.add_argument('-f', '--force', action='store_true', 
                             help='Overwrite existing output files')
    batch_parser.add_argument('-w', '--workers', type=int, default=4,
                             help='Number of worker processes for parallel processing, default is 4')
    batch_parser.add_argument('--no-html', action='store_true',
                             help='Do not generate HTML report')
    batch_parser.add_argument('--embed-images', action='store_true',
                             help='Embed images in base64 format in the HTML report for self-contained report')
    batch_parser.add_argument('-so', '--sort_by', 
                             help='Column name in metadata file for sorting (numeric column, sorted in ascending order)')
    batch_parser.add_argument('--global-columns', nargs='+', default=['ref', 'phasable_{ic}'],
                             help='Database columns to show globally in the report')
    batch_parser.add_argument('--sample-columns', nargs='+', default=['vaf_list_eq_{ic}'],
                             help='Database columns to distribute across samples')
    batch_parser.add_argument('--individual-code', 
                             help='Individual code to replace {ic} in column names')
    batch_parser.add_argument('--column-config', 
                             help='Path to a CSV file defining columns to display and their descriptions')
    batch_parser.add_argument('--report-name',
                             help='Specify the filename for the HTML report (e.g., my_report.html)')
    batch_parser.add_argument('--template-file',
                             help='Path to a custom Jinja2 template file for the HTML report')
    batch_parser.add_argument('--exclude-sample-by', '-es',
                             help='Column name in metadata file to exclude samples if value is TRUE (case-insensitive)')
    batch_parser.add_argument('--remove-sample-by', '-rs',
                             help='Column name in metadata file to keep only samples where value is TRUE (case-insensitive)')
    batch_parser.set_defaults(func=visualize_batch_str)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # If no sub-command is specified, show help information
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    # Execute the corresponding function
    args.func(args)

if __name__ == "__main__":
    main()