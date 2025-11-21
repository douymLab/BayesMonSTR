from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd
import ast
from in_sight.utils import HSNPInfo

## essential infomation for str region processing
class STRRegion(BaseModel):
    chrom: str
    start: int
    end: int
    str_unit: str
    str_id: str
    str_unit_length: int
    average_length: float
    additional_info: Dict[str, Any] = {}  # Field to capture extra keys

    model_config = {
        'extra': 'allow'
    }

    def __getitem__(self, key):
        if key in self.__class__.model_fields:
            return getattr(self, key)
        return self.__dict__.get(key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
    
    def to_region_str(self) -> str:
        return f"{self.chrom}:{self.start}-{self.end}"

def create_str_region_from_file(file_path: str, str_id: str, sep: str = '\t') -> STRRegion:
    """Create a STRRegion object from a plain text file with field detection and warning for missing fields."""
    # Read the table file
    df = pd.read_csv(file_path, sep=sep)

    # Filter the DataFrame for the desired str_id
    matching_records = df[df['str_id'] == str_id].to_dict(orient='records')
    if not matching_records:
        print(f"Warning: str_id '{str_id}' not found in file.")
        return None
    str_region_info = matching_records[0]

    # Get the expected field names from STRRegion (excluding additional_info)
    expected_fields = set(STRRegion.model_fields.keys())
    if 'additional_info' in expected_fields:
        expected_fields.remove('additional_info')

    # Check for any missing expected fields in the TSV data
    missing_fields = expected_fields - set(str_region_info.keys())
    if missing_fields:
        print(f"Warning: The following expected fields are missing in the file: {missing_fields}")

    # Split the dictionary into expected fields and additional fields
    main_fields = {k: v for k, v in str_region_info.items() if k in expected_fields}
    additional_fields = {k: v for k, v in str_region_info.items() if k not in expected_fields}

    # Create the STRRegion model; extra fields go into the additional_info field
    return STRRegion(**main_fields, additional_info=additional_fields)

def create_str_region_from_duckdb(db_path: str, str_id: str, column_mapping: Dict[str, str] = None) -> STRRegion:
    """Create a STRRegion object from a DuckDB database with field detection and warning for missing fields."""
    import duckdb
    
    # 定义默认的数据库列名到STRRegion字段的映射
    default_mapping = {
        'id': 'str_id',
        'start': 'start',
        'end': 'end',
        'motif': 'str_unit',
        'motif_len': 'str_unit_length',
        'len': 'average_length',
        'chr': 'chrom'
    }
    
    # 使用自定义映射或默认映射
    column_mapping = column_mapping or default_mapping
    
    # 连接到DuckDB数据库
    try:
        conn = duckdb.connect(db_path, read_only=True)
        
        # 构建查询所有可能字段的SQL语句
        query = f"""
        SELECT *
        FROM data
        WHERE "id" = '{str_id}'
        LIMIT 1
        """
        
        # 执行查询并获取结果
        result = conn.execute(query).fetchone()
        if not result:
            print(f"Warning: str_id '{str_id}' not found in database.")
            return None
            
        # 获取列名
        columns = [col[0] for col in conn.execute("SELECT * FROM data LIMIT 0").description]
        
        # 转换为字典
        str_region_info = dict(zip(columns, result))
        
        # 获取STRRegion期望的字段
        expected_fields = set(STRRegion.model_fields.keys())
        if 'additional_info' in expected_fields:
            expected_fields.remove('additional_info')
        
        # 映射字段名
        mapped_info = {}
        for db_col, model_field in column_mapping.items():
            if db_col in str_region_info:
                mapped_info[model_field] = str_region_info[db_col]
        
        # 检查缺失的必要字段
        missing_fields = expected_fields - set(mapped_info.keys())
        if missing_fields:
            print(f"Warning: The following expected fields are missing in the database: {missing_fields}")
        
        # 分离主要字段和额外字段
        main_fields = {k: v for k, v in mapped_info.items() if k in expected_fields}
        additional_fields = {k: v for k, v in str_region_info.items() if k not in column_mapping.keys()}
        
        # 创建STRRegion对象
        return STRRegion(**main_fields, additional_info=additional_fields)
        
    except Exception as e:
        print(f"Error connecting to or querying DuckDB database: {e}")
        return None

def get_str_hsnp_info(str_region_info: STRRegion, individual_code: str = '1') -> Optional[HSNPInfo]:
    """
    Get hSNP information for a specific STR ID
    
    Args:
        str_region_info: STRRegion object containing STR region information
        individual_code: Individual code for hSNP data (default: '1')
        
    Returns:
        HSNPInfo object with hSNP information or None if not found
    """
    try:
        # 获取基本信息
        str_id = str_region_info.str_id
        chrom = str_region_info.chrom
        
        if not str_id or not chrom:
            print(f"Missing str_id or chrom in region info")
            return None
        
        # 构建hSNP键名
        hsnp_start_key = f'hSNP_start_{individual_code}'
        hsnp_key = f'hSNP_{individual_code}'
        
        # 检查键是否存在
        if hsnp_start_key not in str_region_info.additional_info or hsnp_key not in str_region_info.additional_info:
            print(f"No hSNP information found for {str_id} with individual code {individual_code}")
            return None
        
        # 获取hSNP数据
        hsnp_start = str_region_info.additional_info[hsnp_start_key]
        hsnp = str_region_info.additional_info[hsnp_key]
        
        # 验证数据有效性
        if pd.isna(hsnp_start) or pd.isna(hsnp) or hsnp == 'None':
            print(f"Invalid hSNP information for {str_id} with individual code {individual_code}")
            return None
        
        # 解析hSNP数据 
        hsnp_base_tuple = ast.literal_eval(hsnp)
        hsnp_base = hsnp_base_tuple[0]
        hsnp_alt = hsnp_base_tuple[1]
        
        # 创建并返回HSNPInfo对象
        return HSNPInfo.create(individual_code, chrom, hsnp_start, hsnp_base, hsnp_alt, is_one_based=False)
        
    except KeyError as e:
        print(f"Missing key in str_region_info: {e}")
        return None
    except ValueError as e:
        print(f"Error parsing hSNP data for {str_id}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error processing hSNP info for {str_id}: {e}")
        return None
