import pysam
from pysam import FastaFile, VariantFile, AlignmentFile
from pathlib import Path
import psutil
import os
from typing import Tuple
import time
import subprocess
from PIL import Image
import io
import cairosvg
from pydantic import BaseModel
from typing import Dict, List, Any
import json


def check_chromosome_prefix(bam_path, chromosome):
    """
    Check if the BAM file uses 'chr' prefix for chromosome names.
    :param bam_path: Path to the BAM file.
    :param chromosome: Chromosome number as a string (e.g., "15").
    :return: The correct chromosome name to use with this BAM file.
    """
    with pysam.AlignmentFile(bam_path, "rb") as bam_file:
        try:
            # Try fetching reads with 'chr' prefix
            if bam_file.count('chr' + chromosome) > 0:
                return 'chr' + chromosome
        except ValueError:
            pass

        try:
            # Try fetching reads without 'chr' prefix
            if bam_file.count(chromosome) > 0:
                return chromosome
        except ValueError:
            pass

    return None  # If neither worked, chromosome name is not found

# Example usage
# bam_path = 'data/bam/SRR19794569_200000000.bam'
# chromosome = '15'
# correct_chromosome_name = check_chromosome_prefix(bam_path, chromosome)
# if correct_chromosome_name:
#     print(f"Correct chromosome name to use: {correct_chromosome_name}")
# else:
#     print("Chromosome name not found in BAM file.")


def adjust_chromosome_name(file_path, chromosome, file_type='fasta'):
    """
    Adjust the chromosome name based on the naming convention used in the file.
    
    :param file_path: Path to the file (BAM, Fasta, or VCF).
    :param chromosome: The chromosome name to adjust.
    :param file_type: The type of file ('bam', 'fasta', or 'vcf').
    :return: Adjusted chromosome name or None if not found.
    """
    try:
        if file_type == 'fasta':
            with FastaFile(file_path) as fasta_file:
                if chromosome in fasta_file.references:
                    return chromosome
                alternative_chrom = ('chr' + chromosome) if not chromosome.startswith('chr') else chromosome[3:]
                if alternative_chrom in fasta_file.references:
                    return alternative_chrom
        elif file_type == 'bam':
            with AlignmentFile(file_path, "rb") as bam_file:
                if chromosome in bam_file.references:
                    return chromosome
                alternative_chrom = ('chr' + chromosome) if not chromosome.startswith('chr') else chromosome[3:]
                if alternative_chrom in bam_file.references:
                    return alternative_chrom
        elif file_type == 'vcf':
            with VariantFile(file_path) as vcf_file:
                chrom_names = [contig for contig in vcf_file.header.contigs]
                if chromosome in chrom_names:
                    return chromosome
                alternative_chrom = ('chr' + chromosome) if not chromosome.startswith('chr') else chromosome[3:]
                if alternative_chrom in chrom_names:
                    return alternative_chrom
    except ValueError as e:
        print(f"Error adjusting chromosome name: {e}")
        return None

    return None  # If the chromosome name doesn't fit any convention

# Example usage
# fasta_path = 'data/ref/hg38.fa.gz'
# bam_path = 'data/bam/SRR19794569_200000000.bam'
# chromosome = '15'
# adjusted_chrom_fasta = adjust_chromosome_name(fasta_path, chromosome, 'fasta')

def get_flanking_sequences(ref_path, motif, flank_length=3, start=0):
    """
    Fetches the left and right flanking sequences of a specified length for a given STR motif using pysam.

    Parameters:
    - ref_path: Path to the reference genome.
    - motif: The motif object containing chromosome, start, and end positions.
    - flank_length: The number of bases to fetch from both sides of the STR.
    - start: The starting number of coordinates

    Returns:
    - A tuple containing the left and right flanking sequences.
    """
    ref = pysam.FastaFile(ref_path)
    adjusted_chrom_name = adjust_chromosome_name(ref_path, motif['chrom'], 'fasta')

    # Adjust start and end positions to include flanking sequences
    # Ensuring the start position does not go below 0
    start_with_flank = max(0, motif['start']-(1-start) - flank_length)
    end_with_flank = motif['end'] + flank_length

    # Fetch the extended sequence
    extended_sequence = ref.fetch(adjusted_chrom_name, start_with_flank, end_with_flank)

    # Calculate the positions of the STR and flanking sequences within the extended sequence
    left_flank_start = start
    left_flank_end = flank_length
    right_flank_start = len(extended_sequence) - flank_length
    right_flank_end = len(extended_sequence)

    # Extract the flanking sequences
    left_flank = extended_sequence[left_flank_start:left_flank_end].upper()
    right_flank = extended_sequence[right_flank_start:right_flank_end].upper()

    ref.close()  # Don't forget to close the reference file after fetching the sequences

    return left_flank, right_flank

def get_head_softclip(read):
    """
    Extracts the soft-clipped sequence and its length at the head (5' end) of a read.
    
    Parameters:
    read (pysam.AlignedSegment): The read from which to extract the soft-clipped sequence.
    
    Returns:
    tuple: A tuple containing the soft-clipped sequence and its length at the head of the read.
           If no soft-clipped sequence exists or the read has no valid CIGAR string, returns an empty string and 0.
    """
    # Extract the CIGAR tuple
    cigar = read.cigartuples
    
    # Check if the CIGAR string is valid
    if cigar is None or len(cigar) == 0:
        return "", 0

    # Check if the first CIGAR operation is a soft clip (5' end)
    if cigar[0][0] == 4:  # 4 corresponds to the S operation in CIGAR
        softclip_length = cigar[0][1]
        # Extract the soft-clipped sequence from the read
        softclip_sequence = read.query_sequence[:softclip_length]
        return softclip_sequence, softclip_length
    return "", 0

def get_head_hardclip(read):
    """
    Extracts the hard-clipped length at the head (5' end) of a read.
    
    Parameters:
    read (pysam.AlignedSegment): The read from which to extract the hard-clipped length.
    
    Returns:
    int: The length of the hard-clipped sequence at the head of the read.
         If no hard-clipped sequence exists or the read has no valid CIGAR string, returns 0.
    """
    # Extract the CIGAR tuple
    cigar = read.cigartuples
    
    # Check if the CIGAR string is valid
    if cigar is None or len(cigar) == 0:
        return 0

    # Check if the first CIGAR operation is a hard clip (5' end)
    if cigar[0][0] == 5:  # 5 corresponds to the H operation in CIGAR
        return cigar[0][1]
    return 0

def determine_file_type(filename):
    """
    Determine the file type (BAM, CRAM, SAM) based on the filename, supporting compressed files.

    Parameters:
        filename (str): The name or path of the file.

    Returns:
        str: The file type ('BAM', 'CRAM', 'SAM') or 'Unknown'.
    """
    suffixes = Path(filename).suffixes
    if not suffixes:
        return 'Unknown'
    
    # Get the last actual extension
    extension = suffixes[-1].lower()
    
    if extension == '.bam':
        return 'BAM'
    elif extension == '.cram':
        return 'CRAM'
    elif extension == '.sam':
        return 'SAM'
    else:
        return 'Unknown'

def get_file_mode(file_type: str) -> str:
    """Determine file mode based on file type"""
    if file_type == "BAM":
        return "rb"
    elif file_type == "CRAM":
        return "rc"
    elif file_type == "SAM":
        return "r"
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
def parse_region(region_str: str) -> Tuple[str, int, int]:
    """Parse region string into chrom, start, end components"""
    if ':' not in region_str:
        raise ValueError(f"Invalid region format: {region_str}")
    
    chrom, pos_part = region_str.split(':', 1)
    pos_part = pos_part.replace(',', '')  # 移除可能存在的千位分隔符
    
    if '-' in pos_part:
        start_str, end_str = pos_part.split('-', 1)
    else:
        start_str = end_str = pos_part
    
    try:
        start = int(start_str)
        end = int(end_str)
    except ValueError:
        raise ValueError(f"Invalid position in region: {region_str}")
    
    # 新增处理：当start=end时自动将end+1
    if start == end:
        end += 1
    
    if start > end:
        raise ValueError(f"Start position ({start}) cannot be greater than end position ({end})")
    
    return chrom, start, end

def get_memory_usage() -> str:
    """Get current memory usage in a human-readable format"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return f"{memory_mb:.2f} MB"


def convert_pdf_to_png(pdf_path: str, png_path: str, dpi: int = 300, retries: int = 3,
                      background_color: str = 'white', max_dpi: int = 1200) -> bool:
    """Convert PDF to PNG using ImageMagick with automatic DPI adjustment and error handling.
    
    Args:
        pdf_path (str): Path to input PDF file
        png_path (str): Path to output PNG file
        dpi (int): Initial DPI value for conversion (default: 300)
        retries (int): Number of retry attempts with reduced DPI (default: 3)
        background_color (str): Background color for the PNG (default: 'white')
        max_dpi (int): Maximum allowed DPI value (default: 1200)
        
    Returns:
        bool: True if conversion successful, False otherwise
        
    Raises:
        FileNotFoundError: If input PDF file does not exist
        ValueError: If DPI value is invalid
    """
    start_time = time.time()
    current_dpi = min(max_dpi, max(1, dpi))  # 确保DPI在有效范围内
    
    # 验证输入文件
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Input PDF file not found: {pdf_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    
    for attempt in range(retries + 1):
        try:
            # 构建ImageMagick命令
            cmd = [
                'convert',
                '-density', str(current_dpi),
                '-background', background_color,
                '-alpha', 'remove',
                '-quality', '95',
                '-strip',  # 移除元数据以减小文件大小
                pdf_path,
                png_path
            ]
            
            # 执行转换命令
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            # 验证输出文件
            if os.path.exists(png_path) and os.path.getsize(png_path) > 0:
                print(f"Successfully converted {pdf_path} to {png_path} (DPI: {current_dpi})")
                return True
            else:
                raise RuntimeError("Output file is empty or was not created")
                
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Conversion failed at DPI {current_dpi}: {e.stderr}")
            if attempt < retries:
                # 指数退避策略
                current_dpi = max(1, current_dpi // 2)
                print(f"Retrying with DPI {current_dpi}...")
                time.sleep(1)  # 添加短暂延迟
            else:
                print("ERROR: Conversion failed after all attempts")
                return False
        except Exception as e:
            print(f"Unexpected error during PDF to PNG conversion: {str(e)}")
            return False
    
    return False

def convert_svg_to_png(svg_path: str, png_path: str, dpi: int = 300, retries: int = 3, 
                      background_color: str = 'white', max_dpi: int = 1200) -> bool:
    """Convert SVG to PNG with automatic DPI adjustment and error handling.
    
    Args:
        svg_path (str): Path to input SVG file
        png_path (str): Path to output PNG file
        dpi (int): Initial DPI value for conversion (default: 300)
        retries (int): Number of retry attempts with reduced DPI (default: 3)
        background_color (str): Background color for the PNG (default: 'white')
        max_dpi (int): Maximum allowed DPI value (default: 1200)
        
    Returns:
        bool: True if conversion successful, False otherwise
        
    Raises:
        FileNotFoundError: If input SVG file does not exist
        ValueError: If DPI value is invalid
    """
    start_time = time.time()
    current_dpi = min(max_dpi, max(1, dpi))  # 确保DPI在有效范围内
    
    # 验证输入文件
    if not os.path.exists(svg_path):
        raise FileNotFoundError(f"Input SVG file not found: {svg_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    
    for attempt in range(retries + 1):
        try:
            # 禁用PIL的最大图像像素限制
            Image.MAX_IMAGE_PIXELS = None
            
            # 读取SVG文件
            with open(svg_path, 'rb') as svg_file:
                svg_data = svg_file.read()
            
            # 使用cairosvg进行转换
            png_data = cairosvg.svg2png(
                bytestring=svg_data,
                dpi=current_dpi,
                background_color=background_color
            )
            
            # 使用PIL处理图像
            with Image.open(io.BytesIO(png_data)) as img:
                # 确保图像是RGB模式
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, background_color)
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 保存图像
                img.save(
                    png_path,
                    "PNG",
                    dpi=(current_dpi, current_dpi),
                    optimize=True,
                    quality=95
                )
            
            # 验证输出文件
            if os.path.exists(png_path) and os.path.getsize(png_path) > 0:
                print(f"Successfully converted to PNG: {png_path} (DPI: {current_dpi})")
                return True
            else:
                raise RuntimeError("Output file is empty or was not created")
                
        except Exception as e:
            print(f"ERROR: Conversion failed at DPI {current_dpi}: {str(e)}")
            if attempt < retries:
                # 指数退避策略
                current_dpi = max(1, current_dpi // 2)
                print(f"Retrying with DPI {current_dpi}...")
                time.sleep(1)  # 添加短暂延迟
            else:
                print("ERROR: Conversion failed after all attempts")
                return False
    
    return False


class HSNPVariant(BaseModel):
    """表示单个hSNP变异的模型
    
    Attributes:
        chrom (str): 染色体标识
        position (int): 变异在基因组上的位置
        reference_base (str): 参考基因组碱基
        alternate_base (str): 变异碱基
        is_one_based (bool): 坐标系统是否为1-based (True)或0-based (False)
        
    Examples:
        >>> variant = HSNPVariant(position=1000, reference_base="A", alternate_base="G")
        >>> print(variant.position)
        1000
        >>> print(variant.to_tuple())
        ('A', 'G')
    """
    chrom: str
    position: int
    reference_base: str
    alternate_base: str
    is_one_based: bool = True
    
    def __getitem__(self, key):
        if key in self.__class__.model_fields:
            return getattr(self, key)
        return self.__dict__.get(key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
    
    def to_json(self, compact: bool = True) -> str:
        """将变异信息转换为JSON字符串，便于存储和跨语言使用
        
        Args:
            compact (bool): 是否生成紧凑格式的JSON，默认为True
            
        Returns:
            str: JSON字符串
            
        Examples:
            >>> variant = HSNPVariant(chrom="1", position=1000, reference_base="A", alternate_base="G")
            >>> json_str = variant.to_json()
            >>> # 在Python中解析
            >>> import json
            >>> data = json.loads(json_str)
            >>> print(data["position"])
            1000
            >>> # 在R中解析
            >>> # library(jsonlite)
            >>> # data <- fromJSON(json_str)
            >>> # print(data$position)
        """
        data = {
            "version": "1.0",
            "chrom": self.chrom,
            "position": self.position,
            "reference_base": self.reference_base,
            "alternate_base": self.alternate_base,
            "is_one_based": self.is_one_based,
            "variant_id": f"{self.chrom}-{self.position}-{self.reference_base}-{self.alternate_base}"
        }
        
        if compact:
            return json.dumps(data, separators=(',', ':'))
        else:
            return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str):
        """从JSON字符串创建HSNPVariant实例
        
        Args:
            json_str: JSON字符串
            
        Returns:
            HSNPVariant: 新的变异实例
            
        Examples:
            >>> json_str = '{"version":"1.0","chrom":"1","position":1000,"reference_base":"A","alternate_base":"G","is_one_based":true}'
            >>> variant = HSNPVariant.from_json(json_str)
            >>> print(variant.chrom)
            1
        """
        data = json.loads(json_str)
        version = data.pop("version", "1.0")
        variant_id = data.pop("variant_id", None)  # 移除不需要的字段
        
        # 可以根据版本添加兼容性处理
        
        return cls(**data)
    
    @property
    def summary(self) -> str:
        """返回变异的简要描述"""
        if self.is_one_based:
            based = "1-based"
        else:
            based = "0-based"
        print(f"position is based on {based}")
        return f"{self.chrom}:{self.position}:{self.reference_base}>{self.alternate_base}"
    
    def __str__(self) -> str:
        return self.summary

class HSNPInfo(BaseModel):
    """表示特定个体的单个hSNP变异信息
    
    Attributes:
        individual_code (str): 个体标识码
        variant (HSNPVariant): 单个变异信息
        
    Examples:
        >>> # 创建新的HSNPInfo实例
        >>> variant = HSNPVariant(chrom="1", position=1000, reference_base="A", alternate_base="G")
        >>> hsnp_info = HSNPInfo(individual_code="SAMPLE1", variant=variant)
        >>> # 访问变异信息
        >>> print(hsnp_info.variant.position)
        1000
        >>> print(hsnp_info.variant.summary)
        1:1000:A>G
    """
    individual_code: str
    variant: HSNPVariant
    
    @classmethod
    def create(cls, individual_code: str, chrom: str, position: int, ref_base: str, alt_base: str, is_one_based: bool = False):
        """创建一个新的HSNPInfo实例
        
        Args:
            individual_code: 个体标识码
            chrom: 染色体标识
            position: 变异位置
            ref_base: 参考碱基
            alt_base: 变异碱基
            
        Returns:
            HSNPInfo: 新创建的HSNPInfo实例
        """
        variant = HSNPVariant(
            chrom=chrom,
            position=position,
            reference_base=ref_base,
            alternate_base=alt_base,
            is_one_based=is_one_based
        )
        return cls(
            individual_code=individual_code,
            variant=variant
        )
    
    def __str__(self) -> str:
        return f"{self.individual_code} - {self.variant.summary}"
    
    def to_region_str(self) -> str:
        return f"{self.variant.chrom}:{self.variant.position}"
