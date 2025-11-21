import pysam
import networkx as nx
import pandas as pd
from typing import List, Dict, Callable, Tuple, Any, Optional, Union, Iterable
from dataclasses import dataclass, field
from collections import defaultdict
from .reads_import import EnhancedRead, AlignState
from .reads_type import ReadInfo, ReadType
from .utils import adjust_chromosome_name
from .samplers import BaseSampler, UniformSampler, MutationBasedSampler
from functools import cmp_to_key
from pathlib import Path
import random
import numpy as np
import traceback

class Region:
    def __init__(self, start: int, end: int, region_type: str):
        self.start = start
        self.end = end
        self.region_type = region_type

class CombinedReadInfo(ReadInfo, EnhancedRead):
    def __init__(self, aligned_segment: pysam.AlignedSegment, read_type: ReadType = ReadType.NONE):
        ReadInfo.__init__(self, aligned_segment, 0, read_type)
        EnhancedRead.__init__(self, aligned_segment, 0)
        self.y_level: Optional[YLevel] = None
        self._y_position: float = 0.0
        self._height: int = self._calculate_height()

    @property
    def y_position(self) -> float:
        return self._y_position

    def update_y_position(self, y_value: float):
        self._y_position = y_value
        self._update_sequence_bases_y(y_value)

    def _update_sequence_bases_y(self, y_value: float):
        for base in self.sequence_bases:
            if base.align_state == AlignState.INSERTION:
                base.y = y_value + 1
            else:
                base.y = y_value

    @property
    def height(self) -> int:
        return self._height

    def _calculate_height(self) -> int:
        return 2 if any(base.align_state == AlignState.INSERTION for base in self.sequence_bases) else 1

    def recalculate_height(self):
        self._height = self._calculate_height()

    @property
    def xmin(self) -> float:
        return min((base.visual_x for base in self.sequence_bases if base.visual_x is not None), default=float('inf'))

    @property
    def xmax(self) -> float:
        return max((base.visual_x for base in self.sequence_bases if base.visual_x is not None), default=float('-inf'))
    
    def __hash__(self):
        return hash((self.query_name, self.reference_start))

    def __eq__(self, other):
        if not isinstance(other, CombinedReadInfo):
            return False
        return (self.query_name, self.reference_start) == (other.query_name, other.reference_start)

    def has_attribute(self, name: str) -> bool:
        return name in self.attributes

    def get_attribute(self, name: str) -> Any:
        return self.attributes.get_attribute(name)

    def add_attribute(self, name: str, value: Any):
        self.attributes[name] = value


class YLevel:
    def __init__(self, level_id: int):
        self.level_id = level_id
        self.reads: List[CombinedReadInfo] = []
        self._base_y_value: float = 0.0
        self.has_localpaired: bool = False
        self._height: int = 1
        self.aggregated_attributes = {}
        self.read_y_offsets: Dict[CombinedReadInfo, float] = {}

    def add_read(self, read: CombinedReadInfo, y_offset: float = 0.0):
        self.reads.append(read)
        read.y_level = self
        self.read_y_offsets[read] = y_offset
        if read.has_type(ReadType.LOCALPAIRED):
            self.has_localpaired = True
        
        # Calculate the height for this specific read
        read_height = 2 if any(base.align_state == AlignState.INSERTION for base in read.sequence_bases) else 1
        
        # Update the YLevel height based on the new read
        self._height = max(self._height, read_height + int(y_offset))
        
        self._update_aggregated_attributes(read)
        self._update_read_y_position(read)

    def _update_read_y_position(self, read: CombinedReadInfo):
        y_position = self._base_y_value + self.read_y_offsets[read]
        read.update_y_position(y_position)

    def _update_height(self):
        self._height = max(
            max(read.height + int(self.read_y_offsets[read]) for read in self.reads),
            max(int(offset) + 1 for offset in self.read_y_offsets.values())
        )

    @property
    def y_value(self) -> float:
        return self._base_y_value

    @y_value.setter
    def y_value(self, value: float):
        self._base_y_value = value
        for read in self.reads:
            self._update_read_y_position(read)

    def _update_aggregated_attributes(self, read: CombinedReadInfo):
        for attr, value in read.attributes.items():
            if isinstance(value, dict):
                for sub_attr, sub_value in value.items():
                    key = f"{attr}.{sub_attr}"
                    if key not in self.aggregated_attributes:
                        self.aggregated_attributes[key] = []
                    self.aggregated_attributes[key].append(sub_value)
            else:
                if attr not in self.aggregated_attributes:
                    self.aggregated_attributes[attr] = []
                self.aggregated_attributes[attr].append(value)

    def get_attribute(self, attr_path: str, aggregation_func: Callable = None):
        attrs = attr_path.split('.')
        if len(attrs) == 1:
            values = self.aggregated_attributes.get(attr_path, [])
        else:
            values = [self._get_nested_attribute(read, attrs) for read in self.reads]
            values = [v for v in values if v is not None]
        
        if not values:
            return None
        
        if aggregation_func:
            return aggregation_func(values)
        return values

    @staticmethod
    def _get_nested_attribute(obj: Any, attrs: list) -> Any:
        for attr in attrs:
            if hasattr(obj, 'get_attribute'):
                obj = obj.get_attribute(attr)
            elif isinstance(obj, dict):
                obj = obj.get(attr)
            else:
                obj = getattr(obj, attr, None)
            if obj is None:
                return None
        return obj

    @property
    def height(self) -> int:
        return self._height

    def _update_height(self):
        max_read_height = max(read.height for read in self.reads) if self.reads else 1
        max_y_offset = max(self.read_y_offsets.values()) if self.read_y_offsets else 0
        self._height = max(max_read_height, int(max_y_offset) + 1)

    @property
    def xmin(self):
        return min(read.xmin for read in self.reads) if self.reads else float('inf')

    @property
    def xmax(self):
        return max(read.xmax for read in self.reads) if self.reads else float('-inf')

class LayoutManager:
    def __init__(self):
        self.y_levels: List[YLevel] = []
        self.reads_graph = nx.Graph()

    def process_reads(self, reads: List[CombinedReadInfo]):
        paired_reads, unpaired_reads = self._separate_reads(reads)
        self._process_paired_reads(paired_reads)
        self._process_unpaired_reads(unpaired_reads)
        self._assign_y_values()

    def _separate_reads(self, reads: List[CombinedReadInfo]) -> Tuple[List[CombinedReadInfo], List[CombinedReadInfo]]:
        paired_reads = []
        unpaired_reads = []
        for read in reads:
            if read.has_type(ReadType.LOCALPAIRED):
                paired_reads.append(read)
            else:
                unpaired_reads.append(read)
        return paired_reads, unpaired_reads

    def _process_paired_reads(self, paired_reads: List[CombinedReadInfo]):
        paired_reads_dict = defaultdict(list)
        for read in paired_reads:
            paired_reads_dict[read.query_name].append(read)
    
        for query_name, reads in paired_reads_dict.items():
            if len(reads) >= 2:
                y_level = YLevel(len(self.y_levels))

                # 按照reference_start排序reads
                sorted_reads = sorted(reads, key=lambda r: r.reference_start)

                # 初始化y偏移量
                current_y_offset = 0.0
                
                for i, read in enumerate(sorted_reads):
                    # 检查当前read是否与之前的reads有重叠
                    overlaps_with_insertion = False
                    for prev_read in sorted_reads[:i]:
                        if self._reads_overlap(read, prev_read):
                            if (any(base.align_state == AlignState.INSERTION for base in read.sequence_bases) and 
                                self._insertion_overlaps_read(read, prev_read)):
                                overlaps_with_insertion = True
                            if (any(base.align_state == AlignState.INSERTION for base in prev_read.sequence_bases) and 
                                self._insertion_overlaps_read(prev_read, read)):
                                overlaps_with_insertion = True
                    
                    # 根据重叠情况决定y偏移量
                    if i > 0 and (overlaps_with_insertion or self._reads_overlap(read, sorted_reads[i-1])):
                        has_insertion = any(base.align_state == AlignState.INSERTION for base in read.sequence_bases)
                        current_y_offset += 3.0 if overlaps_with_insertion else (2.0 if has_insertion else 1.0)
                    
                    y_level.add_read(read, current_y_offset)
                
                self.y_levels.append(y_level)

    def _insertion_overlaps_read(self, read_with_insertion: CombinedReadInfo, other_read: CombinedReadInfo) -> bool:
        min_ref_pos = read_with_insertion.xmin
        insertion_positions = [base.x + min_ref_pos for base in read_with_insertion.sequence_bases if base.align_state == AlignState.INSERTION and base.ref_pos is not None]
        return any(other_read.xmin <= pos <= other_read.xmax for pos in insertion_positions)

    @staticmethod
    def _reads_overlap(read1: CombinedReadInfo, read2: CombinedReadInfo) -> bool:
        if read1.xmax is None or read1.xmin is None or read2.xmax is None or read2.xmin is None:
            return False
        return not (read1.xmax <= read2.xmin or read2.xmax <= read1.xmin)

    def _process_unpaired_reads(self, unpaired_reads: List[CombinedReadInfo]):
        self._build_reads_graph(unpaired_reads)
        layout = self._graph_layout()
        self._assign_layout(layout)

    def _build_reads_graph(self, reads: List[CombinedReadInfo]):
        self.reads_graph.clear()
        for read in reads:
            self.reads_graph.add_node(read)

        for i, read1 in enumerate(reads):
            for read2 in reads[i+1:]:
                if self._reads_overlap(read1, read2):
                    self.reads_graph.add_edge(read1, read2)

    def _graph_layout(self) -> Dict[CombinedReadInfo, int]:
        return nx.greedy_color(self.reads_graph)

    def _assign_layout(self, layout: Dict[CombinedReadInfo, int]):
        max_level_id = max(yl.level_id for yl in self.y_levels) if self.y_levels else -1
        new_y_levels = defaultdict(lambda: YLevel(0))

        for read, color in layout.items():
            level_id = max_level_id + 1 + color
            new_y_levels[level_id].level_id = level_id
            new_y_levels[level_id].add_read(read)
            new_y_levels[level_id]._height = max(new_y_levels[level_id]._height, read.height)  # 确保高度正确

        self.y_levels.extend(new_y_levels.values())

    def _assign_y_values(self):
        current_y = 0.0
        for y_level in self.y_levels:
            y_level.y_value = current_y
            current_y += y_level.height

    def compact_layout(self):
        all_reads = [read for y_level in self.y_levels for read in y_level.reads]
        self.y_levels.clear()
        self.process_reads(all_reads)

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


class CombinedReadManager:
    def __init__(self, alignment_filename: str, chromosome: str, regions: List[Region], reference_filename: str = None, 
                 filter_tags: Dict[str, Union[str, Iterable]] = None, 
                 sampler: BaseSampler = None):
        self.alignment_filename = alignment_filename
        self.chromosome = chromosome
        self.reference_filename = reference_filename
        self.regions = sorted(regions, key=lambda r: r.start)
        self.reads: Dict[str, CombinedReadInfo] = {}
        self.processors: List[Callable[[CombinedReadInfo], None]] = []
        self.localpaired_reads: Dict[str, List[CombinedReadInfo]] = {}
        self.min_region_start = min(region.start for region in regions)
        self.max_region_end = max(region.end for region in regions)
        self.layout_manager = LayoutManager()
        self.sampler = sampler
        self.sampler_info = None  # 添加存储采样统计信息的属性
        
        self.filter_tags = {}
        if filter_tags:
            for tag, values in filter_tags.items():
                if isinstance(values, str) or not isinstance(values, Iterable):
                    values = {str(values)}
                else:
                    values = set(map(str, values))
                self.filter_tags[tag] = values
        self.skipped_reads = 0

    def process_reads(self):
        self._load_reads()
        self._remove_single_localpaired_reads()
        self.layout_manager.process_reads(list(self.reads.values()))
        self._apply_processors()

    def _load_reads(self):
        file_type = determine_file_type(self.alignment_filename)
        if file_type == "BAM":
            mode = "rb"
        elif file_type == "CRAM":
            mode = "rc"
            if self.reference_filename is None:
                print("Warning: Reference filename is required for CRAM files.")
        elif file_type == "SAM":
            mode = "r"
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # 是否需要降采样
        if self.sampler is not None:
            self._load_with_downsampling(mode)
        else:
            self._load_without_downsampling(mode)

    def _create_read_info(self, aligned_segment: pysam.AlignedSegment) -> CombinedReadInfo:
        read_type = self._determine_read_type(aligned_segment)
        return CombinedReadInfo(aligned_segment, read_type)

    def _determine_read_type(self, aligned_segment: pysam.AlignedSegment) -> ReadType:
        read_type = ReadType.NONE
        if aligned_segment.is_paired and aligned_segment.is_proper_pair:
            mate_start = aligned_segment.next_reference_start
            mate_end = mate_start + aligned_segment.query_length if mate_start is not None else None
            if (self._is_in_region(aligned_segment.reference_start, aligned_segment.reference_end) and
                self._is_in_region(mate_start, mate_end)):
                read_type |= ReadType.LOCALPAIRED
        return read_type

    def _is_in_region(self, start, end):
        return (
            (start is not None and self.min_region_start <= start <= self.max_region_end) or
            (end is not None and self.min_region_start <= end <= self.max_region_end) or
            (start is not None and end is not None and start <= self.min_region_start and end >= self.max_region_end)
        )

    def _remove_single_localpaired_reads(self):
        for read in list(self.reads.values()):
            if read.has_type(ReadType.LOCALPAIRED) and len(self.localpaired_reads[read.query_name]) < 2:
                read.remove_type(ReadType.LOCALPAIRED)
                self.localpaired_reads[read.query_name].remove(read)
                if not self.localpaired_reads[read.query_name]:
                    del self.localpaired_reads[read.query_name]

    def _assign_initial_layout(self):
        sorted_reads = sorted(self.reads.values(), key=lambda r: (r.reference_start, -r.reference_end))
        for read in sorted_reads:
            self.layout_manager.add_read(read)

    def _handle_localpaired_reads(self):
        self.layout_manager.handle_localpaired_reads()

    def _apply_processors(self):
        for read in self.reads.values():
            for processor in self.processors:
                try:
                    processor(read)
                except Exception as e:
                    print(f"Error in processor {processor.__name__ if hasattr(processor, '__name__') else processor}:")
                    print(f"  Read: {read.query_name}_{read.reference_start}")
                    print(f"  Error: {type(e).__name__}: {str(e)}")
                    print("  Traceback:")
                    traceback.print_exc()
                    # You can raise the exception again if you want to stop processing
                    # raise

    def add_processor(self, processor: Callable[[CombinedReadInfo], None]):
        self.processors.append(processor)

    def clear_processors(self):
        self.processors = []

    def _get_sort_key(self, y_level: YLevel, key: str) -> Any:
        if key == 'start':
            return y_level.xmin
        # if key == 'reference_start':
            
        #     return y_level.reads[0].reference_start
        elif key == 'end':
            return y_level.xmax
        elif key == 'has_localpaired':
            return y_level.has_localpaired
        elif key.startswith('has_type_'):
            read_type = getattr(ReadType, key[9:], None)
            return any(read.has_type(read_type) for read in y_level.reads) if read_type else None
        elif key.startswith('max_'):
            attr = key[4:]
            return y_level.get_attribute(attr, max)
        elif key.startswith('min_'):
            attr = key[4:]
            return y_level.get_attribute(attr, min)
        elif key.startswith('avg_'):
            attr = key[4:]
            return y_level.get_attribute(attr, lambda x: sum(x) / len(x) if x else None)
        else:
            return y_level.get_attribute(key, lambda x: x[0] if x else None)

    def sort_reads(self, *keys, reverse=False):
        def compare_y_levels(y_level1, y_level2):
            for key in keys:
                val1 = self._get_sort_key(y_level1, key)
                val2 = self._get_sort_key(y_level2, key)
                if val1 is None and val2 is None:
                    continue
                if val1 is None:
                    return 1
                if val2 is None:
                    return -1
                if val1 != val2:
                    return (val1 > val2) - (val1 < val2)
            return 0

        self.layout_manager.y_levels.sort(key=cmp_to_key(compare_y_levels), reverse=reverse)
        self.layout_manager._assign_y_values()

    def compact_layout(self):
        self.layout_manager.compact_layout()

    def generate_dataframes(self, padding: int = 10, ref_start: int = None, ref_end: int = None):
        """
        Generate dataframes
        
        Parameters:
            padding: int - Number of positions to extend on both sides when auto-calculating reference range
            ref_start: int - Optional, manually specify reference start position
            ref_end: int - Optional, manually specify reference end position
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] - (read_df, base_df, reference_df)
        """
        read_data = [self._extract_read_info(read) for read in self.reads.values()]
        base_data = [base_info for read in self.reads.values() for base_info in self._extract_base_info(read)]
        
        # Create empty dataframes
        read_df = pd.DataFrame(read_data) if read_data else pd.DataFrame()
        base_df = pd.DataFrame(base_data) if base_data else pd.DataFrame()
        
        # Generate reference sequence dataframe
        reference_df = self._generate_reference_dataframe(padding=padding, start=ref_start, end=ref_end, 
                                                         empty_reads=(not read_data or not base_data))
        
        return read_df, base_df, reference_df
    
    def _generate_reference_dataframe(self, padding: int = 10, start: int = None, end: int = None, empty_reads: bool = False):
        """
        Generate reference sequence DataFrame
        
        Parameters:
            padding: int - Number of positions to extend on both sides when auto-calculating range
            start: int - Optional, manually specify start position, ignores auto-calculation and padding if specified
            end: int - Optional, manually specify end position, ignores auto-calculation and padding if specified
            empty_reads: bool - Indicates if read_data or base_data is empty
        
        Returns:
            pandas.DataFrame - Reference sequence DataFrame
        """
        # Determine display range
        if start is not None and end is not None:
            # Use manually specified range
            display_range = (start, end)
        else:
            # Auto-calculate range
            if not self.reads:
                if not empty_reads or (start is None and end is None):
                    # No data available to infer reference range
                    if start is not None:
                        display_range = (start, start + 100)  # Default show 100 positions
                    elif end is not None:
                        display_range = (max(0, end - 100), end)  # Default show 100 positions
                    else:
                        return pd.DataFrame()  # Not enough information to generate reference sequence
                else:
                    # Use known values from start and end to determine range
                    if start is None:
                        start = max(0, end - 100)
                    if end is None:
                        end = start + 100
                    display_range = (start, end)
            else:
                # Have reads data, calculate range normally
                min_pos = min(read.xmin for read in self.reads.values() if read.xmin is not None)
                max_pos = max(read.xmax for read in self.reads.values() if read.xmax is not None)
                
                # Check if there's a valid position range
                if min_pos == float('inf') or max_pos == float('-inf'):
                    if start is not None or end is not None:
                        # Use known values from start and end
                        if start is None:
                            start = max(0, end - 100)
                        if end is None:
                            end = start + 100
                        display_range = (start, end)
                    else:
                        return pd.DataFrame()  # Not enough information to generate reference sequence
                else:
                    # Add padding (extension space)
                    min_pos = max(0, int(min_pos) - padding)  # Ensure not less than 0
                    max_pos = int(max_pos) + padding
                    
                    # If start or end was manually specified, apply those values
                    if start is not None:
                        min_pos = start
                    if end is not None:
                        max_pos = end
                    
                    display_range = (min_pos, max_pos)
        
        # Calculate sequence length
        seq_len = display_range[1] - display_range[0] + 1
        
        # Get reference sequence or create N-filled sequence
        if self.reference_filename:
            try:
                with pysam.FastaFile(self.reference_filename) as ref:
                    # Adjust the chromosome name if needed
                    chrom = adjust_chromosome_name(self.reference_filename, self.chromosome)
                    # Chromosome name format can be adjusted here if needed
                    reference_sequence = ref.fetch(chrom, display_range[0], display_range[1] + 1).upper()
            except Exception as e:
                print(f"Unable to retrieve reference sequence: {e}")
                # Create N-filled sequence
                reference_sequence = 'N' * seq_len
        else:
            # No reference sequence file provided, use N-filling
            print("Warning: No reference filename provided. Using 'N' filled reference sequence.")
            reference_sequence = 'N' * seq_len
        
        # Create reference sequence DataFrame
        reference_df = pd.DataFrame({
            'read_id': ['reference_read'] * seq_len,
            'base': list(reference_sequence),
            'query_pos': range(1, seq_len + 1),
            'ref_pos': range(display_range[0], display_range[1] + 1),
            'base_quality': [60] * seq_len,  # Default quality 60 for reference sequence
            'align_state_value': [0] * seq_len,
            'align_state': ['REF'] * seq_len,
            'x': range(1, seq_len + 1),
            'y': [-2] * seq_len,  # Reference sequence at y=-2
            'visual_x': range(display_range[0], display_range[1] + 1),
        })
        
        return reference_df

    def _extract_read_info(self, read: CombinedReadInfo) -> dict:
        read_info = {
            'read_id': f"{read.query_name}_{read.reference_start}",
            'query_name': read.query_name,
            'reference_start': read.reference_start,
            'reference_end': read.reference_end,
            'y_position': read.y_position,
            'y_level': read.y_level.level_id if read.y_level else None,
            'y_offset': read.y_level.read_y_offsets[read] if read.y_level else 0.0,
            'is_paired': read.is_paired,
            'is_proper_pair': read.is_proper_pair,
            'is_reverse': read.is_reverse,
            'is_duplicate': read.is_duplicate,
            'is_secondary': read.is_secondary,
            'is_qcfail': read.is_qcfail,
            'mapping_quality': read.mapping_quality,
            'query_alignment_length': read.query_alignment_length,
            'read_type_value': read.read_type.value,
            'read_type': read.read_type.name,
            'is_localpaired_reads': read.has_type(ReadType.LOCALPAIRED),
            'height': read.height,
            'xmin': read.xmin,
            'xmax': read.xmax,
        }

        read_info.update({k: v for k, v in read.attributes.items() if k != 'sequence_bases'})

        return read_info

    def _extract_base_info(self, read: CombinedReadInfo) -> list:
        base_info = []
        for base in read.sequence_bases:
            base_dict = {
                'read_id': f"{read.query_name}_{read.reference_start}",
                'base': base.base,
                'query_pos': base.query_pos,
                'ref_pos': base.ref_pos,
                'base_quality': base.base_quality,
                'align_state_value': base.align_state.value,
                'align_state': base.align_state.name,
                'x': base.x,
                'y': base.y,
                'visual_x': base.visual_x
            }
            base_dict.update({k: v for k, v in base.attributes.items() if k != 'sequence_bases'})
            base_info.append(base_dict)
        return base_info

    def generate_y_level_dataframe(self):
        y_level_data = []
        for y_level in self.layout_manager.y_levels:
            y_level_info = {
                'y': y_level.y_value,
                'xmin': y_level.xmin,
                'xmax': y_level.xmax,
                'height': y_level.height,
                'has_localpaired': y_level.has_localpaired,
                'num_reads': len(y_level.reads),
            }
            y_level_data.append(y_level_info)

        return pd.DataFrame(y_level_data)

    def _load_with_downsampling(self, mode):
        """使用sampler进行降采样"""
        if self.sampler is None:
            raise ValueError("Sampler is required for downsampling")
            
        with pysam.AlignmentFile(self.alignment_filename, mode, reference_filename=self.reference_filename) as alignment_file:
            # 收集符合条件的reads信息
            all_candidates = []
            
            # 单次扫描收集所有候选
            for region in self.regions:
                for aligned_segment in alignment_file.fetch(self.chromosome, start=region.start, end=region.end):
                    # 前置过滤条件
                    if aligned_segment.is_unmapped or aligned_segment.cigartuples is None:
                        continue
                    
                    # 标签过滤
                    if self.filter_tags:
                        skip = False
                        for tag, allowed_values in self.filter_tags.items():
                            try:
                                actual_value = str(aligned_segment.get_tag(tag))
                                if actual_value not in allowed_values:
                                    self.skipped_reads += 1
                                    skip = True
                                    break
                            except KeyError:
                                self.skipped_reads += 1
                                skip = True
                                break
                        if skip:
                            continue
                    
                    # 记录候选信息
                    all_candidates.append({
                        'ref_pos': aligned_segment.reference_start,
                        'query_name': aligned_segment.query_name,
                        'ref_start': aligned_segment.reference_start,
                        'is_high_quality': aligned_segment.mapping_quality > 30,
                        'segment': aligned_segment
                    })
            
            # 使用sampler进行采样
            result = self.sampler.sample(all_candidates)
            
            # 保存采样统计信息
            self.sampler_info = result.stats
            
            # 处理采样结果
            for idx in result.selected_indices:
                candidate = all_candidates[idx]
                segment = candidate['segment']
                unique_key = f"{segment.query_name}_{segment.reference_start}"
                
                if unique_key not in self.reads:
                    read_info = self._create_read_info(segment)
                    self.reads[unique_key] = read_info
                    if read_info.has_type(ReadType.LOCALPAIRED):
                        self.localpaired_reads.setdefault(read_info.query_name, []).append(read_info)
            
            # 打印采样统计信息
            print(f"Sampling statistics:")
            for key, value in result.stats.items():
                print(f"  - {key}: {value}")
    
    def _load_without_downsampling(self, mode):
        """不使用降采样策略，加载所有符合条件的reads"""
        with pysam.AlignmentFile(self.alignment_filename, mode, reference_filename=self.reference_filename) as alignment_file:
            for region in self.regions:
                for aligned_segment in alignment_file.fetch(self.chromosome, start=region.start, end=region.end):
                    # 前置过滤条件
                    if aligned_segment.is_unmapped or aligned_segment.cigartuples is None:
                        continue
                    
                    # 标签过滤
                    if self.filter_tags:
                        skip = False
                        for tag, allowed_values in self.filter_tags.items():
                            try:
                                actual_value = str(aligned_segment.get_tag(tag))
                                if actual_value not in allowed_values:
                                    self.skipped_reads += 1
                                    skip = True
                                    break
                            except KeyError:
                                self.skipped_reads += 1
                                skip = True
                                break
                        if skip:
                            continue
                    
                    # 创建read_info的逻辑
                    unique_key = f"{aligned_segment.query_name}_{aligned_segment.reference_start}"
                    if unique_key not in self.reads:
                        read_info = self._create_read_info(aligned_segment)
                        self.reads[unique_key] = read_info
                        if read_info.has_type(ReadType.LOCALPAIRED):
                            self.localpaired_reads.setdefault(read_info.query_name, []).append(read_info)

def create_combined_read_manager(bam_file: str, chromosome: str, regions: List[Tuple[int, int, str]], reference_filename: str = None, 
                                 filter_tags: Dict[str, Union[str, Iterable]] = None, sampler: BaseSampler = None) -> CombinedReadManager:
    region_objects = [Region(start, end, region_type) for start, end, region_type in regions]
    return CombinedReadManager(bam_file, chromosome, region_objects, reference_filename, filter_tags, sampler)
