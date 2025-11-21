import pysam
from typing import List, Optional, Dict, Any
from enum import IntEnum, auto
import json
import re

# MATCH (1): Indicates that the bases in the read segment perfectly match the bases in the reference genome.
# INSERTION (2): Indicates that one or more bases are inserted in the read segment, which do not exist in the reference genome.
# DELETION (3): Indicates that one or more bases are missing in the read segment, which exist in the reference genome.
# MISMATCH (4): Indicates that the bases in the read segment do not match the bases in the reference genome.
# SOFT_CLIPPED (5): Indicates that this part of the read segment's bases did not align with the reference genome, but the sequence information of these bases is still retained in the read segment.
# HARD_CLIPPED (6): Indicates that this part of the read segment's bases was completely clipped, and the sequence information of these bases does not exist in the read segment.
# UNMAPPED (7): Indicates that the entire read segment did not map to the reference genome.
# UNKNOWN (8): Indicates that the alignment status cannot be determined, usually used for handling exceptional cases.

class AlignState(IntEnum):
    MATCH = auto()
    INSERTION = auto()
    DELETION = auto()
    MISMATCH = auto()
    SOFT_CLIPPED = auto()
    HARD_CLIPPED = auto()
    UNMAPPED = auto()
    UNKNOWN = auto()
    INTRON_START = auto()
    INTRON_END = auto() 

class StructuredAttributes:
    def __init__(self):
        self._attributes: Dict[str, Any] = {}

    def add_attribute(self, key: str, value: Any):
        if '.' in key:
            # Handle nested attributes
            parts = key.split('.')
            current = self._attributes
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            self._attributes[key] = value

    def get_attribute(self, key: str, default: Any = None) -> Any:
        if '.' in key:
            # Handle nested attributes
            parts = key.split('.')
            current = self._attributes
            for part in parts:
                if part not in current:
                    return default
                current = current[part]
            return current
        return self._attributes.get(key, default)

    def remove_attribute(self, key: str):
        if '.' in key:
            # Handle nested attributes
            parts = key.split('.')
            current = self._attributes
            for part in parts[:-1]:
                if part not in current:
                    return
                current = current[part]
            current.pop(parts[-1], None)
        else:
            self._attributes.pop(key, None)

    def to_dict(self) -> Dict[str, Any]:
        return self._attributes

    def to_json(self) -> str:
        return json.dumps(self._attributes)
    
    def __str__(self) -> str:
        return f"StructuredAttributes: {self._attributes}"

    @classmethod
    def from_json(cls, json_str: str) -> 'StructuredAttributes':
        instance = cls()
        instance._attributes = json.loads(json_str)
        return instance

    # New methods to make StructuredAttributes more dict-like
    def items(self):
        return self._attributes.items()

    def keys(self):
        return self._attributes.keys()

    def values(self):
        return self._attributes.values()

    def __getitem__(self, key):
        return self.get_attribute(key)

    def __setitem__(self, key, value):
        self.add_attribute(key, value)

    def __delitem__(self, key):
        self.remove_attribute(key)

    def __contains__(self, key):
        return key in self._attributes

    def __len__(self):
        return len(self._attributes)

    def __iter__(self):
        return iter(self._attributes)

class SequenceBase:
    def __init__(self, base: Optional[str], query_pos: Optional[int], ref_pos: Optional[int] = None, 
                 base_quality: Optional[int] = None, align_state: Optional[AlignState] = None, 
                 x: Optional[int] = None, y: Optional[float] = None, visual_x: Optional[int] = None):
        self.base = base
        self.query_pos = query_pos
        self.ref_pos = ref_pos
        self.base_quality = base_quality
        self.align_state = align_state
        self.x = x
        self.y = y
        self.visual_x = visual_x 
        self.attributes = StructuredAttributes()

    def __str__(self):
        base_info = f"Base: {self.base}, Query Position: {self.query_pos}, Reference Position: {self.ref_pos}, X: {self.x}, Y: {self.y}, Alignment State: {self.align_state.name if self.align_state else None}, Base Quality: {self.base_quality}"
        attr_info = ", ".join(f"{k}: {v}" for k, v in self.attributes.items())
        return f"{base_info}, Additional Attributes: {{{attr_info}}}"

    def add_attribute(self, key: str, value: Any):
        """Add a new attribute or update an existing one."""
        self.attributes.add_attribute(key, value)

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get the value of an attribute."""
        return self.attributes.get_attribute(key, default)

    def remove_attribute(self, key: str):
        """Remove an attribute if it exists."""
        self.attributes.remove_attribute(key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base": self.base,
            "query_pos": self.query_pos,
            "ref_pos": self.ref_pos,
            "base_quality": self.base_quality,
            "align_state": self.align_state,
            "x": self.x,
            "y": self.y,
            "visual_x": self.visual_x,
            "attributes": self.attributes.to_dict()
        }

class EnhancedRead:
    def __init__(self, aligned_segment: pysam.AlignedSegment, y_position: int = -1):
        self._aligned_segment = aligned_segment
        self._y_position = y_position
        self.sequence_bases: List[SequenceBase] = []
        self.attributes = StructuredAttributes()
        self._parse_read()

    def __getattr__(self, name):
        # Delegate attribute access to the wrapped AlignedSegment
        return getattr(self._aligned_segment, name)

    @property
    def y_position(self):
        return self._y_position

    @y_position.setter
    def y_position(self, value):
        self._y_position = value
        self._update_sequence_bases_y(value)

    def _update_sequence_bases_y(self, new_y):
        for base in self.sequence_bases:
            if base.align_state == AlignState.INSERTION:
                base.y = new_y + 1
            else:
                base.y = new_y

    def _parse_read(self):
        if self.query_sequence is None or self.is_unmapped:
            self._handle_unmapped_or_none_sequence()
            return
    
        try:
            mismatches = get_mismatch_positions(self._aligned_segment)
            # Initialize empty set for mismatches
            mismatch_ref_pos = set()
            
            # Only process mismatches if we have valid data
            if mismatches:
                for mismatch in mismatches:
                    # Check if mismatch tuple has all required elements
                    if len(mismatch) >= 3:
                        pos, _, _ = mismatch
                        mismatch_ref_pos.add(pos)
        except (KeyError, ValueError, TypeError):
            mismatch_ref_pos = set()
        
        self._process_aligned_sequence(mismatch_ref_pos)

    def _handle_unmapped_or_none_sequence(self):
        if self.query_sequence is None:
            self.sequence_bases = [SequenceBase(base=None, query_pos=None, align_state=AlignState.UNKNOWN, x=0, y=self.y_position)]
        else:
            self.sequence_bases = [
                SequenceBase(
                    base=base,
                    query_pos=i,
                    base_quality=self.query_qualities[i] if self.query_qualities else None,
                    align_state=AlignState.UNMAPPED,
                    x=i,
                    y=self.y_position
                ) for i, base in enumerate(self.query_sequence)
            ]

    def _process_aligned_sequence(self, mismatch_ref_pos):
        query_pos = 0
        ref_pos = self.reference_start
        x_pos = 0
        last_ref_pos = ref_pos
        insertion_offset = 0
        min_ref_pos = None
        min_query_pos = None
        bases_list = []
        offset = 0
        intron_gap = 0

        for operation, length in self.cigartuples:
            if operation == 0:  # Match or Mismatch
                for i in range(length):
                    base = SequenceBase(
                        base=self.query_sequence[query_pos],
                        query_pos=query_pos,
                        ref_pos=ref_pos,
                        base_quality=self.query_qualities[query_pos] if self.query_qualities else None,
                        align_state=AlignState.MISMATCH if ref_pos in mismatch_ref_pos else AlignState.MATCH,
                        x=x_pos,
                        y=self.y_position
                    )
                    if min_ref_pos is None or ref_pos < min_ref_pos:
                        min_ref_pos = ref_pos
                    if min_query_pos is None or query_pos < min_query_pos:
                        min_query_pos = query_pos
                    query_pos += 1
                    ref_pos += 1
                    x_pos += 1
                    last_ref_pos = ref_pos - 1
                    insertion_offset = 0
                    base.visual_x = base.x + offset + intron_gap
                    bases_list.append(base)
            elif operation == 1:  # Insertion
                for i in range(length):
                    base = SequenceBase(
                        base=self.query_sequence[query_pos],
                        query_pos=query_pos,
                        ref_pos=last_ref_pos,
                        base_quality=self.query_qualities[query_pos] if self.query_qualities else None,
                        align_state=AlignState.INSERTION,
                        x=x_pos - 1 + i,
                        y=self.y_position + 0.5
                    )
                    if min_query_pos is None or query_pos < min_query_pos:
                        min_query_pos = query_pos
                    query_pos += 1
                    insertion_offset += 1
                    base.visual_x = base.x + offset + intron_gap
                    bases_list.append(base)
            elif operation == 2:  # Deletion
                for i in range(length):
                    base = SequenceBase(
                        base='-',
                        query_pos=None,
                        ref_pos=ref_pos,
                        align_state=AlignState.DELETION,
                        x=x_pos,
                        y=self.y_position
                    )
                    if min_ref_pos is None or ref_pos < min_ref_pos:
                        min_ref_pos = ref_pos
                    ref_pos += 1
                    x_pos += 1
                    last_ref_pos = ref_pos - 1
                    insertion_offset = 0
                    base.visual_x = base.x + offset + intron_gap
                    bases_list.append(base)
            elif operation == 3:  # N (Skipped region from the reference)
                start_base = SequenceBase(
                    base='N',
                    query_pos=None,
                    ref_pos=ref_pos,
                    align_state=AlignState.INTRON_START,
                    x=x_pos,
                    y=self.y_position,
                    visual_x=x_pos  # 暂时只设置x值，后续会更新visual_x
                )
                bases_list.append(start_base)

                ref_pos += length
                x_pos += 1

                end_base = SequenceBase(
                    base='N',
                    query_pos=None,
                    ref_pos=ref_pos - 1,
                    align_state=AlignState.INTRON_END,
                    x=x_pos,
                    y=self.y_position,
                    visual_x=x_pos  # 暂时只设置x值，后续会更新visual_x
                )
                bases_list.append(end_base)

                x_pos += 1
                last_ref_pos = ref_pos - 1
                insertion_offset = 0

                if min_ref_pos is None or ref_pos < min_ref_pos:
                    min_ref_pos = ref_pos

                intron_gap += end_base.ref_pos - start_base.ref_pos - 1
            elif operation == 4:  # Soft clipping
                if x_pos == 0:  # Left soft-clipping
                    soft_clip_ref_pos = ref_pos - length
                else:  # Right soft-clipping
                    soft_clip_ref_pos = last_ref_pos + 1

                for i in range(length):
                    base = SequenceBase(
                        base=self.query_sequence[query_pos],
                        query_pos=query_pos,
                        ref_pos=soft_clip_ref_pos + i,
                        base_quality=self.query_qualities[query_pos] if self.query_qualities else None,
                        align_state=AlignState.SOFT_CLIPPED,
                        x=x_pos,
                        y=self.y_position
                    )
                    if min_query_pos is None or query_pos < min_query_pos:
                        min_query_pos = query_pos
                    query_pos += 1
                    x_pos += 1
                    insertion_offset = 0
                    base.visual_x = base.x + offset + intron_gap
                    bases_list.append(base)
            elif operation == 5:  # Hard clipping
                for i in range(length):
                    base = SequenceBase(
                        base='N',
                        query_pos=None,
                        ref_pos=None,
                        align_state=AlignState.HARD_CLIPPED,
                        x=x_pos,
                        y=self.y_position
                    )
                    x_pos += 1
                    insertion_offset = 0
                    base.visual_x = base.x + offset + intron_gap
                    bases_list.append(base)
            else:
                raise ValueError(f"Unknown CIGAR operation: {operation}")

        # Second pass to calculate and set visual_x for all bases
        # Split bases into segments between INTRON_START and INTRON_END
        segments = []
        current_segment = []
        intron_boundaries = []  # 存储所有的INTRON_START和INTRON_END基础
        in_intron = False
        
        for base in bases_list:
            if base.align_state == AlignState.INTRON_START:
                if current_segment:
                    segments.append(current_segment)
                current_segment = []
                in_intron = True
                intron_boundaries.append(base)  # 添加到内含子边界列表
            elif base.align_state == AlignState.INTRON_END:
                in_intron = False
                intron_boundaries.append(base)  # 添加到内含子边界列表
            elif not in_intron:
                current_segment.append(base)
        if current_segment:
            segments.append(current_segment)

        # Calculate visual_x for each segment separately
        for i, segment in enumerate(segments):
            first_base = next((base for base in segment if base.align_state != AlignState.HARD_CLIPPED), None)
            if first_base and first_base.ref_pos is not None:
                # Track insertions at each ref_pos to avoid overlapping
                insertion_counts = {}
                
                # 对于每个片段中的碱基，直接使用ref_pos作为visual_x
                for base in segment:
                    if base.ref_pos is not None:
                        # 如果是INSERTION碱基，则使用ref_pos+插入索引作为visual_x
                        if base.align_state == AlignState.INSERTION:
                            # Initialize counter for this reference position if not exists
                            if base.ref_pos not in insertion_counts:
                                insertion_counts[base.ref_pos] = 0
                            # Increment counter and use it for positioning
                            insertion_counts[base.ref_pos] += 1
                            base.visual_x = base.ref_pos + insertion_counts[base.ref_pos]
                        else:
                            base.visual_x = base.ref_pos
                    else:
                        # 对于没有ref_pos的碱基（如硬裁剪），使用相对位置
                        base.visual_x = first_base.ref_pos + (base.x - first_base.x)
                
                # 处理内含子边界
                if i > 0 and len(intron_boundaries) >= 2*i:
                    # 获取对应的内含子开始和结束碱基
                    intron_start = intron_boundaries[2*i-2]
                    intron_end = intron_boundaries[2*i-1]
                    
                    # 使用相邻片段的碱基来计算内含子边界的visual_x
                    prev_segment = segments[i-1]
                    if prev_segment:
                        last_base = prev_segment[-1]
                        # 内含子开始碱基的visual_x应该在上一个片段最后一个碱基之后
                        intron_start.visual_x = last_base.ref_pos + 1
                        # 内含子结束碱基的visual_x应该在当前片段第一个碱基之前
                        intron_end.visual_x = first_base.ref_pos - 1
        
        self.sequence_bases = bases_list

    def get_base_at_query_position(self, query_pos: int) -> Optional[SequenceBase]:
        return next((base for base in self.sequence_bases if base.query_pos == query_pos), None)

    def get_base_at_reference_position(self, ref_pos: int) -> Optional[SequenceBase]:
        return next((base for base in self.sequence_bases if base.ref_pos == ref_pos), None)

    def get_bases_by_align_state(self, align_state: AlignState) -> List[SequenceBase]:
        return [base for base in self.sequence_bases if base.align_state == align_state]

    def add_attribute(self, key: str, value: Any):
        self.attributes.add_attribute(key, value)

    def get_attribute(self, key: str, default: Any = None) -> Any:
        return self.attributes.get_attribute(key, default)

    def remove_attribute(self, key: str):
        self.attributes.remove_attribute(key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_name": self.query_name,
            "reference_start": self.reference_start,
            "reference_end": self.reference_end,
            "attributes": self.attributes.to_dict()
        }

    def __str__(self):
        read_info = f"Read Name: {self.query_name}, Y Position: {self.y_position}, Number of Bases: {len(self.sequence_bases)}"
        attr_info = ", ".join(f"{k}: {v}" for k, v in self.attributes.items())
        return f"{read_info}, Additional Attributes: {{{attr_info}}}"

def create_enhanced_read(y_position: int, additional_info: Dict[str, Any] = None) -> EnhancedRead:
    enhanced_read = EnhancedRead()
    enhanced_read.y_position = y_position
    
    if additional_info:
        for key, value in additional_info.items():
            enhanced_read.add_attribute(key, value)
    
    return enhanced_read

def get_mismatch_positions(read):
    """
    Get the positions of mismatches from a read alignment using MD tag.
    
    Args:
        read: A pysam.AlignedSegment object
        
    Returns:
        list: List of tuples containing (reference_position, reference_base, read_base)
    """
    
    if not read.has_tag('MD'):
        return []
    
    md = read.get_tag('MD')
    mismatches = []
    
    # Current position in the read
    read_pos = 0
    # Current position relative to the alignment start
    ref_pos = 0
    
    # Split MD string into components
    md_parts = re.findall(r'(\d+)|([A-Z]|\^[A-Z]+)', md)
    
    # Get read sequence
    read_seq = read.query_sequence
    
    # Process CIGAR string to handle insertions and deletions
    cigar_tuples = read.cigartuples
    cigar_pos = 0
    current_cigar_op = 0
    current_cigar_len = 0
    
    for part in md_parts:
        match_len, mismatch = part
        
        if match_len:  # Handle matching bases
            length = int(match_len)
            ref_pos += length
            read_pos += length
            
        elif mismatch:  # Handle mismatches or deletions
            if mismatch.startswith('^'):  # Deletion
                deleted_bases = mismatch[1:]
                ref_pos += len(deleted_bases)
            else:  # Single base mismatch
                # Calculate absolute reference position
                abs_ref_pos = read.reference_start + ref_pos
                
                # Skip positions affected by insertions
                while cigar_pos < len(cigar_tuples):
                    op, length = cigar_tuples[cigar_pos]
                    # Match or mismatch
                    if op == 0 or op == 7 or op == 8:
                        if read_pos < current_cigar_len + length:
                            break
                    # Insertion to reference
                    elif op == 1:
                        read_pos += length
                    # Deletion from reference
                    elif op == 2:
                        ref_pos += length
                    current_cigar_len += length
                    cigar_pos += 1
                
                if read_pos < len(read_seq):
                    mismatches.append((
                        abs_ref_pos,
                        mismatch,  # reference base
                        read_seq[read_pos]  # read base
                    ))
                read_pos += 1
                ref_pos += 1
    
    return mismatches
