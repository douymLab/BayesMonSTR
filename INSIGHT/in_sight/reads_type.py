from dataclasses import dataclass
from typing import Optional
import pysam
from enum import Flag, auto
from .reads_import import EnhancedRead

class ReadType(Flag):
    NONE = 0
    LOCALPAIRED = auto()
    INSERTION = auto()
    SINGLE = auto()
    SHARED_Y = auto()

    @classmethod
    def add_type(cls, name):
        """
        Dynamically add a new read type.
        Usage: ReadType.add_type('NEW_TYPE')
        """
        new_type = cls._next_value_()
        setattr(cls, name, new_type)
        return new_type


@dataclass
class ReadInfo(EnhancedRead):
    query_name: str
    start: int
    end: int
    read_type: ReadType
    mate_start: Optional[int] = None
    mate_end: Optional[int] = None

    def __init__(
        self,
        aligned_segment: pysam.AlignedSegment,
        y_position: int,
        read_type: ReadType = ReadType.NONE,
    ):
        super().__init__(aligned_segment, y_position)
        self.query_name = aligned_segment.query_name
        self.start = aligned_segment.reference_start
        self.end = (
            aligned_segment.reference_end
            or aligned_segment.reference_start + aligned_segment.query_length
        )
        self.read_type = read_type
        self.mate_start = aligned_segment.next_reference_start
        self.mate_end = (
            self.mate_start + aligned_segment.query_length
            if self.mate_start is not None
            else None
        )

    def add_type(self, new_type):
        self.read_type |= new_type

    def remove_type(self, type_to_remove):
        self.read_type &= ~type_to_remove

    def has_type(self, type_to_check):
        return self.read_type & type_to_check == type_to_check
