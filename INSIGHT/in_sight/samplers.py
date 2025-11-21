from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pysam
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class SamplingResult:
    """存储采样结果的数据类"""
    selected_indices: List[int]
    stats: Dict[str, Any]

class BaseSampler(ABC):
    """Sampler的基础接口类"""
    def __init__(self, target_count: int, random_seed: Optional[int] = None):
        self.target_count = target_count
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    @abstractmethod
    def sample(self, candidates: List[Dict[str, Any]]) -> SamplingResult:
        """
        从候选reads中进行采样
        
        Args:
            candidates: 包含read信息的字典列表，每个字典至少包含：
                - segment: pysam.AlignedSegment对象
                - is_high_quality: bool值表示是否为高质量read
        
        Returns:
            SamplingResult: 包含被选中的indices和相关统计信息
        """
        pass

class UniformSampler(BaseSampler):
    """均匀采样策略"""
    def sample(self, candidates: List[Dict[str, Any]]) -> SamplingResult:
        if len(candidates) <= self.target_count:
            return SamplingResult(
                selected_indices=list(range(len(candidates))),
                stats={"total": len(candidates), "sampled": len(candidates)}
            )
        
        indices = np.round(np.linspace(0, len(candidates) - 1, self.target_count)).astype(int)
        return SamplingResult(
            selected_indices=indices.tolist(),
            stats={"total": len(candidates), "sampled": len(indices)}
        )

class MutationBasedSampler(BaseSampler):
    """基于突变位点的采样策略"""
    def __init__(self, target_count: int, chrom: str, pos: int, ref_base: str,
                 min_base_quality: int = 20, random_seed: Optional[int] = None):
        super().__init__(target_count, random_seed)
        self.chrom = chrom
        self.pos = pos  # 1-based position
        self.ref_base = ref_base.upper()
        self.min_base_quality = min_base_quality
        
    def sample(self, candidates: List[Dict[str, Any]]) -> SamplingResult:
        # 如果候选数量小于目标数量，直接返回所有候选项
        if len(candidates) <= self.target_count:
            return SamplingResult(
                selected_indices=list(range(len(candidates))),
                stats={"total": len(candidates), "sampled": len(candidates)}
            )
        
        base_groups = defaultdict(list)
        stats = {
            "total": len(candidates),
            "base_counts": defaultdict(int),
            "filtered_low_quality": 0
        }
        
        # 按碱基分组
        for idx, candidate in enumerate(candidates):
            read = candidate['segment']
            try:
                # 获取read在参考基因组上的位置
                read_pos = read.get_reference_positions()
                query_index = read_pos.index(self.pos - 1)  # 转换为0-based
                
                # 获取碱基和质量
                base = read.query_sequence[query_index].upper()
                base_quality = read.query_qualities[query_index]
                
                if base_quality >= self.min_base_quality:
                    base_groups[base].append(idx)
                    stats["base_counts"][base] += 1
                else:
                    stats["filtered_low_quality"] += 1
                    
            except (ValueError, IndexError):
                continue
        
        # 计算每个碱基组应该选择的数量
        total_valid_reads = sum(len(indices) for indices in base_groups.values())
        if total_valid_reads == 0:
            return SamplingResult(selected_indices=[], stats=stats)
        
        selected_indices = []
        remaining_count = self.target_count
        
        # 确保每个碱基组至少有一定数量的reads
        min_per_base = max(1, self.target_count // (len(base_groups) * 2))
        
        # 首先为每个碱基组分配最小数量
        for base, indices in base_groups.items():
            count = min(min_per_base, len(indices))
            if count > 0:
                selected = np.random.choice(indices, size=count, replace=False)
                selected_indices.extend(selected)
                remaining_count -= count
        
        # 按比例分配剩余的名额
        if remaining_count > 0:
            remaining_indices = [idx for base, indices in base_groups.items() 
                              for idx in indices if idx not in selected_indices]
            if remaining_indices:
                additional = np.random.choice(
                    remaining_indices,
                    size=min(remaining_count, len(remaining_indices)),
                    replace=False
                )
                selected_indices.extend(additional)
        
        stats["sampled"] = len(selected_indices)
        return SamplingResult(selected_indices=selected_indices, stats=stats)