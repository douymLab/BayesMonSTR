from typing import Dict, Any, Callable, Tuple, List
from .reads_import import EnhancedRead
from .reads_manager import CombinedReadInfo, ReadType
from .utils import get_head_softclip, get_head_hardclip, HSNPInfo
from .str_utils import STRRegion
from .analysis.prob_pm import classify_allelic_reads

# processor for STR data
def str_data_processor(str_info_dict: Dict[str, Dict[str, Any]]):
    def process(read: CombinedReadInfo):
        read_key = f"{read.query_name}_{read.reference_start}"
        str_info = str_info_dict.get(read_key)
        if str_info:
            read.add_attribute('str_info', str_info)
            str_seq = str_info.get('STR_seq')
            str_position = str_info.get('STR_position') # the position of reads index

            if str_seq and str_position:
                str_start, str_end = str_position
                str_index = 0
                head_softclip, left_soft_clip_len = get_head_softclip(read._aligned_segment)
                # left_hardclip = get_head_hardclip(read._aligned_segment)
                # print(f"left_soft_clip_len: {left_soft_clip_len}")
                str_start += left_soft_clip_len
                str_end += left_soft_clip_len
                # str_start -= left_hardclip
                # str_end -= left_hardclip
                for base in read.sequence_bases:
                    if base.query_pos is not None and str_start <= base.query_pos <= str_end:
                        base.add_attribute('str.is_str', True)
                        if str_index < len(str_seq) and base.base == str_seq[str_index]:
                            base.add_attribute('str.position', str_index)
                            str_index += 1
                        else:
                            base.add_attribute('str.mismatch', True)
                    elif base.query_pos is not None and base.query_pos >= str_end:
                        break
    return process

# processor for STR data by str position
def str_data_processor_by_str_position(str_region_info: STRRegion):
    def process(read: CombinedReadInfo):
        # Filter reads that do not sufficiently span the STR region based on reference coordinates
        if not (
            read.reference_start is not None
            and read.reference_end is not None
            and read.reference_start <= str_region_info.start - 12  # region_start - 11 - 1
            and read.reference_end >= str_region_info.end + 10
        ):
            return  # Skip processing this read if it doesn't meet the criteria
        str_index = 0
        str_start, str_end = str_region_info.start, str_region_info.end
        for base in read.sequence_bases:
            if base.query_pos is not None and str_start <= base.query_pos <= str_end:
                base.add_attribute('str.is_str', True)
                base.add_attribute('str.position', str_index)
                str_index += 1
            elif base.query_pos is not None and base.query_pos >= str_end:
                break
    return process

# processor for HSNP data
def hsnp_processor(hsnp_info: HSNPInfo, target_region: Tuple[int, int]):
    paired_reads: Dict[str, List[CombinedReadInfo]] = {}
    str_start, str_end = target_region

    def process(read: CombinedReadInfo):
        chrom = read.reference_name
        if chrom not in hsnp_info.variant.chrom:
            print(f"chrom {chrom} not in hsnp_info.variant.chrom")
            return

        is_any_hsnp_base = False
        hsnp_allele = None
        covers_str = False

        # Check if the read covers the STR region
        if read.reference_start is not None and read.reference_end is not None:
            if read.reference_start <= str_end and read.reference_end >= str_start:
                covers_str = True

        for base in read.sequence_bases:
            if base.ref_pos == hsnp_info.variant.position:
                is_any_hsnp_base = True
                hsnp_alleles = (hsnp_info.variant.reference_base, hsnp_info.variant.alternate_base)
                is_hsnp_allele = base.base in hsnp_alleles
                base.add_attribute('snp', {
                    'is_snp': True,
                    'hsnp_alleles': hsnp_alleles,
                    'is_hsnp_allele': is_hsnp_allele
                })
                if is_hsnp_allele:
                    hsnp_allele = base.base

        if is_any_hsnp_base:
            read.add_attribute('snp', {'is_hsnp': True, 'hsnp_allele': hsnp_allele})
            if covers_str:
                # For single reads containing both hSNP and STR region
                read.add_attribute('snp', {'paired_hsnp_allele': hsnp_allele})

        if read.has_type(ReadType.LOCALPAIRED):
            if read.query_name not in paired_reads:
                paired_reads[read.query_name] = []
            paired_reads[read.query_name].append(read)

            if len(paired_reads[read.query_name]) >= 2:
                current_reads = paired_reads[read.query_name]
                # Check if any read covers the STR region
                any_read_covers_str = any(
                    read.reference_start is not None and 
                    read.reference_end is not None and
                    read.reference_start <= str_end and 
                    read.reference_end >= str_start 
                    for read in current_reads
                )
                
                if any_read_covers_str:
                    # Find the read with hSNP information
                    hsnp_read = next(
                        (r for r in current_reads if r.has_attribute('snp') and 
                         r.get_attribute('snp').get('is_hsnp')), 
                        None
                    )
                    
                    if hsnp_read:
                        hsnp_allele = hsnp_read.get_attribute('snp').get('hsnp_allele')
                        # Add paired_hsnp_allele to all other reads
                        for r in current_reads:
                            if r != hsnp_read:
                                r.add_attribute('snp', {'paired_hsnp_allele': hsnp_allele})
                
                del paired_reads[read.query_name]

    return process

# processor for classify_allelic_reads
class ClassifyAllelicReadsInput:
    def __init__(self, str_data, region, sample_type='SCC', flanking_seq_length=5, individual_code='1'):
        self.str_data = str_data
        self.region = region
        self.sample_type = sample_type
        self.flanking_seq_length = flanking_seq_length
        self.individual_code = individual_code
        self.is_heterozygous = not region.additional_info[f'is_germ_hom_{individual_code}']
        self.pro_type = self._get_pro_type()
        self.str_unit_length = region['str_unit_length']
        self.stutter = self._build_stutter()
        self.mu_g_allele, self.mu_m_allele, self.ref_g_allele = self._parse_mu_mosaic()
        self.read = self._build_read()
        self.read_qualities = self._build_read_qualities()

    def _get_pro_type(self):
        pro_type_map = {'Bulk': 'bulk', 'MDA': 'mda', 'SCC': 'scc', 'PTA': 'pta', 'SC': 'sc'}
        return pro_type_map.get(self.sample_type, 'scc')

    def _build_stutter(self):
        stutter_types = ['bulk', 'scc', 'mda', 'pta', 'sc']
        stutter = {}
        for stype in stutter_types:
            stutter[stype] = {
                f"rho_{stype}_in": float(self.region.additional_info[f"rho_{stype}_in_{self.individual_code}"]),
                f"up_{stype}_in": float(self.region.additional_info[f"up_{stype}_in_{self.individual_code}"]),
                f"down_{stype}_in": float(self.region.additional_info[f"down_{stype}_in_{self.individual_code}"]),
                f"rho_{stype}_out": float(self.region.additional_info[f"rho_{stype}_out_{self.individual_code}"]),
                f"up_{stype}_out": float(self.region.additional_info[f"up_{stype}_out_{self.individual_code}"]),
                f"down_{stype}_out": float(self.region.additional_info[f"down_{stype}_out_{self.individual_code}"]),
            }
        return stutter

    def _parse_mu_mosaic(self):
        mu_mosaic = eval(self.region.additional_info[f'mu_mosaic_{self.individual_code}'])
        # print(mu_mosaic)
        return mu_mosaic[0], mu_mosaic[1], mu_mosaic[5]

    def _build_read(self):
        left_seq = self.str_data['Left_flanking_seq'][-5:]
        right_seq = self.str_data['Right_flanking_seq'][:5]
        
        # Debugging prints
        # print(f"Left_flanking_seq: {left_seq} (type: {type(left_seq)})")
        # print(f"Right_flanking_seq: {right_seq} (type: {type(right_seq)})")
        # print(f"self.flanking_seq_length: {self.flanking_seq_length} (type: {type(self.flanking_seq_length)})")
        
        return [
            left_seq[-self.flanking_seq_length:] if isinstance(left_seq, str) else left_seq[-self.flanking_seq_length:],
            self.str_data['STR_seq'],
            right_seq[:self.flanking_seq_length] if isinstance(right_seq, str) else right_seq[:self.flanking_seq_length]
        ]

    def _build_read_qualities(self):
        def safe_slice(data, start, end):
            if isinstance(data, (str, list)):
                return data[start:end]
            return data

        left_qualities = self.str_data['Left_flanking_seq_qualities'][-5:]
        right_qualities = self.str_data['Right_flanking_seq_qualities'][:5]
        
        # Debugging prints
        # print(f"Left_flanking_seq_qualities: {left_qualities} (type: {type(left_qualities)})")
        # print(f"Right_flanking_seq_qualities: {right_qualities} (type: {type(right_qualities)})")
        
        return [
            safe_slice(left_qualities, -self.flanking_seq_length, None),
            self.str_data['STR_seq_qualities'],
            safe_slice(right_qualities, 0, self.flanking_seq_length)
        ]

    def get_classify_allelic_reads_args(self):
        return {
            'str_unit_length': self.str_unit_length,
            'stutter': self.stutter,
            'read': self.read,
            'mu_g_allele': self.mu_g_allele,
            'mu_m_allele': self.mu_m_allele,
            'ref_g_allele': self.ref_g_allele,
            'read_qualities': self.read_qualities,
            'pro_type': self.pro_type,
            'is_heterozygous': self.is_heterozygous
        }

def classify_allelic_reads_processor(str_info_dict: Dict[str, Dict[str, Any]], region: STRRegion, sample_type: str = 'SCC', flanking_seq_length: int = 5, individual_code: str = '1'):
    def process(read: CombinedReadInfo):
        read_key = f"{read.query_name}_{read.reference_start}"
        str_info = str_info_dict.get(read_key)
        if str_info:
            input_data = ClassifyAllelicReadsInput(str_info, region, sample_type, flanking_seq_length, individual_code)
            # print(input_data.get_classify_allelic_reads_args())
            result = classify_allelic_reads(**input_data.get_classify_allelic_reads_args())
            read.add_attribute('classify_allelic_reads', result)
    return process
