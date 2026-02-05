import pysam
import sys
import gzip
import logging
from collections import defaultdict
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import tqdm


logger = logging.getLogger(__name__)


def setup_logger(log_file: str):
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)


    def add_flush_handler(handler):
        old_emit = handler.emit
        def new_emit(record):
            old_emit(record)
            handler.flush()
        handler.emit = new_emit


    add_flush_handler(fh)
    add_flush_handler(ch)


def parse_bed(bed_panel: str):
    regions = []
    open_func = gzip.open if bed_panel.endswith(".gz") else open
    with open_func(bed_panel, "rt") as f:
        for line in f:
            if line.strip() == '' or line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            # 假设第6列是 id，如果不存在则使用默认值
            region_id = parts[5] if len(parts) >= 6 else f"{chrom}:{start}-{end}"
            regions.append((chrom, start, end, region_id))
    return regions


def get_read_quality(read):
    score = read.mapping_quality * 1000
    base_quals = read.query_qualities
    base_qual_avg = sum(base_quals) / len(base_quals) if base_quals else 0.0
    score += base_qual_avg * 1000
    score += min(read.query_length, 1000)
    try:
        nm = read.get_tag('NM')
        score -= min(nm * 10, 1000)
    except KeyError:
        pass
    return score, base_qual_avg


def process_chromosome(args):
    bam_path, chrom, intervals, temp_dir = args
    stats = defaultdict(int)
    groups = defaultdict(list)
    clear_set = set()
    add_set = set()


    try:
        bamfile = pysam.AlignmentFile(bam_path, "rb")
        if chrom not in bamfile.references:
            logger.warning(f"⚠️ Chromosome {chrom} not in BAM. Skipping.")
            bamfile.close()
            return clear_set, add_set, stats, None


        logger.info(f"🟢 Start processing chromosome {chrom} ({len(intervals)} intervals)")


        temp_bam_path = os.path.join(temp_dir, f"dedup.{chrom}.bam")
        header = bamfile.header.to_dict()
        out_bam = pysam.AlignmentFile(temp_bam_path, "wb", header=header)


        total_intervals = len(intervals)
        last_log_ratio = 0


        for idx, (start, end, region_id) in enumerate(intervals):
            ratio = idx / total_intervals
            if total_intervals > 10 and ratio - last_log_ratio >= 0.1:
                logger.info(f"  ➤ Chromosome {chrom}: {idx}/{total_intervals} intervals processed")
                last_log_ratio = ratio


            for read in bamfile.fetch(chrom, start, end):
                # 扩展区域匹配（允许边界外5bp）
                if read.reference_start > (start - 5) or read.reference_end < (end + 5):
                    continue


                if not read.has_tag('CB'):
                    stats["no_cb"] += 1
                    continue
                if read.is_unmapped or read.query_sequence is None:
                    continue


                cb = read.get_tag('CB')
                key = (read.reference_start, read.reference_end, cb, region_id)
                groups[key].append(read)
                stats["total_in_bed"] += 1


        bamfile.close()


        group_id = 1
        for key, read_list in groups.items():
            start, end, cb, region_id = key


            seq_groups = defaultdict(list)
            for read in read_list:
                seq = read.query_sequence.upper()
                seq_groups[seq].append(read)


            total_count = len(read_list)
            max_count = max(len(rl) for rl in seq_groups.values())
            max_prop = max_count / total_count
            top_seqs = [seq for seq, rl in seq_groups.items() if len(rl) == max_count]


            representative_reads = []
            if max_prop > 0.5:
                for seq in top_seqs:
                    reads = seq_groups[seq]
                    reads.sort(key=lambda r: get_read_quality(r)[0], reverse=True)
                    representative_reads.append(reads[0])
            else:
                for seq, reads in seq_groups.items():
                    reads.sort(key=lambda r: get_read_quality(r)[0], reverse=True)
                    representative_reads.append(reads[0])


            rep_names = {r.query_name for r in representative_reads}


            for read in read_list:
                new_read = read
                qscore_val, bqual_val = get_read_quality(read)
                new_read.set_tag("QS", int(qscore_val), value_type='i')
                new_read.set_tag("QA", round(bqual_val, 2), value_type='f')


                if read.query_name in rep_names:
                    new_read.set_tag("OP", "keep", value_type='Z')
                    if new_read.flag & 1024:
                        clear_set.add(read.query_name)
                        new_read.set_tag("DM", "clear", value_type='Z')
                        new_read.flag -= 1024
                    else:
                        new_read.set_tag("DM", "none", value_type='Z')
                else:
                    new_read.set_tag("OP", "mark", value_type='Z')
                    if not (new_read.flag & 1024):
                        add_set.add(read.query_name)
                        new_read.set_tag("DM", "add", value_type='Z')
                        new_read.flag |= 1024
                    else:
                        new_read.set_tag("DM", "none", value_type='Z')


                out_bam.write(new_read)


            stats["duplicate_groups"] += 1
            stats["duplicates_marked"] += (total_count - len(representative_reads))
            group_id += 1


        out_bam.close()
        logger.info(f"✅ Finished chromosome {chrom}: {stats['duplicates_marked']} duplicates marked, BAM saved to {temp_bam_path}")
        return clear_set, add_set, stats, temp_bam_path


    except Exception as e:
        logger.error(f"❌ Error processing chromosome {chrom}: {e}")
        return set(), set(), stats, None


def merge_bam_files(bam_list: list, output_bam: str, threads: int = 4):
    if not bam_list:
        raise ValueError("No BAM files to merge")
    pysam.merge("-@", str(threads), output_bam, *bam_list)
    logger.info(f"✅ Merged BAM: {output_bam}")


def run(
    input_bam: str,
    bed_panel: str,
    output_bam: str,
    threads: int = 1
):
    threads = threads if threads > 0 else os.cpu_count()
    out_path = os.path.dirname(output_bam)
    os.makedirs(out_path, exist_ok=True)
    log_file = os.path.join(out_path, "deduplicate.log")


    setup_logger(log_file)


    logger.info("Starting optimized parallel deduplication with per-chromosome BAM output...")
    logger.info(f"Input BAM : {input_bam}")
    logger.info(f"BED file  : {bed_panel}")
    logger.info(f"Output BAM: {output_bam}")
    logger.info(f"Threads   : {threads}")


    regions = parse_bed(bed_panel)
    logger.info(f"Found {len(regions)} intervals.")
    if len(regions) == 0:
        logger.warning("No intervals found. Exiting.")
        sys.exit(0)


    regions_by_chrom = defaultdict(list)
    for chrom, start, end, region_id in regions:
        regions_by_chrom[chrom].append((start, end, region_id))


    logger.info(f"Processing {len(regions_by_chrom)} chromosomes in parallel...")


    with tempfile.TemporaryDirectory() as temp_dir:
        args_list = [
            (input_bam, chrom, intervals, temp_dir)
            for chrom, intervals in regions_by_chrom.items()
        ]


        all_clear = set()
        all_add = set()
        total_stats = defaultdict(int)
        temp_bam_files = []


        with ProcessPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(process_chromosome, arg) for arg in args_list]
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="📦 Chromosomes", unit="chrom", colour="green"):
                result = future.result()
                if result is None:
                    continue
                clear_set, add_set, stats, bam_file = result
                if bam_file is None:
                    continue
                all_clear.update(clear_set)
                all_add.update(add_set)
                for k, v in stats.items():
                    total_stats[k] += v
                temp_bam_files.append(bam_file)


        conflict = all_clear.intersection(all_add)
        if conflict:
            logger.warning(f"⚠️ Found {len(conflict)} reads in both clear and add sets. Resolving by prioritizing clear.")
            all_add -= conflict


        logger.info("=== Final Statistics ===")
        logger.info(f"Total reads in BED intervals:        {total_stats['total_in_bed']}")
        logger.info(f"  - No CB tag:                       {total_stats['no_cb']}")
        logger.info(f"Duplicate groups detected:           {total_stats['duplicate_groups']}")
        logger.info(f"Reads marked as duplicate:           {total_stats['duplicates_marked']}")
        logger.info(f"Reads to clear duplicate mark:       {len(all_clear)}")
        logger.info(f"Reads to add duplicate mark:         {len(all_add)}")
        logger.info(f"Conflicts resolved:                  {len(conflict)}")


        merged_bam_path = output_bam + ".merged.tmp.bam"
        logger.info(f"Merging {len(temp_bam_files)} chromosome BAMs into {merged_bam_path}...")
        merge_bam_files(temp_bam_files, merged_bam_path, threads=threads)


        try:
            pysam.sort(
                "-o", output_bam,
                "-@", str(threads),
                merged_bam_path
            )
            os.remove(merged_bam_path)
            logger.info(f"✅ Sorting completed: {output_bam}")
        except Exception as e:
            logger.error(f"❌ Sorting failed: {e}")
            raise


        logger.info(f"Indexing {output_bam}...")
        try:
            pysam.index("-@", str(threads), output_bam)
            logger.info(f"✅ Index created: {output_bam}.bai")
        except Exception as e:
            logger.error(f"❌ Indexing failed: {e}")


    logger.info(f"✅ Final BAM: {output_bam}")
    logger.info(f"✅ Log: {log_file}")
    logger.info("🎉 All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized parallel BAM deduplication without extra summary files.")
    parser.add_argument("-i", "--input_bam", required=True, help="Input BAM file")
    parser.add_argument("-b", "--bed_file", required=True, help="BED file (can be .gz)")
    parser.add_argument("-o", "--output_bam", required=True, help="Output sorted BAM file")
    parser.add_argument("-t", "--threads", type=int, default=-1, help="Number of threads (default: all available)")
    args = parser.parse_args()


    run(args.input_bam, args.bed_file, args.output_bam, args.threads)