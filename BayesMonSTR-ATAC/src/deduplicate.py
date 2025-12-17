import pysam
import sys
import gzip
import logging
from collections import defaultdict
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import random
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
            chrom, start, end, id = parts[0], int(parts[1]), int(parts[2]), parts[5]
            regions.append((chrom, start, end, id))
    return regions


def get_read_key(read):
    if not read or read.is_unmapped:
        return None
    try:
        cb = read.get_tag('CB')
    except (ValueError, KeyError):
        cb = 'None'
    return (
        read.reference_start,
        read.reference_end,
        cb
    )


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


        for idx, (start, end, id) in enumerate(intervals):
            ratio = idx / total_intervals
            if total_intervals > 10 and ratio - last_log_ratio >= 0.1:
                logger.info(f"  ➤ Chromosome {chrom}: {idx}/{total_intervals} intervals processed")
                last_log_ratio = ratio


            for read in bamfile.fetch(chrom, start, end):
                if read.reference_start > (start - 5) or read.reference_end < (end + 5):
                    continue


                if not read.has_tag('CB'):
                    stats["no_cb"] += 1
                    continue
                if read.is_unmapped or read.query_sequence is None:
                    continue


                cb = read.get_tag('CB')
                key = (read.reference_start, read.reference_end, cb, id)
                groups[key].append(read)
                stats["total_in_bed"] += 1


        bamfile.close()


        groups_output_file = os.path.join(temp_dir, f"groups.{chrom}.txt")
        summary_output_file = os.path.join(temp_dir, f"summary.{chrom}.tsv")


        with open(groups_output_file, 'w') as gf, open(summary_output_file, 'w') as sf:
            sf.write("\t".join([
                "id", "CB", "start", "end", "total_reads", "seq_types_count",
                "sequence_details", "final_sequences", "final_proportion",
                "final_rep_count", "final_rep_names", "final_rep_quality",
                "modified_count", "cleared_reads_count", "cleared_read_names",
                "added_reads_count", "added_read_names"
            ]) + "\n")


            group_id = 1
            for key, read_list in groups.items():
                # if len(read_list) <= 1:
                #     continue
                start, end, cb, interval_id = key


                seq_groups = defaultdict(list)
                for read in read_list:
                    seq = read.query_sequence.upper()
                    seq_groups[seq].append(read)


                total_count = len(read_list)
                seq_type_count = len(seq_groups)
                max_count = max(len(rl) for rl in seq_groups.values())
                max_prop = round(max_count / total_count, 4)
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
                cleared_in_group = set()
                added_in_group = set()


                for read in read_list:
                    if read.query_name not in rep_names:
                        if not (read.flag & 1024):
                            add_set.add(read.query_name)
                            added_in_group.add(read.query_name)
                    else:
                        if read.flag & 1024:
                            clear_set.add(read.query_name)
                            cleared_in_group.add(read.query_name)


                seq_details_parts = []
                for seq, seq_reads in seq_groups.items():
                    count = len(seq_reads)
                    prop = count / total_count
                    read_names = ",".join(sorted(r.query_name for r in seq_reads))
                    seq_details_parts.append(f"{seq}_{count}_{prop:.4f}_{read_names}")
                sequence_details_str = ";".join(seq_details_parts)


                final_seqs = ",".join(top_seqs)
                final_prop = max_prop
                final_rep_names = ",".join(sorted(r.query_name for r in representative_reads))
                final_rep_quality = ";".join(f"{get_read_quality(read)[1]:.2f}" for read in representative_reads)


                sf.write("\t".join(map(str, [
                    interval_id, cb, start, end, total_count, seq_type_count,
                    sequence_details_str, final_seqs, f"{final_prop:.4f}",
                    len(representative_reads), final_rep_names, final_rep_quality,
                    len(read_list) - len(representative_reads),
                    len(cleared_in_group), ",".join(sorted(cleared_in_group)) if cleared_in_group else "NA",
                    len(added_in_group), ",".join(sorted(added_in_group)) if added_in_group else "NA"
                ])) + "\n")


                gf.write(f"--- Group {group_id} (chrom={chrom}) start={start}, end={end}, CB={cb}, id={interval_id} ---\n")
                gf.write(f"Total reads: {total_count}\n")
                gf.write(f"Sequence types: {seq_type_count}\n")
                gf.write(f"Representative sequences selected ({seq_type_count} types):\n")


                for seq, reads in seq_groups.items():
                    rep_read = max(reads, key=lambda r: get_read_quality(r)[0])
                    qscore, bqual = get_read_quality(rep_read)
                    gf.write(f"  🔹 {seq} (n={len(reads)})\n")
                    gf.write(f"    🏆 {rep_read.query_name} (MAPQ={rep_read.mapping_quality}, "
                             f"QS={int(qscore)}, BQ_avg={bqual:.2f}, CIGAR={rep_read.cigarstring})\n")


                    sorted_reads = sorted(reads, key=lambda r: get_read_quality(r)[0], reverse=True)
                    for read in sorted_reads:
                        new_read = read

                        qscore_val, bqual_val = get_read_quality(read)
                        new_read.set_tag("QS", int(qscore_val), value_type='i')
                        new_read.set_tag("QA", round(bqual_val, 2), value_type='f')


                        if read.query_name in rep_names:
                            new_read.set_tag("OP", "keep", value_type='Z')
                            if read.query_name in cleared_in_group:
                                new_read.set_tag("DM", "clear", value_type='Z')
                                if new_read.flag & 1024:
                                    new_read.flag -= 1024
                            else:
                                new_read.set_tag("DM", "none", value_type='Z')
                        else:
                            new_read.set_tag("OP", "mark", value_type='Z')
                            if read.query_name in added_in_group:
                                new_read.set_tag("DM", "add", value_type='Z')
                                if not (new_read.flag & 1024):
                                    new_read.flag |= 1024
                            else:
                                new_read.set_tag("DM", "none", value_type='Z')


                        out_bam.write(new_read)


                        line = str(read).rstrip()
                        line += f"\tOP:Z:{'keep' if read.query_name in rep_names else 'mark'}"
                        line += f"\tQS:i:{int(qscore_val)}"
                        line += f"\tQA:f:{bqual_val:.2f}"
                        line += f"\tDM:Z:{new_read.get_tag('DM')}"
                        gf.write(line + "\n")


                    gf.write("\n")
                group_id += 1
                stats["duplicate_groups"] += 1
                stats["duplicates_marked"] += (total_count - len(representative_reads))


        out_bam.close()
        logger.info(f"✅ Finished chromosome {chrom}: {stats['duplicates_marked']} duplicates marked, BAM saved to {temp_bam_path}")
        return clear_set, add_set, stats, (groups_output_file, summary_output_file, temp_bam_path)


    except Exception as e:
        logger.error(f"❌ Error processing chromosome {chrom}: {e}")
        return set(), set(), stats, None


def merge_bam_files(bam_list: list, output_bam: str, threads: int = 4):
    if not bam_list:
        raise ValueError("No BAM files to merge")

    pysam.merge("-@", str(threads), output_bam, *bam_list)
    print(f"✅ Merged BAM: {output_bam}")


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
    modified_file = os.path.join(out_path, "modified_reads.txt")
    groups_file_final = os.path.join(out_path, "duplicate_groups.txt")
    summary_file_final = os.path.join(out_path, "summary.tsv")

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
    for chrom, start, end, id in regions:
        regions_by_chrom[chrom].append((start, end, id))
        
    logger.info(f"Processing {len(regions_by_chrom)} chromosomes in parallel...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        args_list = [
            (input_bam, chrom, intervals, temp_dir)
            for chrom, intervals in regions_by_chrom.items()
        ]
        
        all_clear = set()
        all_add = set()
        total_stats = defaultdict(int)
        temp_group_files = {}
        temp_bam_files = []
        
        with ProcessPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(process_chromosome, arg) for arg in args_list]
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="📦 Chromosomes", unit="chrom", colour="green"):
                result = future.result()
                if result is None:
                    continue
                clear_set, add_set, stats, files = result
                if clear_set is None:
                    continue
                all_clear.update(clear_set)
                all_add.update(add_set)
                for k, v in stats.items():
                    total_stats[k] += v
                if files and len(files) == 3:
                    groups_file, summary_file, bam_file = files
                    chrom = os.path.basename(groups_file).split('.')[1]
                    temp_group_files[chrom] = {
                        'groups': groups_file,
                        'summary': summary_file
                    }
                    temp_bam_files.append(bam_file)
                elif files:
                    chrom = "unknown"
                    temp_group_files[chrom] = {
                        'groups': files[0] if len(files) > 0 else None,
                        'summary': files[1] if len(files) > 1 else None
                    }
                    if len(files) > 2:
                        temp_bam_files.append(files[2])
        
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
        
        with open(modified_file, 'w') as f:
            f.write("# Cleared duplicate marks\n")
            for name in sorted(all_clear):
                f.write(f"{name}\n")
            f.write("\n# Added duplicate marks\n")
            for name in sorted(all_add):
                f.write(f"{name}\n")
        logger.info(f"Modified reads list saved to: {modified_file}")
        
        with open(groups_file_final, 'w') as final_out:
            group_id = 1
            for chrom in sorted(temp_group_files.keys()):
                entry = temp_group_files[chrom]
                if not entry.get('groups'):
                    continue
                try:
                    with open(entry['groups'], 'r') as f:
                        for line in f:
                            if line.startswith("--- Group"):
                                final_out.write(f"--- Group {group_id} {line.split('--- Group')[1]}")
                                group_id += 1
                            else:
                                final_out.write(line)
                except Exception as e:
                    logger.error(f"Failed to read groups file: {e}")
        logger.info(f"✅ Merged duplicate groups saved to: {groups_file_final}")


        with open(summary_file_final, 'w') as final_sf:
            header_written = False
            for chrom in sorted(temp_group_files.keys()):
                entry = temp_group_files[chrom]
                if not entry.get('summary'):
                    continue
                try:
                    with open(entry['summary'], 'r') as f:
                        lines = f.readlines()
                        if not header_written:
                            final_sf.writelines(lines)
                            header_written = True
                        else:
                            final_sf.writelines(lines[1:])
                except Exception as e:
                    logger.error(f"Failed to read summary file: {e}")
        logger.info(f"✅ Merged summary saved to: {summary_file_final}")


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
    parser = argparse.ArgumentParser(description="Final streaming parallel BAM deduplication with per-chromosome BAM output.")
    parser.add_argument("-i", "--input_bam", help="Input BAM file")
    parser.add_argument("-b", "--bed_file", help="BED file (can be .gz)")
    parser.add_argument("-o", "--output_bam", help="Output sorted BAM file")
    parser.add_argument("-t", "--threads", type=int, default=-1, help="Number of threads to use (default: -1, use all available cores)")
    args = parser.parse_args()
    
    run(args.input_bam, args.bed_file, args.output_bam, args.threads)