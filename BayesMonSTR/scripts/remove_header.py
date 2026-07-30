"""Make a BayesMonSTR VCF readable by standard VCF tools.

The genotyper writes an extra tabular header row right after ``#CHROM`` and
appends the model feature columns to every data row, so ``pysam.VariantFile``
and ``bcftools`` reject the file even though the repo's own downstream scripts
(e.g. ``mosaic.py``) parse it fine.  This script rewrites the VCF so that

  * the duplicated ``chr pos id ref alt ...`` header row is dropped,
  * data rows are truncated to the columns declared by ``#CHROM``,
  * CRLF line endings left by ``csv.writer`` are normalised to LF.

The input file is never modified.  Pass ``--features`` to keep the dropped
feature columns in a separate TSV.
"""

import argparse
import gzip
import os
import sys

FIXED_FIELDS = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
EXTRA_HEADER_FIELDS = [
    "chr",
    "pos",
    "id",
    "ref",
    "alt",
    "qual",
    "filter",
    "info",
    "format",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove the extra tabular header row and feature columns "
        "from a BayesMonSTR VCF"
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input VCF ('-' for stdin)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="-",
        help="Output VCF, gzipped if it ends with .gz (default: stdout)",
    )
    parser.add_argument(
        "-f",
        "--features",
        type=str,
        default=None,
        help="Optional TSV to write the dropped feature columns to",
    )
    return parser.parse_args()


def open_text(path, mode):
    """Open a plain or gzipped file, or stdin/stdout for '-'."""
    if path == "-":
        return sys.stdin if mode == "r" else sys.stdout
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return open(path, mode, encoding="utf-8", newline="")


def is_extra_header(fields):
    """Is this the lowercase tabular header row written after #CHROM?"""
    return [f.lower() for f in fields[: len(FIXED_FIELDS)]] == EXTRA_HEADER_FIELDS


def clean_vcf(fin, fout, ffeat=None):
    n_cols = None
    feature_names = None
    n_records = 0
    n_dropped = 0
    n_short = 0

    for line in fin:
        line = line.rstrip("\r\n")
        if not line:
            continue

        if line.startswith("##"):
            fout.write(line + "\n")
            continue

        fields = line.split("\t")

        if line.startswith("#CHROM"):
            n_cols = len(fields)
            fout.write("\t".join(fields) + "\n")
            continue

        if n_cols is None:
            raise ValueError("no #CHROM header line found before the first record")

        if feature_names is None and is_extra_header(fields):
            feature_names = fields[n_cols:]
            if ffeat is not None:
                ffeat.write("\t".join(FIXED_FIELDS[:3] + feature_names) + "\n")
            continue

        extra = fields[n_cols:]
        if extra:
            n_dropped = max(n_dropped, len(extra))
            if ffeat is not None:
                if feature_names is None:
                    # no tabular header row to name them after
                    feature_names = [f"feature_{i + 1}" for i in range(len(extra))]
                    ffeat.write("\t".join(FIXED_FIELDS[:3] + feature_names) + "\n")
                ffeat.write("\t".join(fields[:3] + extra) + "\n")
        elif len(fields) < n_cols:
            n_short += 1

        fout.write("\t".join(fields[:n_cols]) + "\n")
        n_records += 1

    if n_cols is None:
        raise ValueError("no #CHROM header line found")

    return {
        "records": n_records,
        "columns": n_cols,
        "dropped_columns": n_dropped,
        "short_records": n_short,
        "extra_header_removed": feature_names is not None,
    }


def main():
    args = parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if args.features:
        feat_dir = os.path.dirname(args.features)
        if feat_dir:
            os.makedirs(feat_dir, exist_ok=True)

    fin = open_text(args.input, "r")
    fout = open_text(args.output, "w")
    ffeat = open_text(args.features, "w") if args.features else None
    try:
        stats = clean_vcf(fin, fout, ffeat)
    finally:
        for handle in (fin, fout, ffeat):
            if handle is not None and handle not in (sys.stdin, sys.stdout):
                handle.close()

    if not stats["extra_header_removed"]:
        print("No extra header row found; input was already standard", file=sys.stderr)
    if stats["short_records"]:
        print(
            f"Warning: {stats['short_records']} record(s) have fewer than "
            f"{stats['columns']} columns and were left as-is",
            file=sys.stderr,
        )
    print(
        f"Wrote {stats['records']} record(s) with {stats['columns']} column(s); "
        f"dropped up to {stats['dropped_columns']} feature column(s) per record",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

# python remove_header.py -i ../results/str_gt.vcf -o ../results/str_gt.clean.vcf
