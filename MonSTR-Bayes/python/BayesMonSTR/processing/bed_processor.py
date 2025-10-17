# -*- coding: utf-8 -*-

import ast
import csv
import os

import duckdb
import numpy as np
import pandas as pd
import pysam

from ..utils.tools import logger


def load_ref_panel(region):
    """load MS reference panel"""
    with open(region, "r", encoding="utf-8") as file:
        first_line = file.readline()
        num_col = len(first_line.strip().split("\t"))
    try:
        if num_col == 8:
            colname = [
                "chr",
                "start",
                "end",
                "motif_len",
                "len",
                "name",
                "motif",
                "mappability",
            ]
            data_type = {
                "chr": str,
                "start": np.uint,
                "end": np.uint,
                "motif_len": int,
                "len": float,
                "name": str,
                "motif": str,
                "mappability": float,
            }
            bed_data = pd.read_csv(
                region,
                sep="\t",
                header=None,
                names=colname,
                dtype=data_type,
                comment="@",
            )
        elif num_col == 7:
            colname = ["chr", "start", "end", "motif_len", "len", "name", "motif"]
            data_type = {
                "chr": str,
                "start": np.uint,
                "end": np.uint,
                "motif_len": int,
                "len": float,
                "name": str,
                "motif": str,
            }
            bed_data = pd.read_csv(
                region,
                sep="\t",
                header=None,
                names=colname,
                dtype=data_type,
                comment="@",
            )
        elif num_col == 5:
            colname = ["chr", "start", "end", "motif_len", "motif"]
            data_type = {
                "chr": str,
                "start": np.uint,
                "end": np.uint,
                "motif_len": int,
                "motif": str,
            }
            bed_data = pd.read_csv(
                region,
                sep="\t",
                header=None,
                names=colname,
                dtype=data_type,
                comment="@",
            )
        else:
            raise Exception("unknown ref panel format")
    except Exception as e:
        print(f"Load ref panel failed: {e}")
    return bed_data


# @snoop()
def load_metadata(args):
    """load metadata from args"""
    data_type = {
        "individual": str,
        "cell": str,
        "type": str,
        "gender": str,
        "path": str,
        "alpha": float,
        "beta": float,
        "dp": float,
        "note": str,
    }
    metadata_col = pd.read_csv(args.info, sep=",", nrows=0, comment="@").columns
    dtype_dict = {col: data_type[col] for col in metadata_col if col in data_type}
    return pd.read_csv(args.info, sep=",", dtype=dtype_dict, comment="@")


def generate_stutter(args, results):
    """Generate stutter error model parameters"""
    dir_stutter = os.path.dirname(args.stutter)
    if dir_stutter:
        os.makedirs(dir_stutter, exist_ok=True)
    with open(args.stutter, "w", encoding="utf-8") as f:
        csv_out = csv.writer(f)
        csv_out.writerow(
            ["name", "rho_bulk", "up_bulk", "down_bulk", "rho_sc", "up_sc", "down_sc"]
        )
        for row in results:
            csv_out.writerow([i if isinstance(i, str) else f"{i}" for i in row])


def get_stutter(args, region):
    stutter = {
        f"{param}_{prot}_{frame}": 0.9 if param == "rho" else 0.01
        for frame in ("in", "out")
        for prot in ("bulk", "sc", "mda", "pta", "scc")
        for param in ("rho", "up", "down")
    }
    reader = pd.read_csv(args.stutter, chunksize=100000)
    for chunk in reader:
        filtered_rec = chunk[chunk["name"] == region.get("name")]
        if not filtered_rec.empty:
            try:
                target_row = filtered_rec.iloc[0]
                for key, value in target_row.items():
                    try:
                        if not pd.isna(value) and float(value) == float(value):
                            stutter[key] = float(value)
                    except:
                        pass
                return stutter
            except:
                return stutter
    return stutter


def get_region(region, ref_panel=""):
    data = load_ref_panel(ref_panel)
    return dict(data[data.name == region].iloc[0])


def get_afs(args, region_dict, file_names):
    if not args.ab_info:
        return [
            [0.5 for column in sublist] if sublist else None for sublist in file_names
        ]
    flat_file_names = [
        item[0]
        for sublist in file_names
        if sublist
        for item in sublist
        if item and item[1] == "mda"
    ]

    if not flat_file_names:
        return [[None for _ in sublist] if sublist else None for sublist in file_names]

    results = {file_name: None for file_name in flat_file_names}
    for chunk in pd.read_csv(
        args.ab_info, chunksize=100000, index_col=0, usecols=["name"] + flat_file_names
    ):
        if region_dict.get("name") in chunk.index:
            for file_name in flat_file_names:
                value = chunk.at[region_dict.get("name"), file_name]
                try:
                    value = float(value)
                    results[file_name] = None if pd.isna(value) else value
                except ValueError:
                    results[file_name] = None
            break

    res = [
        [
            results.get(column[0], None) if column[1] == "mda" else 0.5
            for column in sublist
        ]
        if sublist
        else None
        for sublist in file_names
    ]
    return res


def get_freq_db(freq_path, region):
    conn = duckdb.connect(freq_path, read_only=True)
    result = conn.execute(
        f"""
        SELECT hap_freq
        FROM hap_freqs
        WHERE name = '{region.get("name")}'
        LIMIT 1
        """
    ).fetchone()
    conn.close()
    if result:
        try:
            return ast.literal_eval(result[0])
        except:
            return None
    return None


def check_overlaps(chrom, start, end, bed_files):
    if not bed_files:
        return False
    overlaps = {}
    for bed_path in bed_files:
        try:
            with pysam.TabixFile(bed_path) as bed:
                print(bed_path)
                records = bed.fetch(f"{chrom}:{start}-{end}", parser=pysam.asBed())
                overlaps[bed_path] = [str(v) for v in records if v]
        except Exception as e:
            logger.error(
                "%s: bed file %s is invalid %s",
                f"{chrom}:{start}-{end}",
                bed_path,
                e,
            )
    return overlaps
