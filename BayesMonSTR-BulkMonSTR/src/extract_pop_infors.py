import argparse

# import myutils
import pysam

# TODO: 需要用上群体信息来过滤 germline variants loci DONE #
# HACK: missing informations shouldn't be added in denominator


def process_vcf(vcf_file, output_file):
    vcf_in = pysam.VariantFile(vcf_file)
    gif = open(output_file, "a")
    mof = ".".join(output_file.split(".")[:-1]) + "_mosaic_recurrent_info.txt"
    moff = open(mof, "a")
    for vcf_record in vcf_in.fetch():
        all_sample_name = vcf_record.samples.keys()
        line_gi = ""
        line_mos = ""
        mos_list = []
        for sample_name in all_sample_name:
            germ_index = vcf_record.samples[sample_name]["GT"]
            mosaic_index = vcf_record.samples[sample_name]["MGT"][0].split("/")
            GI = vcf_record.samples[sample_name]["GI"]
            if GI[0] == ".":
                GermGT2_string = "."
            else:
                GermGT = GI[0].split("/")
                GermGT1 = [int(allele) for allele in GermGT]
                GermGT1.sort()
                GermGT2 = [str(allele) for allele in GermGT1]
                GermGT2_string = "_".join(GermGT2)
            if germ_index[0] is None:
                if GermGT2_string == ".":
                    mosaicGT = [".", ".", ".", "."]
                else:
                    mosaicGT = [GermGT2[0], GermGT2[0], GermGT2[1], GermGT2[1]]
                mosaicGT = "_".join(mosaicGT)
            else:
                mosaicGT = [
                    int(germ_index[0]),
                    int(germ_index[1]),
                    int(mosaic_index[0]),
                    int(mosaic_index[1]),
                ]
                mosaicGT.sort()
                mosaicGT = [str(allele) for allele in mosaicGT]
                mosaicGT = "_".join(mosaicGT)
            mos_list.append(mosaicGT)
            line_mos += mosaicGT + "\t"
            line_gi += GermGT2_string + "\t"
        uncallable_num = mos_list.count("._._._.")
        uncallable_frac = uncallable_num / len(all_sample_name)
        chrom = vcf_record.chrom
        STRSTART = vcf_record.info["STRSTART"]
        STREND = vcf_record.info["STREND"]
        MOTIF = vcf_record.info["MOTIF"]
        MOTIF_length = len(MOTIF)
        PERIOD = vcf_record.info["PERIOD"]
        str_id = vcf_record.info["STRID"]
        line_mos = (
            str(chrom)
            + "\t"
            + str(STRSTART)
            + "\t"
            + str(STREND)
            + "\t"
            + str(MOTIF_length)
            + "\t"
            + str(PERIOD)
            + "\t"
            + str_id
            + "\t"
            + MOTIF
            + "\t"
            + line_mos.strip()
            + "\t"
            + str(uncallable_num)
            + "\t"
            + str(uncallable_frac)
        )
        line_mos = line_mos.strip()
        line_mos += "\n"
        line_gi = (
            str(chrom)
            + "\t"
            + str(STRSTART)
            + "\t"
            + str(STREND)
            + "\t"
            + str(MOTIF_length)
            + "\t"
            + str(PERIOD)
            + "\t"
            + str_id
            + "\t"
            + MOTIF
            + "\t"
            + line_gi
        )
        line_gi = line_gi.strip()
        line_gi += "\n"
        # if myutils.acquire_lock(gif):
        gif.write(line_gi)
        # myutils.release_lock(gif)
        # if myutils.acquire_lock(moff):
        moff.write(line_mos)
        # myutils.release_lock(moff)
        # print(
        #     "Finish "
        #     + vcf_record.chrom
        #     + ":"
        #     + str(vcf_record.pos)
        #     + " "
        #     + vcf_record.id
        # )
    moff.close()
    gif.close()


if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Extract population information from BulkMonSTR vcf file"
    )
    parser.add_argument(
        "-v", "--vcf_file", type=str, help="BulkMonSTR vcf file"
    )
    parser.add_argument("-o", "--output_file", type=str, help="output file")
    args = parser.parse_args()

    # 调用函数提取关键词并输出文件名
    process_vcf(args.vcf_file, args.output_file)

# python3 extract_mosaicgt_per_loci_all_samples.py -v /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/mosaic_calling/result_mc_mutation_rate_0613/mosaic_calling_result_seq_based_GRCh37_mosaic_calling_add_mc_2897_add_header_sorted_0613.vcf.gz -o /storage/douyanmeiLab/wangweixiang/data/MosaicSTR/all_samples_stats/results/mosaic_calling_result_seq_based_GRCh37_mosaic_calling_add_mc_2897_mosaicGT_all_loci_0613_2024.txt
# python3 09.extract_mosaicgt_per_loci_all_samples.py -v ${vcf_file} -o ${output_file}
# python3 extract_pop_infors.py -v /storage/douyanmeiLab/wangweixiang/data/MosaicSTR2_20240817/mosaic_calling_20241204/BulkMonSTR_Code_Test/test/mosaic_genotyping/mosaic_genotyping_result/mosaic_fraction_estimation_results/mosaic_result_hg38_chr1_30216700_30216750_mosaic_calling.vcf.gz -o /storage/douyanmeiLab/wangweixiang/data/MosaicSTR2_20240817/mosaic_calling_20241204/BulkMonSTR_Code_Test/test/extract_pop_infors/pop_infors_output/pop_infors_output.txt
# python3 extract_pop_infors.py -v /storage/douyanmeiLab/wangweixiang/data/MosaicSTR2_20240817/mosaic_calling_20241204/merge_vcf_20241205/mosaic_calling_result_seq_based_GRCh37_2897_header_sorted.vcf.gz -o /storage/douyanmeiLab/wangweixiang/data/MosaicSTR2_20240817/mosaic_calling_20241204/extract_features_1485/popGT_32inds/phs1485v3_32inds_str_gi_test2.txt
