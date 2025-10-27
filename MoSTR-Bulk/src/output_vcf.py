## locus: bed,fasta,vcf germline and somatic ##
## 一个位点有一个 bed 几个 fasta items 和 一个 vcf item 记录 germline 位点 和 一个 vcf item 记录 somatic 位点 ##
## 为什么说 STR 突变率高，原因有二：1. STR 位点碱基数量多可突变位点数量多 2. STR 位点本身突变率高由于重复结构（突变发生的机制）##
## locus-level:all sample-level
## sample-level
## read-level
## genome-level
## allele-level
## using hap-based vcf or alignment-based vcf
## link all infos through str_id
import datetime
import pysam
import fcntl

# import os
# import logging
# XXX TODO: 为了适配 vcf 格式，将所有标题标记的数据格式 int 和 float 改正为 string 格式


def acquire_lock(file):
    try:
        fcntl.flock(file, fcntl.LOCK_EX)
        return True
    except BlockingIOError:
        return False


def release_lock(file):
    fcntl.flock(file, fcntl.LOCK_UN)


def create_vcf_header(genome_wide_info_dict, output_vcf):
    # Get the current date
    current_date = datetime.date.today()
    # Format the current date as "MM/DD/YYYY"
    formatted_date = current_date.strftime("%Y%m%d")
    source = "MosaicSTRV0.1"
    reference = genome_wide_info_dict["reference_version"]
    # sample_mode = genome_wide_info_dict["sample_mode"]
    # sim_depth = genome_wide_info_dict["sim_depth"]
    # mutation_mode = genome_wide_info_dict["mutation_mode"]
    # dbSNP = genome_wide_info_dict["dbSNP"]
    commands = genome_wide_info_dict["commands"]
    ref_fa = genome_wide_info_dict["ref_fa"]
    # homopolymer_in_frame_u=genome_wide_info_dict["homopolymer_in_frame_u"]
    # homopolymer_in_frame_d=genome_wide_info_dict["homopolymer_in_frame_d"]
    # homopolymer_in_frame_p=genome_wide_info_dict["homopolymer_in_frame_p"]
    # tandem_repeats_in_frame_u=genome_wide_info_dict["tandem_repeats_in_frame_u"]
    # tandem_repeats_in_frame_d=genome_wide_info_dict["tandem_repeats_in_frame_d"]
    # tandem_repeats_in_frame_p=genome_wide_info_dict["tandem_repeats_in_frame_p"]
    # out_frame_model = genome_wide_info_dict["out_frame_model"]
    # out_frame_u=genome_wide_info_dict["out_frame_u"]
    # out_frame_d=genome_wide_info_dict["out_frame_d"]
    # out_frame_p=genome_wide_info_dict["out_frame_p"]
    # hap_based_or_align_based = genome_wide_info_dict["hap_based_or_align_based"]
    # stutter_model_params = genome_wide_info_dict["stutter_model_params"]
    # inframe_single_step_prob = stutter_model_params["inframe_single_step_prob"]
    # inframe_ins_prob = stutter_model_params["inframe_ins_prob"]
    # inframe_del_prob = stutter_model_params["inframe_del_prob"]
    # outframe_single_step_prob = stutter_model_params[
    #     "outframe_single_step_prob"
    # ]
    # outframe_ins_prob = stutter_model_params["outframe_ins_prob"]
    # outframe_del_prob = stutter_model_params["outframe_del_prob"]
    sample_name_list = genome_wide_info_dict["sample_name_list"]
    sample_name_list_string = "\t".join(sample_name_list)
    sample_num = len(sample_name_list)
    contigs = ""
    fasta_file = pysam.Fastafile(ref_fa)
    # %d用于格式化字符串,表示整数的占位符
    for contig in fasta_file.references:
        contigs += (
            "##contig=<ID=%s,length=%d>"
            % (contig, fasta_file.get_reference_length(contig))
            + "\n"
        )
    header = "##fileformat=VCFv4.2" + "\n"
    header += "##fileDate=%s" % (formatted_date) + "\n"
    header += "##source=%s" % (source) + "\n"
    # header += '##vcftype=%s'%(hap_based_or_align_based) + '\n'
    header += "##reference=%s" % (reference) + "\n"
    # header += '##samplemode=%s' % (sample_mode) + '\n'
    # header += '##mutationmode=%s' % (mutation_mode) + '\n' # Repeat expansion MSS MSI
    # header += '##simdepth=%s' % (sim_depth) + '\n'
    header += "##samplenumber=%s" % (sample_num) + "\n"
    # header += '##out-frame-model=%s' % (out_frame_model) + '\n'
    # if out_frame_model == "geometric":
    #     header += '##out-frame-u=%s' % (out_frame_u) + '\n'
    #     header += '##out-frame-d=%s' % (out_frame_d) + '\n'
    #     header += '##out-frame-p=%s' % (out_frame_p) + '\n'
    # else:
    #     pass
    # header += '##homopolymer-in-frame-u=%s' % (homopolymer_in_frame_u) + '\n'
    # header += '##homopolymer-in-frame-d=%s' % (homopolymer_in_frame_d) + '\n'
    # header += '##homopolymer-in-frame-p=%s' % (homopolymer_in_frame_p) + '\n'
    # header += '##tandem-repeats-in-frame-u=%s' % (tandem_repeats_in_frame_u) + '\n'
    # header += '##tandem-repeats-in-frame-d=%s' % (tandem_repeats_in_frame_d) + '\n'
    # header += '##tandem-repeats-in-frame-p=%s' % (tandem_repeats_in_frame_p) + '\n'
    header += "##commands=%s" % (commands) + "\n"
    # header += '##FILTER=<ID=q10,Description="Quality below 10">'+ '\n'
    # header += '##FILTER=<ID=s50,Description="Less than 50% of samples have data">'+ '\n'
    header += '##FILTER=<ID=PASS,Description="PASS Filtering">' + "\n"
    header += '##FILTER=<ID=FAIL,Description="FAIL Filtering">' + "\n"
    # header += '##FILTER=<ID=FP,Description="False positive">'+ '\n'
    # header += '##FILTER=<ID=SOMATIC,Description="Somatic mutation">'+ '\n'
    # header += '##FILTER=<ID=GERMLINE,Description="Germline variants">'+ '\n'
    header += (
        '##INFO=<ID=STRID,Number=1,Type=String,Description="STR id from HipSTR'
        ' reference panel">'
        + "\n"
    )
    header += (
        '##INFO=<ID=STRSTART,Number=1,Type=String,Description="Inclusive'
        ' start coodinate for the repetitive portion of the reference allele">'
        + "\n"
    )
    header += (
        '##INFO=<ID=STREND,Number=1,Type=String,Description="Inclusive end'
        ' coordinate for the repetitive portion of the reference allele">'
        + "\n"
    )
    header += (
        '##INFO=<ID=MOTIF,Number=1,Type=String,Description="STR motif">' + "\n"
    )
    header += (
        '##INFO=<ID=PERIOD,Number=1,Type=String,Description="Length of STR'
        ' motif">'
        + "\n"
    )
    header += (
        '##INFO=<ID=RLEN,Number=1,Type=String,Description="Total length of'
        ' Ref STR region">'
        + "\n"
    )
    header += (
        '##INFO=<ID=ZYGOSITY,Number=1,Type=String,Description="Haploid or'
        ' Diploid">'
        + "\n"
    )
    header += (
        '##INFO=<ID=AN,Number=1,Type=String,Description="Total Allele Number">'
        + "\n"
    )
    # header += '##INFO=<ID=AV,Number=A,Type=String,Description="Alignment-based variants representation for each alternative allele">' + '\n'
    header += (
        '##INFO=<ID=RNDIFFS,Number=A,Type=String,Description="Difference in'
        ' repeats number between REF and ALT alleles">'
        + "\n"
    )
    header += (
        '##INFO=<ID=BPDIFFS,Number=A,Type=String,Description="Base pair'
        ' difference of each alternate allele from the reference allele">'
        + "\n"
    )
    # header += '##INFO=<ID=DB,Number=1,Type=Flag,Description="dbSNP membership, %s">'%(dbSNP)+ '\n'
    header += (
        '##INFO=<ID=STUTTER_INFRAME_UP,Number=1,Type=String,Description="Probability'
        ' that stutter causes an in-frame increase in obs. STR size">'
        + "\n"
    )
    header += (
        '##INFO=<ID=STUTTER_INFRAME_DOWN,Number=1,Type=String,Description="Probability'
        ' that stutter causes an in-frame decrease in obs. STR size">'
        + "\n"
    )
    header += (
        '##INFO=<ID=STUTTER_INFRAME_PGEOM,Number=1,Type=String,Description="Parameter'
        ' for in-frame geometric step size distribution">'
        + "\n"
    )
    header += (
        '##INFO=<ID=STUTTER_OUTFRAME_UP,Number=1,Type=String,Description="Probability'
        ' that stutter causes an in-frame increase in obs. STR size">'
        + "\n"
    )
    header += (
        '##INFO=<ID=STUTTER_OUTFRAME_DOWN,Number=1,Type=String,Description="Probability'
        ' that stutter causes an in-frame decrease in obs. STR size">'
        + "\n"
    )
    header += (
        '##INFO=<ID=STUTTER_OUTFRAME_PGEOM,Number=1,Type=String,Description="Parameter'
        ' for in-frame geometric step size distribution">'
        + "\n"
    )
    # HACK: temp annotation for header, need revise and description TODO list
    # Add missing FORMAT fields to the header
    header += (
        '##INFO=<ID=ALL_HVAR_NUM,Number=1,Type=String,Description="Description'
        ' of ALL_HVAR_NUM">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=CMP,Number=1,Type=String,Description="Description of'
        ' CMP">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=RMF,Number=1,Type=String,Description="Description of'
        ' RMF">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=EMLEAD,Number=1,Type=String,Description="Description of'
        ' EMLEAD">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=EMLEAF,Number=1,Type=String,Description="Description of'
        ' EMLEAF">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=MLEUAF,Number=1,Type=String,Description="Description of'
        ' MLEUAF">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=NALLELES,Number=1,Type=String,Description="Description'
        ' of NALLELES">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=MBP,Number=1,Type=String,Description="Description of'
        ' MBP">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PHA,Number=1,Type=String,Description="Description of'
        ' PHA">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=UPR,Number=1,Type=String,Description="Description of'
        ' UPR">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=GERMLR,Number=1,Type=String,Description="Description of'
        ' GERMLR">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=GERMLRTP,Number=1,Type=String,Description="Description'
        ' of GERMLRTP">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PPP,Number=1,Type=String,Description="Description of'
        ' PPP">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PODD,Number=1,Type=String,Description="Description of'
        ' PODD">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PODR,Number=1,Type=String,Description="Description of'
        ' PODR">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PODH,Number=1,Type=String,Description="Description of'
        ' PODH">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=POAH,Number=1,Type=String,Description="Description of'
        ' POAH">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PODHR,Number=1,Type=String,Description="Description of'
        ' PODHR">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PMAP,Number=1,Type=String,Description="Description of'
        ' PMAP">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PMLEAH,Number=1,Type=String,Description="Description of'
        ' PMLEAH">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PMLEEH,Number=1,Type=String,Description="Description of'
        ' PMLEEH">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PMLEEHR,Number=1,Type=String,Description="Description of'
        ' PMLEEHR">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PMLEDD,Number=1,Type=String,Description="Description of'
        ' PMLEDD">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PMLEDR,Number=1,Type=String,Description="Description of'
        ' PMLEDR">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PFAH,Number=1,Type=String,Description="Description of'
        ' PFAH">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=MAPLR,Number=1,Type=String,Description="Description of'
        ' MAPLR">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=MAPLRTP,Number=1,Type=String,Description="Description of'
        ' MAPLRTP">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=MLEPEAD,Number=1,Type=String,Description="Description of'
        ' MLEPEAD">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=MLEDISHD,Number=1,Type=String,Description="Description'
        ' of MLEDISHD">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=MLEPUAF,Number=1,Type=String,Description="Description of'
        ' MLEPUAF">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=MLEPALLAD,Number=1,Type=String,Description="Description'
        ' of MLEPALLAD">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=MLEPALLAL,Number=1,Type=String,Description="Description'
        ' of MLEPALLAL">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PDOAD,Number=1,Type=String,Description="Description of'
        ' PDOAD">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PDOADL,Number=1,Type=String,Description="Description of'
        ' PDOADL">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=POMDD,Number=1,Type=String,Description="Description of'
        ' POMDD">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=POMDR,Number=1,Type=String,Description="Description of'
        ' POMDR">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PSOMDD,Number=1,Type=String,Description="Description of'
        ' PSOMDD">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PSOMDR,Number=1,Type=String,Description="Description of'
        ' PSOMDR">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PGOMDD,Number=1,Type=String,Description="Description of'
        ' PGOMDD">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PGOMDR,Number=1,Type=String,Description="Description of'
        ' PGOMDR">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PMMDD,Number=1,Type=String,Description="Description of'
        ' PMMDD">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PMMDR,Number=1,Type=String,Description="Description of'
        ' PMMDR">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PSMMDD,Number=1,Type=String,Description="Description of'
        ' PSMMDD">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PSMMDR,Number=1,Type=String,Description="Description of'
        ' PSMMDR">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PGMMDD,Number=1,Type=String,Description="Description of'
        ' PGMMDD">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PGMMDR,Number=1,Type=String,Description="Description of'
        ' PGMMDR">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=NHSNPN,Number=1,Type=String,Description="Description of'
        ' NHSNPN">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=NHSNPINDELN,Number=1,Type=String,Description="Description'
        ' of NHSNPINDELN">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=MAPLRS,Number=1,Type=String,Description="Description of'
        ' MAPLRS">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=MAPLRTPS,Number=1,Type=String,Description="Description'
        ' of MAPLRTPS">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=FDPC,Number=1,Type=String,Description="Description of'
        ' FDPC">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=GIOAF,Number=1,Type=String,Description="Description of'
        ' GIOAF">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=GIMLEAF,Number=1,Type=String,Description="Description of'
        ' GIMLEAF">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=GIUAF,Number=1,Type=String,Description="Description of'
        ' GIUAF">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PRHN,Number=1,Type=String,Description="Description of'
        ' PRHN">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=OPALLAD,Number=1,Type=String,Description="Description of'
        ' OPALLAD">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=MLEPEAF,Number=1,Type=String,Description="Description of'
        ' MLEPEAF">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PMB,Number=1,Type=String,Description="Description of'
        ' PMB">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=POB,Number=1,Type=String,Description="Description of'
        ' POB">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PGHAF,Number=1,Type=String,Description="Description of'
        ' PGHAF">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PMHAF,Number=1,Type=String,Description="Description of'
        ' PMHAF">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PH3AF,Number=1,Type=String,Description="Description of'
        ' PH3AF">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PGD,Number=1,Type=String,Description="Description of'
        ' PGD">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PSD,Number=1,Type=String,Description="Description of'
        ' PSD">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PMD,Number=1,Type=String,Description="Description of'
        ' PMD">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=ALLSINGLE,Number=1,Type=String,Description="Description'
        ' of ALLSINGLE">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=HPRHN,Number=1,Type=String,Description="Description of'
        ' HPRHN">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=obs_phase_state,Number=1,Type=String,Description="Description'
        ' of obs_phase_state">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=obs_hap_state,Number=1,Type=String,Description="Description'
        ' of obs_hap_state">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=obs_hap_count,Number=1,Type=String,Description="Description'
        ' of obs_hap_count">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=mle_phase_state,Number=1,Type=String,Description="Description'
        ' of mle_phase_state">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=mle_hap_state,Number=1,Type=String,Description="Description'
        ' of mle_hap_state">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=mle_hap_count,Number=1,Type=String,Description="Description'
        ' of mle_hap_count">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=NORMGERM1,Number=1,Type=String,Description="Description'
        ' of NORMGERM1">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=NORMGERM2,Number=1,Type=String,Description="Description'
        ' of NORMGERM2">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=NORMMOSAIC,Number=1,Type=String,Description="Description'
        ' of NORMMOSAIC">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=NORMSECOND,Number=1,Type=String,Description="Description'
        ' of NORMSECOND">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=SECISGERM,Number=1,Type=String,Description="Description'
        ' of SECISGERM">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=obs_mut_order,Number=1,Type=String,Description="Description'
        ' of obs_mut_order">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=obs_depth_string,Number=1,Type=String,Description="Description'
        ' of obs_depth_string">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=mle_mut_order,Number=1,Type=String,Description="Description'
        ' of mle_mut_order">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=mle_depth_string,Number=1,Type=String,Description="Description'
        ' of mle_depth_string">'
        + "\n"
    )
    # HACK: temp annotation for header, need revise and description TODO list
    header += (
        '##FORMAT=<ID=GT,Number=M,Type=String,Description="Germline'
        ' Genotypes">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=GI,Number=M,Type=String,Description="Germline'
        ' Genotypes based on germline likelihood">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=GIP,Number=1,Type=String,Description="Germline'
        ' Genotypes based on germline likelihood posterior">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=GIQ,Number=1,Type=String,Description="Germline Genotypes'
        ' based on germline likelihood best vs second likelihood ratio">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=GIQP,Number=1,Type=String,Description="Germline'
        " Genotypes based on germline likelihood best vs second likelihood"
        ' ratio test p-value">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=MGT,Number=M,Type=String,Description="Mosaic Genotypes">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=VARIANTTYPE,Number=1,Type=String,Description="Germline'
        ' variants or Somatic variants or unknown">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=MUTP,Number=1,Type=String,Description="Mismatch or'
        ' INDEL">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=FRAME,Number=1,Type=String,Description="In-frame or'
        ' out-frame mutation">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=MN,Number=1,Type=String,Description="Mutation Allele'
        ' Number">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=REGION,Number=1,Type=String,Description="STR region or'
        ' Flanking region">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=MP,Number=1,Type=String,Description="Max Mosaic'
        ' Genotypes posterior">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=AMP,Number=1,Type=String,Description="All Mosaic'
        ' Genotypes posterior">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=GQ,Number=1,Type=String,Description="Log-likelihood'
        ' ratio test of max mosaic GT vs second mosaic GT">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=GG,Number=1,Type=String,Description="Log-likelihood'
        ' ratio test of max mosaic GT vs germline mosaic GT">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=MF,Number=1,Type=String,Description="Mosaic fraction">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PSTR,Number=2,Type=String,Description="Perfect STR for'
        ' mutated allele and mutation allele">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=DP,Number=1,Type=String,Description="Available Read'
        ' Depth">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=FDP,Number=1,Type=String,Description="Filtered out Read'
        ' Depth">'
        + "\n"
    )
    # header += '##FORMAT=<ID=FPD,Number=1,Type=String,Description="Read Depth for all false positive alleles">' + '\n'
    # header += '##FORMAT=<ID=FPN,Number=1,Type=String,Description="False positive allele number">' + '\n'
    header += (
        '##FORMAT=<ID=EAF,Number=2,Type=String,Description="expected allele'
        ' frequency of mosaic allele">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=EAD,Number=2,Type=String,Description="expected allele'
        ' counts of mosaic allele">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=AAD,Number=N,Type=String,Description="All allele'
        ' counts">'
        + "\n"
    )
    # header += '##FORMAT=<ID=MAF,Number=2,Type=String,Description="Mosaic cells allele frequency">' + '\n'
    header += (
        '##FORMAT=<ID=NSTUTTER,Number=1,Type=String,Description="Number of'
        ' stutter alleles">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=NFSTUTTER,Number=1,Type=String,Description="Fraction of'
        ' stutter alleles">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=DSTUTTER,Number=1,Type=String,Description="Number of'
        ' reads with a stutter indel in the STR region">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=DFSTUTTER,Number=1,Type=String,Description="Fraction of'
        ' reads with a stutter indel in the STR region">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=FILTER,Number=1,Type=String,Description="Reason for'
        ' filtering the current call, or PASS if the call was not filtered">'
        + "\n"
    )

    header += (
        '##FORMAT=<ID=PS,Number=1,Type=String,Description="Phase location">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=NSNP,Number=2,Type=String,Description="Nearby SNP seq">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=HVAF,Number=2,Type=String,Description="hSNP VAF">' + "\n"
    )
    header += (
        '##FORMAT=<ID=PP,Number=1,Type=String,Description="Phase Posterior'
        ' P1/(P1+P2)">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PQ,Number=1,Type=String,Description="binominal test'
        ' pvalue for concordant reads">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PGT,Number=M,Type=String,Description="Phased Germline'
        ' Genotypes">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PMGT,Number=M,Type=String,Description="Phase Mosaic'
        ' Genotypes">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PVARIANTTYPE,Number=1,Type=String,Description="Phase'
        ' Germline variants or Somatic variants or unknown">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PMUTP,Number=1,Type=String,Description="Phase Mismatch'
        ' or INDEL">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PFRAME,Number=1,Type=String,Description="Phase In-frame'
        ' or out-frame mutation">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PMN,Number=1,Type=String,Description="Phase Mutation'
        ' Allele Number">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PREGION,Number=1,Type=String,Description="Phase STR'
        ' region or Flanking region">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PMP,Number=1,Type=String,Description="Phase Max Mosaic'
        ' Genotypes posterior">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PAMP,Number=1,Type=String,Description="Phase All Mosaic'
        ' Genotypes posterior">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PGQ,Number=1,Type=String,Description="Phase'
        ' Log-likelihood ratio test of max mosaic GT vs second mosaic GT">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PGG,Number=1,Type=String,Description="Phase'
        ' Log-likelihood ratio test of max mosaic GT vs germline mosaic GT">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PMF,Number=1,Type=String,Description="Phase Mosaic'
        ' fraction">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PPSTR,Number=2,Type=String,Description="Phase Perfect'
        ' STR for mutated allele and mutation allele">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PDP,Number=1,Type=String,Description="Phase Available'
        ' Read Depth">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PFDP,Number=1,Type=String,Description="Phase Filtered'
        ' out Read Depth">'
        + "\n"
    )
    # header += '##FORMAT=<ID=FPD,Number=1,Type=String,Description="Read Depth for all false positive alleles">' + '\n'
    # header += '##FORMAT=<ID=FPN,Number=1,Type=String,Description="False positive allele number">' + '\n'
    header += (
        '##FORMAT=<ID=PEAF,Number=2,Type=String,Description="Phase expected'
        ' allele frequency of mosaic allele">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PEAD,Number=2,Type=String,Description="Phase expected'
        ' allele counts of mosaic allele">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PAAD,Number=N,Type=String,Description="Phase All allele'
        ' counts">'
        + "\n"
    )
    # header += '##FORMAT=<ID=MAF,Number=2,Type=String,Description="Mosaic cells allele frequency">' + '\n'
    header += (
        '##FORMAT=<ID=PNSTUTTER,Number=1,Type=String,Description="Phase'
        ' Number of stutter alleles">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PDSTUTTER,Number=1,Type=String,Description="Phase'
        ' Number of reads with a stutter indel in the STR region">'
        + "\n"
    )
    header += (
        '##FORMAT=<ID=PFILTER,Number=1,Type=String,Description="Phase Reason'
        " for filtering the current call, or PASS if the call was not"
        ' filtered">'
        + "\n"
    )
    header = header + contigs
    header += (
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t%s"
        % (sample_name_list_string)
        + "\n"
    )
    with open(output_vcf, "w") as vcf_out:
        # try:
        #     if acquire_lock(vcf_out):
        vcf_out.write(header)
        # finally:
        #     release_lock(vcf_out)


# MBP
# PHA
# UPR
## PS
## NSNP
## HVAF
# GERMLR
# GERMLRTP
# PPP
# PODD
# PODR
# PODH
# POAH
# PODHR
# PMAP
# PMLEAH
# PMLEEH
# PMLEEHR
# PMLEDD
# PMLEDR
# PFAH
## PEAD
## PEAF
# MAPLR
# MAPLRTP
## PP


def write_STR_vcf(samples_dict, locus_info, variant_info, output_vcf):
    chrom = locus_info["chr"]
    pos = locus_info["str_zero_based_start_included"]
    str_id = locus_info["STR_id"]
    if type(variant_info["ref"]) == int:
        ref = str(variant_info["ref"])
    else:
        ref = "".join(list(variant_info["ref"]))
    assert ref != ""
    alt = variant_info["alt"]
    # alt_variant_str = variant_info["alts_alignment_based"]
    qual = 100
    filter = variant_info["filter"]  # True or False Positive or Germline
    # region = variant_info["region"] # STR or Flanking
    start = locus_info["str_zero_based_start_included"]
    end = locus_info["str_zero_based_end_excluded"]
    motif = locus_info["motif"]
    period = locus_info["period"]
    str_total_length = locus_info["str_total_length"]
    zygosity = locus_info["ploidy"]
    allele_num = variant_info["allele_num"]
    # variant_type_list = variant_info["variant_type"] ## SNV or INDEL
    # variant_type = ','.join(variant_type_list)
    # mutation_type_per_site = variant_info.get("mutation_type_per_site",0)
    altered_repeat_number_list = variant_info["altered_repeat_number"]
    altered_repeat_number_list = list(
        map(lambda x: str(x), altered_repeat_number_list)
    )
    altered_repeat_number = ",".join(altered_repeat_number_list)
    altered_base_pairs_list = variant_info["altered_base_pairs"]
    altered_base_pairs_list = list(
        map(lambda x: str(x), altered_base_pairs_list)
    )
    all_samples_hSNP_INDEL_num = variant_info["all_samples_hSNP_INDEL_num"]
    altered_base_pairs = ",".join(altered_base_pairs_list)
    stutter_model_params = locus_info["stutter_model_params"]
    stutter_inframe_up = stutter_model_params["inframe_ins_prob"]
    stutter_inframe_down = stutter_model_params["inframe_del_prob"]
    stutter_inframe_pgeom = stutter_model_params["inframe_single_step_prob"]
    stutter_outframe_up = stutter_model_params["outframe_ins_prob"]
    stutter_outframe_down = stutter_model_params["outframe_del_prob"]
    stutter_outframe_pgeom = stutter_model_params["outframe_single_step_prob"]
    # phasing_variants_id = variant_info["phasing_variants_id"]
    # phasing_location = variant_info["phasing_location"] # SNP locations
    output_item = (
        f"{chrom}\t{pos}\t{str_id}\t{ref}\t{alt}\t{qual}\t{filter}\tSTRID={str_id};"
        + f"STRSTART={start};STREND={end};MOTIF={motif};PERIOD={period};RLEN={str_total_length};"
        + f"ZYGOSITY={zygosity};AN={allele_num};"
        + f"RNDIFFS={altered_repeat_number};BPDIFFS={altered_base_pairs};STUTTER_INFRAME_UP={stutter_inframe_up};"
        + f"STUTTER_INFRAME_DOWN={stutter_inframe_down};STUTTER_INFRAME_PGEOM={stutter_inframe_pgeom};"
        + f"STUTTER_OUTFRAME_UP={stutter_outframe_up};STUTTER_OUTFRAME_DOWN={stutter_outframe_down};STUTTER_OUTFRAME_PGEOM={stutter_outframe_pgeom};ALL_HVAR_NUM={all_samples_hSNP_INDEL_num}"
        + f"\tGT:MGT:VARIANTTYPE:MUTP:FRAME:MN:REGION:MP:CMP:AMP:GQ:GG:MF:RMF:PSTR:DP:FDP:EAF:EAD:AAD:EMLEAD:EMLEAF:MLEUAF:"
        + f"NSTUTTER:DSTUTTER:NFSTUTTER:DFSTUTTER:NALLELES:GI:GIP:GIQ:GIQP:FILTER:PS:NSNP:HVAF:PP:PQ:PGT:PMGT:PVARIANTTYPE:PMUTP:PFRAME:PMN:PREGION:PMP:PAMP:PGQ:PGG:PMF:PPSTR:PDP:PFDP:PEAF:PEAD:PAAD:PNSTUTTER:PDSTUTTER:PFILTER:"
        + f"MBP:PHA:UPR:GERMLR:GERMLRTP:PPP:PODD:PODR:PODH:POAH:PODHR:PMAP:PMLEAH:PMLEEH:PMLEEHR:PMLEDD:PMLEDR:PFAH:MAPLR:MAPLRTP:MLEPEAD:MLEDISHD:MLEPUAF:MLEPALLAD:MLEPALLAL:PDOAD:PDOADL:POMDD:POMDR:PSOMDD:PSOMDR:PGOMDD:PGOMDR:PMMDD:PMMDR:PSMMDD:PSMMDR:PGMMDD:PGMMDR:NHSNPN:NHSNPINDELN:MAPLRS:MAPLRTPS:FDPC:GIOAF:GIMLEAF:GIUAF:PRHN:OPALLAD:MLEPEAF:PMB:POB:PGHAF:PMHAF:PH3AF:PGD:PSD:PMD:ALLSINGLE:HPRHN:obs_phase_state:obs_hap_state:obs_hap_count:mle_phase_state:mle_hap_state:mle_hap_count:NORMGERM1:NORMGERM2:NORMMOSAIC:NORMSECOND:SECISGERM:obs_mut_order:obs_depth_string:mle_mut_order:mle_depth_string\t"
    )
    for sample_name, variant_dict in samples_dict.items():
        # g_or_s_type = variant_dict["g_or_s_type"] ## germline or somatic
        # GT = variant_dict["GT"]
        # sample_dp = variant_dict["sample_dp"]
        # eaf = variant_dict["eaf"]
        # ead = variant_dict["ead"]
        # mosaic_fraction = variant_dict["mosaic_fraction"]
        # mosaic_VAF = variant_dict["mosaic_VAF"]
        # stutter_allele_num = variant_dict["stutter_allele_num"]
        # stutter_allele_depth = variant_dict["stutter_allele_dp"]
        # stutter_in_frame_u=variant_dict["stutter_in_frame_u"]
        # stutter_in_frame_d=variant_dict["stutter_in_frame_d"]
        # stutter_in_frame_p=variant_dict["stutter_in_frame_p"]
        # filter_type=variant_dict["filter"] # /true positive/false positive/phasing_or_not_or_unknown/germline/somatic_or_non_somatic/
        # output_item = output_item + f"{g_or_s_type}:{GT}:{sample_dp}:{eaf}:{ead}:{mosaic_fraction}:{mosaic_VAF}:{stutter_allele_num}:{stutter_allele_depth}:{stutter_in_frame_u}:"+\
        # f"{stutter_in_frame_d}:{stutter_in_frame_p}:{filter_type}\t"
        # output_item = (
        #     output_item
        #     + f'{variant_dict["GT"]}:{variant_dict["MGT"]}:{variant_dict["VARIANTTYPE"]}:{variant_dict["MUTP"]}:{variant_dict["FRAME"]}:{variant_dict["MN"]}:{variant_dict["REGION"]}:{variant_dict["MP"]}:{variant_dict["AMP"]}:{variant_dict["GQ"]}:'
        #     + f'{variant_dict["GG"]}:{variant_dict["MF"]}:{variant_dict["PSTR"]}:{variant_dict["DP"]}:{variant_dict["FDP"]}:{variant_dict["EAF"]}:{variant_dict["EAD"]}:{variant_dict["AAD"]}:'
        #     + f'{variant_dict["NSTUTTER"]}:{variant_dict["DSTUTTER"]}:{variant_dict["FILTER"]}:{variant_dict["PS"]}:{variant_dict["NSNP"]}:{variant_dict["HVAF"]}:{variant_dict["PP"]}:{variant_dict["PQ"]}:'
        #     + f'{variant_dict["PGT"]}:{variant_dict["PMGT"]}:{variant_dict["PVARIANTTYPE"]}:{variant_dict["PMUTP"]}:{variant_dict["PFRAME"]}:{variant_dict["PMN"]}:{variant_dict["PREGION"]}:{variant_dict["PMP"]}:{variant_dict["PAMP"]}:{variant_dict["PGQ"]}:'
        #     + f'{variant_dict["PGG"]}:{variant_dict["PMF"]}:{variant_dict["PPSTR"]}:{variant_dict["PDP"]}:{variant_dict["PFDP"]}:{variant_dict["PEAF"]}:{variant_dict["PEAD"]}:{variant_dict["PAAD"]}:'
        #     + f'{variant_dict["PNSTUTTER"]}:{variant_dict["PDSTUTTER"]}:{variant_dict["PFILTER"]}\t'
        # )
        aad = variant_dict.get("AAD", ".")
        if aad in [".", ""]:
            variant_dict["sim_AAD"] = "."
        else:
            aad_list = aad.split(",")
            sim_aad = ";".join(
                f"{i}|{dp}" for i, dp in enumerate(aad_list) if int(dp) != 0
            )
            variant_dict["sim_AAD"] = sim_aad if sim_aad else "."

        output_item = (
            output_item
            + f'{variant_dict.get("GT",".")}:{variant_dict.get("MGT",".")}:{variant_dict.get("VARIANTTYPE",".")}:{variant_dict.get("MUTP",".")}:{variant_dict.get("FRAME",".")}:{variant_dict.get("MN",".")}:{variant_dict.get("REGION",".")}:{variant_dict.get("MP",".")}:{variant_dict.get("CMP",".")}:{variant_dict.get("AMP",".")}:{variant_dict.get("GQ",".")}:'
            + f'{variant_dict.get("GG",".")}:{variant_dict.get("MF",".")}:{variant_dict.get("MF_hom2het_het2het",".")}:{variant_dict.get("PSTR",".")}:{variant_dict.get("DP",".")}:{variant_dict.get("FDP",".")}:{variant_dict.get("EAF",".")}:{variant_dict.get("EAD",".")}:{variant_dict.get("sim_AAD",".")}:{variant_dict.get("EMLEAD",".")}:{variant_dict.get("EMLEAF",".")}:{variant_dict.get("MLEUAF",".")}:'
            + f'{variant_dict.get("NSTUTTER",".")}:{variant_dict.get("DSTUTTER",".")}:{variant_dict.get("NFSTUTTER",".")}:{variant_dict.get("DFSTUTTER",".")}:{variant_dict.get("NALLELES",".")}:{variant_dict.get("GI",".")}:{variant_dict.get("GIP",".")}:{variant_dict.get("GIQ",".")}:{variant_dict.get("GIQP",".")}:{variant_dict.get("FILTER",".")}:{variant_dict.get("PS",".")}:{variant_dict.get("NSNP",".")}:{variant_dict.get("HVAF",".")}:{variant_dict.get("PP",".")}:{variant_dict.get("PQ",".")}:'
            + f'{variant_dict.get("PGT",".")}:{variant_dict.get("PMGT",".")}:{variant_dict.get("PVARIANTTYPE",".")}:{variant_dict.get("PMUTP",".")}:{variant_dict.get("PFRAME",".")}:{variant_dict.get("PMN",".")}:{variant_dict.get("PREGION",".")}:{variant_dict.get("PMP",".")}:{variant_dict.get("PAMP",".")}:{variant_dict.get("PGQ",".")}:'
            + f'{variant_dict.get("PGG",".")}:{variant_dict.get("PMF",".")}:{variant_dict.get("PPSTR",".")}:{variant_dict.get("PDP",".")}:{variant_dict.get("PFDP",".")}:{variant_dict.get("PEAF",".")}:{variant_dict.get("PEAD",".")}:{variant_dict.get("PAAD",".")}:'
            + f'{variant_dict.get("PNSTUTTER",".")}:{variant_dict.get("PDSTUTTER",".")}:{variant_dict.get("PFILTER",".")}:{variant_dict.get("MBP",".")}:{variant_dict.get("PHA",".")}:{variant_dict.get("UPR",".")}:{variant_dict.get("GERMLR",".")}:'
            + f'{variant_dict.get("GERMLRTP",".")}:{variant_dict.get("PPP",".")}:{variant_dict.get("PODD",".")}:{variant_dict.get("PODR",".")}:{variant_dict.get("PODH",".")}:{variant_dict.get("POAH",".")}:{variant_dict.get("PODHR",".")}:{variant_dict.get("PMAP",".")}:{variant_dict.get("PMLEAH",".")}:{variant_dict.get("PMLEEH",".")}:'
            + f'{variant_dict.get("PMLEEHR",".")}:{variant_dict.get("PMLEDD",".")}:{variant_dict.get("PMLEDR",".")}:{variant_dict.get("PFAH",".")}:{variant_dict.get("MAPLR",".")}:{variant_dict.get("MAPLRTP",".")}:{variant_dict.get("MLEPEAD",".")}:{variant_dict.get("MLEDISHD",".")}:{variant_dict.get("MLEPUAF")}:{variant_dict.get("MLEPALLAD",".")}:{variant_dict.get("MLEPALLAL",".")}:{variant_dict.get("PDOAD",".")}:{variant_dict.get("PDOADL",".")}:'  # XXX: \t for all samples
            + f'{variant_dict.get("POMDD",".")}:{variant_dict.get("POMDR",".")}:{variant_dict.get("PSOMDD",".")}:{variant_dict.get("PSOMDR",".")}:{variant_dict.get("PGOMDD",".")}:{variant_dict.get("PGOMDR",".")}:{variant_dict.get("PMMDD",".")}:{variant_dict.get("PMMDR",".")}:{variant_dict.get("PSMMDD",".")}:{variant_dict.get("PSMMDR",".")}:{variant_dict.get("PGMMDD",".")}:{variant_dict.get("PGMMDR",".")}:{variant_dict.get("NHSNPN",".")}:{variant_dict.get("NHSNPINDELN",".")}:{variant_dict.get("MAPLRS",".")}:{variant_dict.get("MAPLRTPS",".")}:{variant_dict.get("FDPC",".")}:{variant_dict.get("GIOAF",".")}:{variant_dict.get("GIMLEAF",".")}:{variant_dict.get("GIUAF",".")}:'
            + f'{variant_dict.get("PRHN",".")}:{variant_dict.get("OPALLAD",".")}:{variant_dict.get("MLEPEAF",".")}:{variant_dict.get("PMB",".")}:{variant_dict.get("POB",".")}:{variant_dict.get("PGHAF",".")}:{variant_dict.get("PMHAF",".")}:{variant_dict.get("PH3AF",".")}:{variant_dict.get("PGD",".")}:{variant_dict.get("PSD",".")}:{variant_dict.get("PMD",".")}:{variant_dict.get("ALLSINGLE",".")}:{variant_dict.get("HPRHN",".")}:{variant_dict.get("obs_phase_state",".")}:{variant_dict.get("obs_hap_state",".")}:{variant_dict.get("obs_hap_count",".")}:{variant_dict.get("mle_phase_state",".")}:{variant_dict.get("mle_hap_state",".")}:{variant_dict.get("mle_hap_count",".")}:{variant_dict.get("NORMGERM1",".")}:{variant_dict.get("NORMGERM2",".")}:{variant_dict.get("NORMMOSAIC",".")}:{variant_dict.get("NORMSECOND",".")}:{variant_dict.get("SECISGERM",".")}:'
            + f'{variant_dict.get("obs_mut_order",".")}:{variant_dict.get("obs_depth_string",".")}:{variant_dict.get("mle_mut_order",".")}:{variant_dict.get("mle_depth_string",".")}\t'
        )
    output_item = output_item.strip()
    output_item = output_item + "\n"
    with open(output_vcf, "a") as vcf_out:
        try:
            if acquire_lock(vcf_out):
                vcf_out.write(output_item)
        finally:
            release_lock(vcf_out)
