# cli.py
import click
import typer
from typing import Optional




app = typer.Typer(help="BayesMonSTR-ATAC: Tool for mosaic STR mutation calling in snATAC-seq data")




# 定义 Choice 类型
LOG_LEVEL = click.Choice(["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False)
MUT_TYPE = click.Choice(["both", "cell_specific", "share"], case_sensitive=False)

@app.command(help="Run stutter model estimation.")
def stutter(
    metadata: str = typer.Option(..., "--metadata", "-i", help="Metadata CSV file with ind, sex, tissue, bam_path, etc."),
    reference_genome: str = typer.Option(..., "--reference-genome", "-r", help="Reference genome FASTA file"),
    bed_panel: str = typer.Option(..., "--bed-panel", "-b", help="STR genome annotation BED file"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output path for files"),
    loglevel: str = typer.Option("INFO", "--loglevel", "-ll", click_type=LOG_LEVEL, help="Logging threshold"),
    log_to_file: bool = typer.Option(True, "--log-to-file", "-lf", help="Whether save log to file"),
    chrom: str = typer.Option("", "--chrom", "-c", help="Chromosome"),
    start: int = typer.Option(0, "--start", "-s", help="Genomic start position"),
    end: int = typer.Option(1_000_000_000, "--end", "-e", help="Genomic end position"),
    threads: int = typer.Option(11, "--threads", "-t", help="Number of threads to use (default: -1, use all available cores)"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Verbose level"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
):
    """Run stutter model estimation."""
    typer.echo("Running stutter model estimation...")
    import stutter_model_estimation
    stutter_model_estimation.run(
        metadata=metadata, reference_genome=reference_genome, bed_panel=bed_panel,
        output_dir=output_dir, loglevel=loglevel, log_to_file=log_to_file, 
        chrom=chrom, start=start, end=end, threads=threads, verbose=verbose, debug=debug
    )
    typer.echo("✅ Stutter model estimation complete.")




@app.command(help="Remove duplicate reads from snATAC-seq data.")
def deduplicate(
    input_bam: str = typer.Option(..., "--input-bam", "-i", help="Input BAM file"),
    bed_panel: str = typer.Option(..., "--bed-panel", "-b", help="BED file (can be .gz)"),
    output_bam: str = typer.Option(..., "--output-bam", "-o", help="Output sorted BAM file"),
    threads: int = typer.Option(-1, "--threads", "-t", help="Number of threads to use (default: -1, use all available cores)"),
):
    """Remove duplicate reads from snATAC-seq data."""
    typer.echo(f"Deduplicating {input_bam} -> {output_bam}...")
    import deduplicate
    deduplicate.run(input_bam=input_bam, bed_panel=bed_panel, output_bam=output_bam, threads=threads)
    typer.echo("✅ Deduplication complete.")




@app.command(help="Run mosaic genotyping.")
def genotyping(
    metadata: str = typer.Option(..., "--metadata", "-i", help="Metadata CSV file"),
    reference_genome: str = typer.Option(..., "--reference-genome", "-r", help="Reference genome FASTA"),
    bed_panel: str = typer.Option(..., "--bed-panel", "-b", help="STR annotation BED"),
    stutter_model: Optional[str] = typer.Option(None, "--stutter-model", "-s", help="Precomputed stutter model file"),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="Output directory for VCFs"),
    gene_model: Optional[str] = typer.Option(None, "--gene-model", "-g", help="Gene model GTF/GFF for phasing"),
    gnomad_freq_in: Optional[str] = typer.Option(None, "--gnomad-freq-in", "-gn", help="gnomAD population frequency file"),
    phasing: Optional[str] = typer.Option(None, "--phasing", "-p", help="Phased SNP VCF"),
    allele_imbalance: Optional[str] = typer.Option(None, "--allele-imbalance", "-a", help="Allele imbalance from GPR"),
    loglevel: str = typer.Option("INFO", "--loglevel", "-ll", click_type=LOG_LEVEL),
    log_to_file: bool = typer.Option(True, "--log-to-file", "-lf", help="Whether save log to file"),
    chrom: str = typer.Option("", "--chrom", "-c", help="Chromosome"),
    start: int = typer.Option(0, "--start", "-s", help="Genomic start position"),
    end: int = typer.Option(1_000_000_000, "--end", "-e", help="Genomic end position"),
    threads: int = typer.Option(-1, "--threads", "-t", help="Number of threads to use (default: -1, use all available cores)"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
    debug: bool = typer.Option(False, "--debug", "-d"),
):
    """Run mosaic genotyping."""
    typer.echo("Running mosaic genotyping...")
    import mosaic_calling
    mosaic_calling.run(
        metadata=metadata, reference_genome=reference_genome, bed_panel=bed_panel,
        output_dir=output_dir, stutter_model=stutter_model,
        gene_model=gene_model, gnomad_freq_in=gnomad_freq_in,
        phasing=phasing, allele_imbalance=allele_imbalance, loglevel=loglevel, log_to_file=log_to_file,
        chrom=chrom, start=start, end=end, threads=threads,
        verbose=verbose, debug=debug
    )
    typer.echo("✅ Genotyping complete.")




@app.command(help="Extract population information from BulkMonSTR VCF file.")
def pop(
    vcf_file: str = typer.Option(..., "--vcf-file", "-v", help="Input BulkMonSTR VCF file"),
    output_file: str = typer.Option(..., "--output-file", "-o", help="Output file path"),
):
    """Extract population information."""
    typer.echo(f"Extracting population info from {vcf_file} -> {output_file}...")
    import extract_pop_infors
    extract_pop_infors.run(vcf_file=vcf_file, output_file=output_file)
    typer.echo("✅ Population information extracted.")




@app.command(help="Filter genotyping results.")
def filter(
    sample: str = typer.Option(..., "--sample", "-sp", help="Sample name"),
    reference_genome: str = typer.Option(..., "--reference-genome", "-r", help="Reference genome FASTA"),
    vcf: str = typer.Option(..., "--vcf", "-v", help="Input VCF file"),
    stutter_model: str = typer.Option(..., "--stutter-model", "-sm", help="Stutter result BED file"),
    metadata: str = typer.Option(..., "--metadata", "-i", help="Features metadata CSV"),
    mappability: str = typer.Option(..., "--mappability", "-mp", help="Path to a BED file containing mappability information (K24 and K100) for regions in the reference genome."),
    cell_barcode: str = typer.Option(None, "--cell-barcode", "-cb", help="Cell barcode list file"),
    pop_info: str = typer.Option(None, "--pop-info", "-pi", help="Population info file"),
    recurrent_info: str = typer.Option(None, "--recurrent-info", "-ri", help="Recurrent mosaic info file"),
    chrom: str = typer.Option(None, "--chrom", "-c", help="Chromosome"),
    start: int = typer.Option(0, "--start", "-s", help="Genomic start position"),
    end: int = typer.Option(1_000_000_000, "--end", "-e", help="Genomic end position"),
    filters_json: str = typer.Option(None, "--filters-json", "-fj", help="Json file for filtering thresholds. Default path is src/filters.json."),
    mutation_type: str = typer.Option("both", '--mutation-type', "-mt", click_type=MUT_TYPE, help='Expected type of mutation'),
    output_dir: str = typer.Option("./04filter", "--output-dir", "-o", help="Root output directory"),
    plot: bool = typer.Option(False, "--plot", "-p", help='Add --plot in command line to plot count of loci during filtering.'),
    keep_temp: bool = typer.Option(False, "--keep-temp", "-k", help='Add --keep-temp in command line to keep the temporary files.'),

):
    """Filter genotyping results."""
    typer.echo(f"Filtering results for {sample}:{chrom}:{start}-{end}...")
    import filter_per_sample
    filter_per_sample.run(
        sample=sample, reference_genome=reference_genome, vcf=vcf,
        stutter_model=stutter_model, cell_barcode=cell_barcode, pop_info=pop_info,
        recurrent_info=recurrent_info, mappability=mappability, metadata=metadata,
        chrom=chrom, start=start, end=end,
        filters_json=filters_json, mutation_type=mutation_type, output_dir=output_dir, keep_temp=keep_temp, plot=plot
    )
    typer.echo("✅ Filtering complete.")




@app.command(help="Combine chunked results.")
def combine(
    input_dir: str = typer.Option("./04filter", "--input-dir", "-i", help="Root input directory"),
    output_prefix: str = typer.Option("./04filter/results", "--output-prefix", "-o", help="Prefix of outputs"),
    filters_json: str = typer.Option(None, "--filters-json", "-fj", help="Json file for filtering thresholds. Default path is src/filters.json."),
    mutation_type: str = typer.Option("both", '--mutation-type', "-mt", click_type=MUT_TYPE, help='Expected type of mutation')
):
    """Combine chunked results."""
    typer.echo(f"Combine chunked results from {input_dir} -> {output_prefix}...")
    import combine
    combine.run(input_dir=input_dir, output_prefix=output_prefix, filters_json=filters_json, mutation_type=mutation_type)
    typer.echo("✅ Results combined.")



if __name__ == "__main__":
    app()