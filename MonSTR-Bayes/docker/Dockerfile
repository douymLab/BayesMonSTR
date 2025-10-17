FROM condaforge/miniforge3:25.3.0-3

RUN mamba install -y -c conda-forge -c bioconda \
    biopython \
    duckdb \
    GPy \
    matplotlib \
    "numpy>=1.8" \
    pandas \
    psutil \
    "pysam>=0.21.0" \
    rich \
    scikit-learn \
    scipy \
    statsmodels \
    tqdm

COPY python /python

COPY scripts /scripts

COPY README.md /

RUN pip install python/
