# =========================================================
# Dockerfile for BayesMonSTR + BayesMonSTR-ATAC
# =========================================================

# Use a lightweight Conda-compatible base image
FROM mambaorg/micromamba:1.5.10

# Use bash for environment activation
SHELL ["/bin/bash", "-lc"]

# Set working directory inside the container
WORKDIR /opt/BayesMonSTR

# Copy the entire repository into the container
COPY . /opt/BayesMonSTR

# Build argument: Conda environment name
# Must match the "name:" field in BayesMonSTR-ATAC/environment.yml
ARG ATAC_ENV=BayesMonSTR-ATAC

# Create the Conda environment from environment.yml
RUN micromamba create -y -f BayesMonSTR-ATAC/environment.yml && \
    micromamba clean -a -y

# Install both BayesMonSTR and BayesMonSTR-ATAC into the same environment
RUN micromamba run -n ${ATAC_ENV} python -m pip install --upgrade pip && \
    micromamba run -n ${ATAC_ENV} python -m pip install ./BayesMonSTR/python && \
    micromamba run -n ${ATAC_ENV} python -m pip install -e ./BayesMonSTR-ATAC

# Automatically activate the environment when the container starts
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV CONDA_DEFAULT_ENV=${ATAC_ENV}
ENV PATH=/opt/conda/envs/${ATAC_ENV}/bin:$PATH

# Default working directory for users
WORKDIR /work

# Default command
CMD ["bash"]
