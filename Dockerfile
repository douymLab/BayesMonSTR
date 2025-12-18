FROM mambaorg/micromamba:1.5.10

SHELL ["/bin/bash", "-lc"]
WORKDIR /opt/BayesMonSTR
COPY . /opt/BayesMonSTR

# Labels (edit as needed)
LABEL org.opencontainers.image.title="BayesMonSTR"
LABEL org.opencontainers.image.version="1.0"
ENV IMAGE_VERSION=1.0

ARG ENV_BAYES=BayesMonSTR-py311
ARG ENV_ATAC=BayesMonSTR-ATAC

# ATAC env
RUN micromamba create -y -n ${ENV_ATAC} -c conda-forge -f BayesMonSTR-ATAC/environment.yml && \
    micromamba run -n ${ENV_ATAC} python -m pip install --upgrade pip && \
    micromamba run -n ${ENV_ATAC} python -m pip install -e ./BayesMonSTR-ATAC

# Configure channels so python/pip can be resolved
RUN micromamba config prepend channels conda-forge && \
    micromamba config append channels defaults && \
    micromamba config set channel_priority strict

# BayesMonSTR env (Python 3.11)
RUN micromamba create -y -n ${ENV_BAYES} -c conda-forge python=3.11 pip && \
    micromamba run -n ${ENV_BAYES} python -m pip install --upgrade pip && \
    micromamba run -n ${ENV_BAYES} python -m pip install ./BayesMonSTR/python


USER root

COPY docker/bayesmonstr /usr/local/bin/bayesmonstr
COPY docker/bayesmonstr-atac /usr/local/bin/bayesmonstr-atac
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod 0755 /usr/local/bin/bayesmonstr \
               /usr/local/bin/bayesmonstr-atac \
               /usr/local/bin/entrypoint.sh

USER mambauser

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
