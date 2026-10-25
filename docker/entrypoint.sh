#!/usr/bin/env bash
set -euo pipefail

# Conda/micromamba environment names (must match your Dockerfile)
ENV_BAYES="bayesmonstr"     # Python 3.11 env for BayesMonSTR
ENV_ATAC="bayesmonstr-atac"       # ATAC env created from environment.yml
IMAGE_VERSION="${IMAGE_VERSION:-unknown}"

print_help() {
  cat <<EOF
BayesMonSTR Docker image
Version: ${IMAGE_VERSION}

Usage:
  bayesmonstr ...        Run BayesMonSTR
  bayesmonstr-atac ...   Run BayesMonSTR-ATAC
  python ...             Run Python in the BayesMonSTR environment
  bash / sh              Open a shell

Examples:
  docker run --rm bayesmonstr bayesmonstr --help
  docker run --rm bayesmonstr bayesmonstr-atac --help
  docker run --rm bayesmonstr python -c 'import pandas; print(pandas.__version__)'
  docker run --rm -it bayesmonstr
EOF
}

# If no args, open an interactive shell
if [ "$#" -eq 0 ]; then
  exec bash
fi

cmd="$1"
shift || true

case "$cmd" in
  bayesmonstr)
    exec micromamba run -n "${ENV_BAYES}" bayesmonstr "$@"
    ;;
  bayesmonstr-atac)
    exec micromamba run -n "${ENV_ATAC}" bayesmonstr-atac "$@"
    ;;
  python|python3)
    exec micromamba run -n "${ENV_BAYES}" python "$@"
    ;;
  help|-h|--help)
    print_help
    exit 0
    ;;
  bash|sh)
    exec "$cmd" "$@"
    ;;
  *)
    echo "ERROR: Unknown command: $cmd" >&2
    echo >&2
    print_help >&2
    exit 1
    ;;
esac