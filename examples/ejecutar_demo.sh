#!/usr/bin/env bash
set -euo pipefail

# Demo rápida: genera una forma desde examples/ejemplo.txt
# Requiere haber instalado dependencias y el modelo spaCy:
#
#   sudo apt install -y python3-venv python3-pip
#   python3 -m venv .venv
#   source .venv/bin/activate
#   pip install -e ".[es]"
#   python -m spacy download es_core_news_sm
#

ktool-es generate \
  --input examples/ejemplo.txt \
  --lexicon examples/lexicon_es_min.txt \
  --single-form

