#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _norm(s: str) -> str:
    return " ".join(s.strip().split())


def _read_texts(paths: list[str]) -> str:
    buf: list[str] = []
    for p in paths:
        buf.append(Path(p).read_text(encoding="utf-8"))
    return "\n\n".join(buf)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Genera un léxico externo en español (sustantivos + opcionalmente sintagmas nominales) "
            "para usar con ktool-es como reemplazo del placeholder."
        )
    )
    ap.add_argument("--out", required=True, help="Archivo de salida .txt (un término por línea).")
    ap.add_argument(
        "--top-n",
        type=int,
        default=50000,
        help="Cantidad de palabras más frecuentes (wordfreq) a considerar (default: 50000).",
    )
    ap.add_argument(
        "--min-len",
        type=int,
        default=3,
        help="Longitud mínima del término (default: 3).",
    )
    ap.add_argument(
        "--spacy-model",
        default="es_core_news_sm",
        help="Modelo spaCy para filtrar sustantivos (default: es_core_news_sm).",
    )
    ap.add_argument(
        "--add-nps-from",
        nargs="*",
        default=[],
        help="Archivos .txt (UTF-8) desde los que extraer sintagmas nominales (NPs).",
    )
    ap.add_argument(
        "--max-np-len",
        type=int,
        default=6,
        help="Máximo de tokens por NP (default: 6).",
    )
    args = ap.parse_args(argv)

    try:
        from wordfreq import top_n_list  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Falta dependencia `wordfreq`. Instala con: pip install -e '.[es]'"
        ) from e

    import spacy

    nlp = spacy.load(args.spacy_model, disable=["ner", "parser"])

    terms: list[str] = []
    for w in top_n_list("es", args.top_n):
        w = _norm(w.lower())
        if len(w) < args.min_len:
            continue
        # Filtrado POS a sustantivos (evita function words)
        doc = nlp(w)
        if not doc or len(doc) != 1:
            continue
        tok = doc[0]
        if tok.is_stop or tok.is_punct or tok.like_num:
            continue
        if tok.pos_ != "NOUN":
            continue
        terms.append(w)

    # Opcional: extraer NPs desde un mini-corpus (aprox. MWEs nominales)
    if args.add_nps_from:
        nlp_np = spacy.load(args.spacy_model, disable=["ner"])
        text = _read_texts(args.add_nps_from)
        doc = nlp_np(text)
        try:
            noun_chunks = list(doc.noun_chunks)  # type: ignore[attr-defined]
        except Exception:
            noun_chunks = []
        for span in noun_chunks:
            s = _norm(span.text.lower())
            if len(s) < args.min_len:
                continue
            if len(s.split(" ")) > args.max_np_len:
                continue
            if all(t.is_stop or t.is_punct for t in span):
                continue
            terms.append(s)

    # Uniq preservando orden
    seen: set[str] = set()
    out_terms: list[str] = []
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        out_terms.append(t)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_terms) + "\n", encoding="utf-8")
    print(f"OK: {len(out_terms)} términos escritos en {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

