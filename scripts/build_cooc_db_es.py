#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterator, TextIO


_WS_RE = re.compile(r"\s+")
_EDGE_PUNCT_RE = re.compile(r"^[\W_]+|[\W_]+$")


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = _WS_RE.sub(" ", s)
    s = _EDGE_PUNCT_RE.sub("", s)
    return s


def iter_paragraphs_from_stream(fp: TextIO) -> Iterator[str]:
    # Párrafos = bloques separados por líneas en blanco, leyendo en streaming.
    buf: list[str] = []
    for raw in fp:
        line = raw.rstrip("\n")
        if line.strip() == "":
            if not buf:
                continue
            par = " ".join(s for s in buf if s)
            if par.strip():
                yield par.strip()
            buf = []
            continue
        t = line.strip()
        if t:
            buf.append(t)
    if buf:
        par = " ".join(s for s in buf if s)
        if par.strip():
            yield par.strip()


def iter_sentence_windows_from_stream(
    fp: TextIO, *, window_sentences: int, step_sentences: int
) -> Iterator[str]:
    """
    Construye "pseudo-párrafos" a partir de texto con 1 oración por línea.
    Las líneas en blanco cortan el flujo (no se mezclan ventanas entre bloques).
    """
    if window_sentences <= 0:
        raise ValueError("window_sentences must be > 0")
    if step_sentences <= 0:
        raise ValueError("step_sentences must be > 0")

    def flush_block(lines: list[str]) -> Iterator[str]:
        for i in range(0, len(lines), step_sentences):
            chunk = lines[i : i + window_sentences]
            if len(chunk) < window_sentences:
                return
            yield " ".join(chunk)

    cur_block: list[str] = []
    for raw in fp:
        s = raw.strip()
        if s == "":
            yield from flush_block(cur_block)
            cur_block = []
            continue
        cur_block.append(s)
    yield from flush_block(cur_block)


def tokens_from_paragraph(
    par: str, *, nlp, allowed_pos: set[str] | None = None
) -> set[str]:
    doc = nlp(par)
    terms: set[str] = set()
    for tok in doc:
        if tok.is_space or tok.is_punct or tok.like_num:
            continue
        if tok.is_stop:
            continue
        if allowed_pos is not None and tok.pos_ not in allowed_pos:
            continue
        # Para PNPMI conviene algo estable: usamos lemma para unigrams.
        t = _norm(tok.lemma_)
        if len(t) < 3:
            continue
        # Filtra basura (números, etc.)
        if not re.match(r"^[a-záéíóúüñ]+$", t, re.IGNORECASE):
            continue
        terms.add(t)
    return terms


def _bump_n(cur: "sqlite3.Cursor", delta: int) -> None:
    cur.execute(
        "INSERT INTO meta(key, value) VALUES('N', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=meta.value + excluded.value",
        (float(delta),),
    )


def _apply_unit(cur: "sqlite3.Cursor", terms: set[str]) -> None:
    for t in terms:
        cur.execute(
            "INSERT INTO unigram(term, c) VALUES(?, 1) "
            "ON CONFLICT(term) DO UPDATE SET c=unigram.c + 1",
            (t,),
        )
    ts = sorted(terms)
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            a, b = ts[i], ts[j]
            cur.execute(
                "INSERT INTO bigram(term1, term2, c) VALUES(?, ?, 1) "
                "ON CONFLICT(term1, term2) DO UPDATE SET c=bigram.c + 1",
                (a, b),
            )


def _prune_to_top_unigrams(conn: sqlite3.Connection, *, keep_top: int) -> None:
    """
    Reduce |V| a los `keep_top` términos más frecuentes (por conteos de unigram)
    y elimina pares cuyo extremo haya sido podado. Útil para acotar el tamaño de
    la matriz de co-ocurrencia en corpus enormes.
    """
    if keep_top <= 0:
        raise ValueError("keep_top must be > 0")

    cur = conn.cursor()
    n0 = int(cur.execute("SELECT COUNT(*) FROM unigram").fetchone()[0])
    if n0 <= keep_top:
        return

    # Determinar umbral: el conteo del K-ésimo término más frecuente (con empates
    # incluimos términos con el mismo conteo).
    row = cur.execute(
        "SELECT c FROM unigram ORDER BY c DESC LIMIT 1 OFFSET ?",
        (keep_top - 1,),
    ).fetchone()
    if not row:  # pragma: no cover
        return
    cutoff = int(row[0])
    to_delete = n0 - int(
        cur.execute("SELECT COUNT(*) FROM unigram WHERE c >= ?", (cutoff,)).fetchone()[0]
    )
    if to_delete > 0:
        cur.execute("DELETE FROM unigram WHERE c < ?", (cutoff,))

    # Limpia pares: si falta un extremo, el par no aporta a PNPMI con este vocab.
    # (Evita re-hacer pares: borramo por extremos faltantes.)
    cur.execute(
        """
        DELETE FROM bigram
        WHERE term1 NOT IN (SELECT term FROM unigram)
           OR term2 NOT IN (SELECT term FROM unigram)
        """
    )
    conn.commit()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Construye una DB SQLite de co-ocurrencias (por párrafos) para calcular PNPMI."
        )
    )
    ap.add_argument("--out", required=True, help="Ruta de salida .sqlite")
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Archivos .txt (UTF-8) para construir el corpus.",
    )
    ap.add_argument(
        "--spacy-model",
        default="es_core_news_sm",
        help="Modelo spaCy para tokenizar/lematizar (default: es_core_news_sm).",
    )
    ap.add_argument(
        "--sentence-per-line",
        action="store_true",
        help=(
            "Interpretar el input como 1 oración por línea y agrupar en ventanas de N "
            "oraciones (pseudo-párrafos) para co-ocurrencias."
        ),
    )
    ap.add_argument(
        "--window-sentences",
        type=int,
        default=5,
        help="Tamaño de ventana (en oraciones) si --sentence-per-line (default: 5).",
    )
    ap.add_argument(
        "--step-sentences",
        type=int,
        default=5,
        help=(
            "Paso entre ventanas (en oraciones) si --sentence-per-line. "
            "Default=5 (ventanas no solapadas). Usa 1 para sliding."
        ),
    )
    ap.add_argument(
        "--commit-every",
        type=int,
        default=5000,
        help="Cada cuántas unidades (párrafos/ventanas) hacer COMMIT a SQLite (default: 5000).",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Cada cuántas unidades imprimir progreso a stderr (default: 5000).",
    )
    ap.add_argument(
        "--keep-top-terms",
        type=int,
        default=0,
        help=(
            "Si >0, al final mantiene solo los K términos (lemmas) con mayor conteo de "
            "unigram y recorta pares afectados. Útil para acotar memoria/disco (p.ej. 50000)."
        ),
    )
    ap.add_argument(
        "--pos",
        default="NOUN,PROPN,VERB,AUX,ADJ",
        help=(
            "Lista separada por comas de POS universales (UD) a incluir, según `token.pos_` "
            "de spaCy. Default: NOUN,PROPN,VERB,AUX,ADJ. Para un conjunto más estricto, usá p.ej. "
            "NOUN,VERB o NOUN,VERB,PROPN."
        ),
    )
    args = ap.parse_args(argv)

    import spacy

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    nlp = spacy.load(args.spacy_model, disable=["ner"])
    t0 = time.time()

    allowed_pos = {
        p.strip().upper()
        for p in str(args.pos).split(",")
        if p.strip()
    }
    if not allowed_pos:
        raise ValueError("--pos no puede quedar vacío")

    N = 0

    def iter_units_for_file(path: Path) -> Iterator[str]:
        f = path.open("r", encoding="utf-8", errors="ignore", newline="")
        try:
            if args.sentence_per_line:
                yield from iter_sentence_windows_from_stream(
                    f,
                    window_sentences=args.window_sentences,
                    step_sentences=args.step_sentences,
                )
            else:
                yield from iter_paragraphs_from_stream(f)
        finally:
            f.close()

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(out))
        cur = conn.cursor()
        # Tunings razonables para ingesta larga
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA temp_store=MEMORY")

        cur.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value REAL)")
        cur.execute("INSERT INTO meta(key, value) VALUES('N', 0.0)")
        cur.execute("CREATE TABLE unigram (term TEXT PRIMARY KEY, c INTEGER)")
        cur.execute(
            "CREATE TABLE bigram (term1 TEXT, term2 TEXT, c INTEGER, PRIMARY KEY(term1, term2))"
        )
        # Índices al final, tras ingesta, para no penalizar demasiado los updates.
        print(
            f"START: out={out}  pos={','.join(sorted(allowed_pos))}  "
            f"inputs={', '.join(args.inputs)}",
            file=sys.stderr,
            flush=True,
        )
        conn.commit()
        last_commit = 0

        for fp in args.inputs:
            p = Path(fp)
            print(f"FILE: {p}", file=sys.stderr, flush=True)
            for par in iter_units_for_file(p):
                terms = tokens_from_paragraph(par, nlp=nlp, allowed_pos=allowed_pos)
                if not terms:
                    continue
                _apply_unit(cur, terms)
                _bump_n(cur, 1)
                N += 1

                if args.progress_every > 0 and (N % args.progress_every) == 0:
                    dt = time.time() - t0
                    print(
                        f"PROGRESS: N={N}  file={p.name}  elapsed_s={dt:.1f}",
                        file=sys.stderr,
                        flush=True,
                    )

                if args.commit_every > 0 and (N - last_commit) >= args.commit_every:
                    conn.commit()
                    last_commit = N

        if last_commit != N and args.commit_every > 0:
            conn.commit()
        if args.commit_every <= 0:
            conn.commit()

        if args.keep_top_terms and args.keep_top_terms > 0:
            _prune_to_top_unigrams(conn, keep_top=args.keep_top_terms)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_bigram_1 ON bigram(term1)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bigram_2 ON bigram(term2)")
        conn.commit()
    except BaseException:
        # Si aborta, evita dejar un .sqlite "vacío" engañoso.
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass
        try:
            if out.exists():
                out.unlink()
        except Exception:
            pass
        raise
    else:
        if conn is not None:
            conn.close()

    # |V| y |Pares| (consultas baratas; sqlite COUNT(*))
    conn2 = sqlite3.connect(str(out))
    cur2 = conn2.cursor()
    n_vocab = int(cur2.execute("SELECT COUNT(*) FROM unigram").fetchone()[0])
    n_pairs = int(cur2.execute("SELECT COUNT(*) FROM bigram").fetchone()[0])
    conn2.close()

    print(f"OK: {out}  N(units)={N}  |V|={n_vocab}  |Pares|={n_pairs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

