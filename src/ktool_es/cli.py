from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PipelineConfig, generate_from_file


def _fmt_float(x: object) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return ""


def _render_report_markdown(diag: dict) -> str:
    ls = diag.get("lexicon_scoring", {}) if isinstance(diag, dict) else {}
    cuts = ls.get("cuts", {})
    tod = ls.get("TOD", {})
    nt = ls.get("NT", {})

    def _summary_block(name: str, obj: dict) -> str:
        summ = obj.get("support_summary", {})
        cos = obj.get("max_cos_summary", {})
        pnp = obj.get("max_pnpmi_summary", {})
        def q50(s: dict) -> str:
            return _fmt_float(s.get("percentiles", {}).get("50"))
        def q10(s: dict) -> str:
            return _fmt_float(s.get("percentiles", {}).get("10"))
        def q90(s: dict) -> str:
            return _fmt_float(s.get("percentiles", {}).get("90"))

        # IMPORTANTE: terminamos con doble salto para que la tabla siguiente
        # se renderice siempre como tabla Markdown.
        return "\n".join(
            [
                f"## {name}",
                "",
                f"- n: {summ.get('n', 0)}",
                f"- support p10/p50/p90: {q10(summ)} / {q50(summ)} / {q90(summ)}",
                f"- max_cos p10/p50/p90: {q10(cos)} / {q50(cos)} / {q90(cos)}",
                f"- max_pnpmi p10/p50/p90: {q10(pnp)} / {q50(pnp)} / {q90(pnp)}",
                "",
            ]
        ) + "\n"

    def _rows_table(obj: dict) -> str:
        rows = obj.get("rows", []) if isinstance(obj, dict) else []
        out = [
            "| term | support | max_cosine_to_anchor | max_pnpmi_to_anchor |",
            "|---|---:|---:|---:|",
        ]
        for r in rows:
            out.append(
                f"| {r.get('term','')} | {_fmt_float(r.get('support'))} | "
                f"{_fmt_float(r.get('max_cosine_to_anchor'))} | {_fmt_float(r.get('max_pnpmi_to_anchor'))} |"
            )
        return "\n".join(out) + "\n"

    header = [
        "# ktool-es report",
        "",
        f"- n_anchors: {ls.get('n_anchors', 0)}",
        f"- n_lex_candidates_after_filter: {ls.get('n_lex_candidates_after_filter', 0)}",
        f"- cuts: support>={cuts.get('min_support')}  cos>={cuts.get('min_cos')}  pnpmi>={cuts.get('min_pnpmi')}",
        "",
    ]

    return (
        "\n".join(header)
        + _summary_block("TOD", tod)
        + _rows_table(tod)
        + "\n"
        + _summary_block("NT", nt)
        + _rows_table(nt)
    )


def _render_report_tsv(diag: dict) -> str:
    ls = diag.get("lexicon_scoring", {}) if isinstance(diag, dict) else {}
    rows: list[tuple[str, str, str, str, str]] = []
    for label in ("TOD", "NT"):
        obj = ls.get(label, {})
        for r in obj.get("rows", []) if isinstance(obj, dict) else []:
            rows.append(
                (
                    label,
                    str(r.get("term", "")),
                    _fmt_float(r.get("support")),
                    _fmt_float(r.get("max_cosine_to_anchor")),
                    _fmt_float(r.get("max_pnpmi_to_anchor")),
                )
            )
    lines = ["set\tterm\tsupport\tmax_cosine_to_anchor\tmax_pnpmi_to_anchor"]
    lines += ["\t".join(x) for x in rows]
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ktool-es", description="Genera pruebas de vocabulario temático desde un texto.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generar una o dos formas (fácil/difícil) a partir de un pasaje.")
    g.add_argument("--input", required=True, help="Ruta a archivo .txt (UTF-8).")
    g.add_argument("--output", help="Ruta de salida .json (si se omite, imprime a stdout).")
    g.add_argument(
        "--report",
        default=None,
        help="Ruta de salida para un informe JSON con distribuciones de scores (TOD/NT).",
    )
    g.add_argument(
        "--report-format",
        default="md",
        choices=["json", "md", "tsv"],
        help="Formato del informe (--report): json|md|tsv (default: md).",
    )
    g.add_argument("--lang", default="es", help="Idioma (por ahora: es).")
    g.add_argument(
        "--model",
        default=PipelineConfig().model_name,
        help="Modelo SentenceTransformers (por ejemplo paraphrase-multilingual-MiniLM-L12-v2).",
    )
    g.add_argument("--spacy-model", default=None, help="Modelo spaCy (default: es_core_news_sm).")
    g.add_argument(
        "--lexicon",
        default=None,
        help="Archivo con un término por línea (léxico externo para TOD/NT).",
    )
    g.add_argument(
        "--cooc-db",
        default=None,
        help="DB SQLite de co-ocurrencias para PNPMI (opcional).",
    )
    g.add_argument(
        "--alpha-pnpmi",
        type=float,
        default=PipelineConfig().alpha_pnpmi,
        help="Peso del componente PNPMI en el score TOD (default: 1.0).",
    )
    g.add_argument(
        "--tod-min-support",
        type=float,
        default=PipelineConfig().tod_min_support_score,
        help="Corte opcional: score mínimo para TOD (default: sin corte).",
    )
    g.add_argument(
        "--tod-min-cos",
        type=float,
        default=PipelineConfig().tod_min_anchor_cosine,
        help="Corte opcional: exige coseno mínimo con alguna ancla (default: sin corte).",
    )
    g.add_argument(
        "--tod-min-pnpmi",
        type=float,
        default=PipelineConfig().tod_min_anchor_pnpmi,
        help="Corte opcional: exige PNPMI mínimo con alguna ancla (default: sin corte).",
    )
    g.add_argument(
        "--no-np-mwe",
        action="store_true",
        help="No extrae MWEs nominales (noun chunks).",
    )
    g.add_argument(
        "--no-vp-mwe",
        action="store_true",
        help="No extrae MWEs verbales (heurística de perifrasis).",
    )
    g.add_argument(
        "--max-np-len",
        type=int,
        default=PipelineConfig().max_np_len,
        help="Límite de tokens para noun-chunk/MWE nominal (default: 6).",
    )
    g.add_argument(
        "--max-vp-len",
        type=int,
        default=PipelineConfig().max_vp_len,
        help="Límite de tokens para MWE verbal (default: 6).",
    )
    g.add_argument(
        "--lex-mwe-allow",
        default="nominal,verbal",
        help=(
            "Qué MWEs del léxico aceptar, separado por comas: nominal,verbal. "
            "Default: nominal,verbal."
        ),
    )
    g.add_argument(
        "--single-form",
        action="store_true",
        help="Genera una sola forma (sin separar fácil/difícil).",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "generate":
        allow = {p.strip().lower() for p in str(args.lex_mwe_allow).split(",") if p.strip()}
        mwe_allow_nominal = "nominal" in allow
        mwe_allow_verbal = "verbal" in allow
        if not allow.issubset({"nominal", "verbal"}):
            raise SystemExit("Error: --lex-mwe-allow solo admite nominal y/o verbal (separado por comas).")

        cfg = PipelineConfig(
            lang=args.lang,
            model_name=args.model,
            spacy_model=args.spacy_model,
            alpha_pnpmi=float(args.alpha_pnpmi),
            include_noun_phrases=not bool(args.no_np_mwe),
            include_verbal_phrases=not bool(args.no_vp_mwe),
            max_np_len=int(args.max_np_len),
            max_vp_len=int(args.max_vp_len),
            mwe_allow_nominal=mwe_allow_nominal,
            mwe_allow_verbal=mwe_allow_verbal,
            tod_min_support_score=float(args.tod_min_support),
            tod_min_anchor_cosine=float(args.tod_min_cos),
            tod_min_anchor_pnpmi=float(args.tod_min_pnpmi),
        )
        res = generate_from_file(
            args.input,
            config=cfg,
            lexicon_path=args.lexicon,
            cooc_db_path=args.cooc_db,
            output_path=args.output,
            make_easy_hard=not args.single_form,
        )
        if args.report:
            diag = res.diagnostics or {}
            if args.report_format == "json":
                Path(args.report).write_text(res.to_json(), encoding="utf-8")
            elif args.report_format == "md":
                Path(args.report).write_text(_render_report_markdown(diag), encoding="utf-8")
            elif args.report_format == "tsv":
                Path(args.report).write_text(_render_report_tsv(diag), encoding="utf-8")
        if args.output:
            print(f"OK: escrito {args.output}")
        else:
            print(res.to_json())
        return 0

    raise AssertionError("Comando no implementado")


if __name__ == "__main__":
    raise SystemExit(main())

