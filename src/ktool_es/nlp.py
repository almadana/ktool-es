from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TermCandidate:
    # `text` es la representación "humana" (típicamente superficie, sobre todo en MWEs).
    # `key` se usa internamente (embeddings, dedup, PNPMI vs co-ocs basadas en lemas).
    text: str
    key: str
    count: int
    kind: str  # "token" | "np" | "vp"
    pos: str   # spaCy POS tag, o "NP" | "VP"


_WS_RE = re.compile(r"\s+")
_EDGE_PUNCT_RE = re.compile(r"^[\W_]+|[\W_]+$")
_SAFE_TERM_RE = re.compile(r"^[a-záéíóúüñ]+(?: [a-záéíóúüñ]+)*$", re.IGNORECASE)


def _normalize(s: str) -> str:
    s = s.strip()
    s = _WS_RE.sub(" ", s)
    s = _EDGE_PUNCT_RE.sub("", s)
    return s


def _norm(s: str) -> str:
    # Normalización estable para claves internas (principalmente lemas).
    return _normalize(s.lower())


def _is_reasonable_term(s: str) -> bool:
    if not s:
        return False
    if len(s) < 3:
        return False
    if not _SAFE_TERM_RE.match(s):
        return False
    return True


def _mwe_key_for_np_span(span) -> str | None:
    parts: list[str] = []
    for tok in span:
        if tok.is_space or tok.is_punct:
            continue
        if tok.is_stop:
            continue
        if tok.like_num:
            continue
        if tok.pos_ in {"DET", "PRON", "ADP"}:
            continue
        if tok.pos_ == "VERB":
            return None
        if tok.pos_ in {"NOUN", "PROPN", "ADJ"}:
            t = _norm(tok.lemma_.lower())
        elif tok.pos_ in {"ADP", "SCONJ", "CCONJ", "PART"}:
            t = _norm(tok.text.lower())
        else:
            continue
        if len(t) < 1:
            continue
        if not re.match(r"^[a-záéíóúüñ.]+$", t, re.IGNORECASE):
            continue
        parts.append(t)
    s = _normalize(" ".join(parts))
    if len(s) < 3:
        return None
    if not _is_reasonable_term(s):
        return None
    if len(s.split(" ")) < 2:
        return None
    return s


def _mwe_key_for_vp_tokens(toks) -> str | None:
    parts: list[str] = []
    for tok in toks:
        if tok.is_space or tok.is_punct:
            continue
        if tok.like_num:
            continue
        if tok.pos_ in {"AUX", "VERB"}:
            t = _norm(tok.lemma_.lower())
        elif tok.pos_ in {"ADV", "PRON"} and tok.dep_ in {
            "neg",
            "expl",
            "expl:pass",
            "expl:subj",
            "expl:pv",
        }:
            t = _norm(tok.text.lower())
        elif tok.pos_ in {"PART"}:
            t = _norm(tok.text.lower())
        elif tok.pos_ in {"ADP", "SCONJ"} and tok.dep_ in {"mark", "case"}:
            t = _norm(tok.text.lower())
        else:
            # Evita meter sustantivos/objetos u otros adjuntos "largos" a la clave
            continue
        if len(t) < 1:
            continue
        if not re.match(r"^[a-záéíóúüñ.]+$", t, re.IGNORECASE):
            continue
        parts.append(t)
    s = _normalize(" ".join(parts))
    if len(s) < 3:
        return None
    if not _is_reasonable_term(s):
        return None
    if len(s.split(" ")) < 2:
        return None
    return s


def _is_vp_left_token(tok) -> bool:
    # Heurística conservadora: expandir a la izquierda del verbo con piezas
    # típicas de la perifrasis (AUX, negación, clíticos) sin intentar comer
    # el sujeto u otros adjuntos.
    if tok is None:
        return False
    if tok.is_space or tok.is_punct:
        return False
    # En español, negación, auxiliares, pronombres clíticos (se/me/lo/...),
    # y algunas partículas/marcadores entran a menudo en "grupos" contiguos.
    if tok.pos_ in {"AUX", "PART"}:
        return True
    if tok.pos_ in {"ADV", "PRON", "ADP", "SCONJ"} and tok.dep_ in {
        "neg",
        "aux",
        "aux:pass",
        "expl",
        "expl:pass",
        "expl:subj",
        "expl:pv",
    }:
        return True
    return False


def _verbal_phrase_span_for_head(head, *, max_len: int):
    # Cabeza: un VERB (no AUX) para anclar la frase.
    if head.is_space or head.is_punct:
        return None
    if head.pos_ not in {"VERB"}:
        return None
    i = int(head.i)
    # Expande a la izquierda para captar perifrasis ("no ha estudiado", "se ha dado", ...)
    # sin arrastrar el sujeto: solo mientras el token previo "parezca" parte del grupo verbal.
    while i > head.sent.start and _is_vp_left_token(head.doc[i - 1]):
        if (int(head.i) - (i - 1) + 1) > max_len:
            break
        i -= 1
    # A la derecha: piezas inmediatamente contiguas típicas (xcomp/infinitivo, etc.) sin comer
    # el objeto (normalmente se corta con puntuación o con POS "no permitidos").
    r = int(head.i)
    while (r + 1) < head.sent.end:
        t = head.doc[r + 1]
        if t.is_space or t.is_punct:
            break
        if (r + 1 - i + 1) > max_len:
            break
        if t.pos_ in {"AUX", "PART"} or (t.pos_ in {"VERB"} and t.dep_ in {"xcomp", "ccomp", "aux"}):
            r += 1
            continue
        break
    if (r - i + 1) < 2:
        return None
    # Debe contener al menos un VERB; si solo hay AUX+PRON, no lo consideramos.
    toks = [head.doc[k] for k in range(i, r + 1)]
    if not any(t.pos_ == "VERB" for t in toks):
        return None
    if not any(t.pos_ in {"AUX", "VERB"} for t in toks if not t.is_space and not t.is_punct):
        return None
    return head.doc[i : r + 1]


def extract_term_candidates_spacy(
    text: str,
    lang: str = "es",
    spacy_model: str | None = None,
    *,
    include_pos: tuple[str, ...] = ("NOUN", "ADJ", "VERB"),
    include_noun_phrases: bool = True,
    include_verbal_phrases: bool = True,
    max_np_len: int = 6,
    max_vp_len: int = 6,
) -> list[TermCandidate]:
    """
    Extrae candidatos (términos) desde el documento.

    Alineación con el paper:
    - El paper clusteriza términos de contenido (sust/verb/adj) para hallar tópicos,
      pero para el test final usa solo sustantivos y MWEs nominales.
    - Aquí extraemos:
      - tokens con POS en include_pos (para clustering),
      - y además noun phrases (NPs) para aproximar MWEs nominales,
        y (opcional) frases verbales contiguas (AUX+VERB+...) heurísticas.

    Nota: spaCy en español expone `doc.noun_chunks` solo en algunos modelos/pipelines.
    En caso de no estar disponible, seguimos solo con sustantivos como fallback.
    """
    if lang != "es":
        raise ValueError("Por ahora este extractor está pensado para lang='es'.")

    import spacy

    nlp = spacy.load(spacy_model or "es_core_news_sm", disable=["ner"])
    doc = nlp(text)

    counts: dict[tuple[str, str, str, str], int] = {}

    noun_chunks = []

    if include_noun_phrases:
        try:
            noun_chunks = list(doc.noun_chunks)  # type: ignore[attr-defined]
        except Exception:
            noun_chunks = []

    # Aproximación al paper: si un término aparece como parte de un NP (≈MWE),
    # evitamos contar su token “suelto”, salvo cuando aparezca fuera de NPs.
    covered_token_idxs: set[int] = set()
    for span in noun_chunks:
        for tok in span:
            covered_token_idxs.add(tok.i)

    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        if tok.is_stop:
            continue
        if tok.like_num:
            continue
        if tok.pos_ not in include_pos:
            continue
        if tok.i in covered_token_idxs:
            continue

        t = _normalize(tok.lemma_.lower())
        if not _is_reasonable_term(t):
            continue
        key = (t, t, "token", tok.pos_)
        counts[key] = counts.get(key, 0) + 1

    for span in noun_chunks:
        # Canonicalización NP: lemas, sin determinantes/stopwords/puntuación.
        # Restricción: si el span contiene verbos, lo descartamos (evita recortes raros).
        parts: list[str] = []
        bad = False
        for tok in span:
            if tok.is_space or tok.is_punct:
                continue
            if tok.is_stop:
                continue
            if tok.like_num:
                continue
            if tok.pos_ in {"DET", "PRON", "ADP"}:
                continue
            if tok.pos_ == "VERB":
                bad = True
                break
            # HÍBRIDO: para NPs usamos forma superficial (tok.text) limpiada,
            # no lema (evita "inestabilidad" -> "estabilidad", y cambios raros de género).
            surf = _normalize(tok.text.lower())
            if not surf:
                continue
            parts.append(surf)

        if bad:
            continue

        s = _normalize(" ".join(parts))
        if not _is_reasonable_term(s):
            continue
        # Si el NP colapsa a una sola palabra, ya lo capturamos como token.
        if len(s.split(" ")) < 2:
            continue
        if len(s.split(" ")) > max_np_len:
            continue
        k = _mwe_key_for_np_span(span)
        if not k:
            continue
        key = (k, s, "np", "NP")
        counts[key] = counts.get(key, 0) + 1
        for tok in span:
            covered_token_idxs.add(tok.i)

    # Frases verbales (MWE): spans contiguos heurísticos, evitando solape con NPs.
    if include_verbal_phrases:
        seen_spans: set[tuple[int, int]] = set()
        for tok in doc:
            if tok.is_space or tok.is_punct:
                continue
            if tok.i in covered_token_idxs:
                continue
            if tok.pos_ != "VERB":
                continue
            # Evita múltiples cabezas para el mismo span
            vspan = _verbal_phrase_span_for_head(tok, max_len=max_vp_len)
            if vspan is None:
                continue
            a, b = int(vspan.start), int(vspan.end)
            key_span = (a, b)
            if key_span in seen_spans:
                continue
            if any(t.i in covered_token_idxs for t in vspan):
                continue
            parts: list[str] = []
            for t in vspan:
                if t.is_space or t.is_punct:
                    continue
                if t.like_num:
                    continue
                surf = _normalize(t.text.lower())
                if not surf:
                    continue
                parts.append(surf)
            s = _normalize(" ".join(parts))
            if not _is_reasonable_term(s):
                continue
            if len(s.split(" ")) < 2:
                continue
            if len(s.split(" ")) > max_vp_len:
                continue
            k = _mwe_key_for_vp_tokens([x for x in vspan if not x.is_space and not x.is_punct])
            if not k:
                continue
            counts[(k, s, "vp", "VP")] = counts.get((k, s, "vp", "VP"), 0) + 1
            seen_spans.add(key_span)
            for t in vspan:
                covered_token_idxs.add(t.i)

    out: list[TermCandidate] = []
    for (k, t, kind, pos), c in counts.items():
        out.append(TermCandidate(text=t, key=k, count=c, kind=kind, pos=pos))

    # Orden aproximado por frecuencia desc
    out.sort(key=lambda x: (-x.count, x.text))
    return out

