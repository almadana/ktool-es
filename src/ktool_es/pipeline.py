from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import AffinityPropagation

from .nlp import TermCandidate, extract_term_candidates_spacy
from .resources import (
    CooccurrenceBackend,
    DifficultyEstimator,
    Lexicon,
    NullCooccurrenceBackend,
    ZipfDifficultyEstimator,
    normalize_term,
    unique_preserve_order,
)


@dataclass(frozen=True)
class PipelineConfig:
    lang: str = "es"
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    spacy_model: str | None = None

    # Targets inspirados en el paper (ajustables)
    form_size: int = 50
    n_tid: int = 14
    n_tod: int = 14
    n_nt: int = 22

    # Pools (oversampling) para permitir múltiples formas
    tid_pool_size: int = 25
    tod_pool_size: int = 80
    nt_pool_size: int = 120

    # Clusters "centrales" (paper: usualmente top 3)
    top_k_clusters: int = 3
    min_cluster_cosine_to_doc: float = 0.5

    # Chunking para documentos largos (paper: promedio de chunks)
    max_words_per_chunk: int = 350

    # Scoring TOD: combinación coseno + PNPMI (placeholder PNPMI=0 si no hay backend)
    alpha_pnpmi: float = 1.0

    # MWE: extracción y filtrado
    include_noun_phrases: bool = True
    include_verbal_phrases: bool = True
    max_np_len: int = 6
    max_vp_len: int = 6

    # Léxico (TOD/NT): acepta MWEs compuestas nominales y/o verbales (heurístico)
    mwe_allow_nominal: bool = True
    mwe_allow_verbal: bool = True

    # TOD: cortes opcionales (si quedan muy laxamente vinculados)
    # - min_support_score: score mínimo final (avg sobre anclas)
    # - min_anchor_cosine: exige que el término tenga al menos un coseno >= umbral con alguna ancla
    # - min_anchor_pnpmi: exige que el término tenga al menos un PNPMI >= umbral con alguna ancla
    # Defaults = sin cortes.
    tod_min_support_score: float = -1e9
    tod_min_anchor_cosine: float = -1.0
    tod_min_anchor_pnpmi: float = -1.0


@dataclass(frozen=True)
class GeneratedForm:
    difficulty_label: str  # "easy" | "hard" | "single"
    tid: list[str]
    tod: list[str]
    nt: list[str]


@dataclass(frozen=True)
class GenerationResult:
    config: PipelineConfig
    topic_clusters: list[dict]
    tid_pool: list[str]
    tod_pool: list[str]
    nt_pool: list[str]
    forms: list[GeneratedForm]
    diagnostics: dict | None = None

    def to_json(self) -> str:
        obj = {
            "config": self.config.__dict__,
            "topic_clusters": self.topic_clusters,
            "pools": {
                "tid": self.tid_pool,
                "tod": self.tod_pool,
                "nt": self.nt_pool,
            },
            "forms": [
                {
                    "difficulty": f.difficulty_label,
                    "TID": f.tid,
                    "TOD": f.tod,
                    "NT": f.nt,
                }
                for f in self.forms
            ],
        }
        if self.diagnostics is not None:
            obj["diagnostics"] = self.diagnostics
        return json.dumps(obj, ensure_ascii=False, indent=2)


def _chunk_by_words(text: str, max_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks: list[str] = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i : i + max_words]))
    return chunks


class Embedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(emb, dtype=np.float32)


def embed_document(embedder: Embedder, text: str, max_words_per_chunk: int) -> np.ndarray:
    chunks = _chunk_by_words(text, max_words=max_words_per_chunk)
    embs = embedder.embed_texts(chunks)
    doc_vec = embs.mean(axis=0)
    # L2 normalize
    denom = float(np.linalg.norm(doc_vec)) or 1.0
    return (doc_vec / denom).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def build_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    # vectors ya vienen normalizados => coseno = dot
    sim = vectors @ vectors.T
    return np.asarray(sim, dtype=np.float32)


def affinity_propagation_clusters(vectors: np.ndarray) -> list[list[int]]:
    if len(vectors) == 0:
        return []
    if len(vectors) == 1:
        return [[0]]

    sim = build_similarity_matrix(vectors)
    # AffinityPropagation espera una matriz de "similaridad" (no distancia)
    ap = AffinityPropagation(affinity="precomputed", random_state=0, damping=0.8)
    labels = ap.fit_predict(sim)

    clusters: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(idx)
    return list(clusters.values())


def centroid(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = weights.reshape(-1, 1).astype(np.float32)
    c = (vectors * w).sum(axis=0) / (float(w.sum()) or 1.0)
    denom = float(np.linalg.norm(c)) or 1.0
    return (c / denom).astype(np.float32)


def _term_kind_for_test(c: TermCandidate, *, config: PipelineConfig) -> bool:
    # En el paper: el test se centra en material nominal; aquí además soportamos
    # MWEs verbales (frases) si están habilitadas.
    if c.kind == "np":
        return True
    if c.kind == "vp":
        return bool(config.include_verbal_phrases)
    # Tokens: restringimos a sustantivos (evita verbos sueltos como "estudiar")
    return c.kind == "token" and c.pos == "NOUN"


def generate(
    text: str,
    *,
    config: PipelineConfig,
    lexicon: Lexicon | None = None,
    cooc: CooccurrenceBackend | None = None,
    difficulty: DifficultyEstimator | None = None,
    make_easy_hard: bool = True,
) -> GenerationResult:
    lexicon = lexicon or Lexicon.tiny_default_es()
    cooc = cooc or NullCooccurrenceBackend()
    difficulty = difficulty or ZipfDifficultyEstimator()

    embedder = Embedder(config.model_name)
    doc_vec = embed_document(embedder, text, max_words_per_chunk=config.max_words_per_chunk)

    candidates = extract_term_candidates_spacy(
        text,
        lang=config.lang,
        spacy_model=config.spacy_model,
        include_noun_phrases=config.include_noun_phrases,
        include_verbal_phrases=config.include_verbal_phrases,
        max_np_len=config.max_np_len,
        max_vp_len=config.max_vp_len,
    )
    if not candidates:
        return GenerationResult(
            config=config,
            topic_clusters=[],
            tid_pool=[],
            tod_pool=[],
            nt_pool=[],
            forms=[],
        )

    # Embeddings y PNPMI: preferimos `key` (lemas / clave lematizada) para alinear con la DB
    # de co-ocurrencias; `text` conserva la superficie para el output del test.
    term_keys = [c.key for c in candidates]
    term_displays = [c.text for c in candidates]
    term_vecs = embedder.embed_texts(term_keys)
    term_counts = np.array([c.count for c in candidates], dtype=np.float32)

    clusters_idx = affinity_propagation_clusters(term_vecs)

    # Centroides y ranking por similitud al documento
    cluster_infos: list[dict] = []
    for cluster in clusters_idx:
        vecs = term_vecs[np.array(cluster)]
        w = term_counts[np.array(cluster)]
        cent = centroid(vecs, w)
        sim = cosine(cent, doc_vec)
        cluster_infos.append(
            {
                "indices": cluster,
                "centroid_cosine_to_doc": sim,
                "terms": [term_displays[i] for i in cluster],
            }
        )

    cluster_infos.sort(key=lambda d: d["centroid_cosine_to_doc"], reverse=True)
    # Restringimos a los primeros K clusters, pero además exigimos un umbral mínimo de similitud
    # al documento para evitar "ruido" cuando hay muchos subtemas/términos.
    top_clusters = [
        c
        for c in cluster_infos[: max(1, config.top_k_clusters)]
        if float(c["centroid_cosine_to_doc"]) >= float(config.min_cluster_cosine_to_doc)
    ]
    # Fallback: si ningún cluster pasa el umbral, usamos el top-1 para no quedarnos sin términos.
    if not top_clusters and cluster_infos:
        top_clusters = [cluster_infos[0]]

    # Pool TID: nominales desde clusters centrales (oversampling)
    tid_terms: list[str] = []
    for cl in top_clusters:
        tid_terms.extend(cl["terms"])
    # Filtra a términos aptos para test (NOUN + NPs).
    allowed = {c.text for c in candidates if _term_kind_for_test(c, config=config)}
    tid_pool = [t for t in unique_preserve_order(tid_terms) if t in allowed]
    tid_pool = tid_pool[: config.tid_pool_size]

    tid_pool_set = set(tid_pool)
    in_doc_set = set(term_displays)

    key_by_display = {c.text: c.key for c in candidates}
    count_by_key: dict[str, int] = {}
    for c in candidates:
        prev = count_by_key.get(c.key)
        if prev is None or int(c.count) > int(prev):
            count_by_key[c.key] = int(c.count)

    # Construye lista de anclas (topical in-document): términos de clusters centrales
    anchors = unique_preserve_order([t for t in tid_terms if t in tid_pool_set])
    anchor_keys = unique_preserve_order([key_by_display[t] for t in anchors if t in key_by_display])
    key_to_idx = {k: i for i, k in enumerate(term_keys)}
    anchor_vecs = {k: term_vecs[key_to_idx[k]] for k in anchor_keys if k in key_to_idx}
    anchor_counts = count_by_key

    # Score para candidatos externos (TOD)
    # Paper: avg_j [cos(tc, tdj) + PNPMI(tc, tdj)] * log10(count(tdj)+1)
    def support(term: str, term_vec: np.ndarray) -> tuple[float, float, float]:
        if not anchor_keys:
            return (0.0, -1.0, -1.0)
        s = 0.0
        max_cos = -1.0
        max_p = -1.0
        for a in anchor_keys:
            av = anchor_vecs.get(a)
            if av is None:
                continue
            cos_s = float(np.dot(term_vec, av))
            p = float(cooc.pnpmi(term, a))
            if cos_s > max_cos:
                max_cos = cos_s
            if p > max_p:
                max_p = p
            w = math.log10(float(anchor_counts.get(a, 1)) + 1.0)
            s += (cos_s + config.alpha_pnpmi * p) * w
        return (s / max(1, len(anchor_keys)), max_cos, max_p)

    # Léxico externo: filtrado básico para evitar verbos/adjetivos sueltos.
    # - Palabras: solo NOUN
    # - Multi-palabras: acepta si no contiene VERB (heurística)
    lex_terms_raw = [normalize_term(t.lower()) for t in lexicon.terms]
    lex_terms_raw = unique_preserve_order(lex_terms_raw)

    def _filter_lex_terms(terms: list[str]) -> list[str]:
        if config.lang != "es":
            return terms
        try:
            import spacy

            nlp_lex = spacy.load(config.spacy_model or "es_core_news_sm", disable=["ner"])
        except Exception:
            return terms

        def mwe_is_nominal_multiword(doc) -> bool:
            # Heurística: MWE "nominal" si predomina el núcleo nominal y no es principalmente verbal.
            toks = [x for x in doc if not x.is_space and not x.is_punct and not x.like_num]
            if len(toks) < 2:
                return False
            has_nouny = any(x.pos_ in {"NOUN", "PROPN"} for x in toks)
            if not has_nouny:
                return False
            # Si trae un verbo finito, lo tratamos como no-nominal (p.ej. oraciones)
            if any(
                (x.pos_ == "VERB" and ("VerbForm" in x.morph))
                for x in toks
            ):
                return False
            if any(x.pos_ in {"AUX"} for x in toks):
                return False
            return True

        def mwe_is_verbal_multiword(doc) -> bool:
            toks = [x for x in doc if not x.is_space and not x.is_punct and not x.like_num]
            if len(toks) < 2:
                return False
            has_verb = any(x.pos_ in {"VERB", "AUX"} for x in toks)
            if not has_verb:
                return False
            # Evita "nominal plana con adj" mal clasificada: exige al menos AUX/VERB+algo más.
            return True

        out: list[str] = []
        for t in terms:
            if not t:
                continue
            if " " not in t:
                doc = nlp_lex(t)
                if len(doc) != 1:
                    continue
                tok = doc[0]
                if tok.is_stop or tok.is_punct or tok.like_num:
                    continue
                if tok.pos_ != "NOUN":
                    continue
                out.append(t)
            else:
                doc = nlp_lex(t)
                ok = False
                if config.mwe_allow_nominal and mwe_is_nominal_multiword(doc):
                    ok = True
                if (not ok) and config.mwe_allow_verbal and mwe_is_verbal_multiword(doc):
                    ok = True
                if ok:
                    out.append(t)
        return out

    lex_terms = _filter_lex_terms(lex_terms_raw)
    lex_terms = [t for t in lex_terms if t and t not in in_doc_set]
    if not lex_terms:
        tod_pool = []
        nt_pool = []
        diagnostics = {
            "lexicon_scoring": {
                "n_anchors": int(len(anchor_keys)),
                "n_lex_candidates_after_filter": 0,
            }
        }
    else:
        lex_vecs = embedder.embed_texts(lex_terms)
        supp = [support(t, v) for t, v in zip(lex_terms, lex_vecs)]
        scores = np.array([x[0] for x in supp], dtype=np.float32)
        max_cos = np.array([x[1] for x in supp], dtype=np.float32)
        max_pnpmi = np.array([x[2] for x in supp], dtype=np.float32)

        # Cortes opcionales para TOD: filtra candidatos externos demasiado "laxos".
        keep_mask = (
            (scores >= float(config.tod_min_support_score))
            & (max_cos >= float(config.tod_min_anchor_cosine))
            & (max_pnpmi >= float(config.tod_min_anchor_pnpmi))
        )
        if np.any(keep_mask):
            kept_terms = [t for t, ok in zip(lex_terms, keep_mask) if bool(ok)]
            kept_scores = scores[keep_mask]
            lex_terms = kept_terms
            scores = kept_scores
            max_cos = max_cos[keep_mask]
            max_pnpmi = max_pnpmi[keep_mask]

        # TOD: top por score
        tod_order = np.argsort(-scores)
        tod_pool = [lex_terms[i] for i in tod_order[: config.tod_pool_size]]

        # NT: bottom por score (más "no relacionado")
        nt_order = np.argsort(scores)
        nt_pool = [lex_terms[i] for i in nt_order[: config.nt_pool_size]]

        def _summary(arr: np.ndarray) -> dict:
            if arr.size == 0:
                return {"n": 0}
            a = arr.astype(np.float64)
            qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
            qv = np.percentile(a, qs).tolist()
            return {
                "n": int(a.size),
                "min": float(np.min(a)),
                "max": float(np.max(a)),
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "percentiles": {str(k): float(v) for k, v in zip(qs, qv)},
            }

        idx_by_term = {t: i for i, t in enumerate(lex_terms)}

        def _rows_for_terms(terms: list[str]) -> list[dict]:
            rows: list[dict] = []
            for t in terms:
                i = idx_by_term.get(t)
                if i is None:
                    continue
                rows.append(
                    {
                        "term": t,
                        "support": float(scores[i]),
                        "max_cosine_to_anchor": float(max_cos[i]),
                        "max_pnpmi_to_anchor": float(max_pnpmi[i]),
                    }
                )
            return rows

        tod_idx = np.array([idx_by_term[t] for t in tod_pool if t in idx_by_term], dtype=np.int64)
        nt_idx = np.array([idx_by_term[t] for t in nt_pool if t in idx_by_term], dtype=np.int64)
        diagnostics = {
            "lexicon_scoring": {
                "n_anchors": int(len(anchor_keys)),
                "n_lex_candidates_after_filter": int(len(lex_terms)),
                "cuts": {
                    "min_support": float(config.tod_min_support_score),
                    "min_cos": float(config.tod_min_anchor_cosine),
                    "min_pnpmi": float(config.tod_min_anchor_pnpmi),
                },
                "TOD": {
                    "support_summary": _summary(scores[tod_idx] if tod_idx.size else np.array([], dtype=np.float32)),
                    "max_cos_summary": _summary(max_cos[tod_idx] if tod_idx.size else np.array([], dtype=np.float32)),
                    "max_pnpmi_summary": _summary(
                        max_pnpmi[tod_idx] if tod_idx.size else np.array([], dtype=np.float32)
                    ),
                    "rows": _rows_for_terms(tod_pool),
                },
                "NT": {
                    "support_summary": _summary(scores[nt_idx] if nt_idx.size else np.array([], dtype=np.float32)),
                    "max_cos_summary": _summary(max_cos[nt_idx] if nt_idx.size else np.array([], dtype=np.float32)),
                    "max_pnpmi_summary": _summary(
                        max_pnpmi[nt_idx] if nt_idx.size else np.array([], dtype=np.float32)
                    ),
                    "rows": _rows_for_terms(nt_pool),
                },
            }
        }

    # Ensamblaje de formas: single o easy/hard (según dificultad estimada)
    def pick_by_difficulty(terms: list[str], n: int, easy: bool) -> list[str]:
        if not terms:
            return []
        scored = [(t, float(difficulty.difficulty(t, config.lang))) for t in terms]
        # dificultad mayor => más difícil
        scored.sort(key=lambda x: x[1], reverse=not easy)
        return [t for (t, _) in scored[:n]]

    forms: list[GeneratedForm] = []
    if make_easy_hard:
        forms.append(
            GeneratedForm(
                difficulty_label="easy",
                tid=pick_by_difficulty(tid_pool, config.n_tid, easy=True),
                tod=pick_by_difficulty(tod_pool, config.n_tod, easy=True),
                nt=pick_by_difficulty(nt_pool, config.n_nt, easy=True),
            )
        )
        forms.append(
            GeneratedForm(
                difficulty_label="hard",
                tid=pick_by_difficulty(tid_pool, config.n_tid, easy=False),
                tod=pick_by_difficulty(tod_pool, config.n_tod, easy=False),
                nt=pick_by_difficulty(nt_pool, config.n_nt, easy=False),
            )
        )
    else:
        forms.append(
            GeneratedForm(
                difficulty_label="single",
                tid=tid_pool[: config.n_tid],
                tod=tod_pool[: config.n_tod],
                nt=nt_pool[: config.n_nt],
            )
        )

    return GenerationResult(
        config=config,
        topic_clusters=cluster_infos,
        tid_pool=tid_pool,
        tod_pool=tod_pool,
        nt_pool=nt_pool,
        forms=forms,
        diagnostics=diagnostics,
    )


def generate_from_file(
    input_path: str | Path,
    *,
    config: PipelineConfig,
    lexicon_path: str | Path | None = None,
    cooc_db_path: str | Path | None = None,
    output_path: str | Path | None = None,
    make_easy_hard: bool = True,
) -> GenerationResult:
    p = Path(input_path)
    text = p.read_text(encoding="utf-8")
    lex = Lexicon.from_file(lexicon_path) if lexicon_path else None
    cooc = None
    if cooc_db_path:
        from .resources import SqliteCooccurrenceBackend

        cooc = SqliteCooccurrenceBackend(cooc_db_path)
    res = generate(text, config=config, lexicon=lex, cooc=cooc, make_easy_hard=make_easy_hard)
    if cooc is not None:
        try:
            cooc.close()
        except Exception:
            pass
    if output_path:
        Path(output_path).write_text(res.to_json(), encoding="utf-8")
    return res

