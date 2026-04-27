from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


class CooccurrenceBackend(Protocol):
    """
    Backend opcional para replicar el PNPMI del paper.

    El paper usa una base grande de co-ocurrencias por párrafos y calcula PNPMI.
    Aquí definimos una interfaz para poder enchufar una implementación real.
    """

    def pnpmi(self, a: str, b: str) -> float:  # pragma: no cover
        raise NotImplementedError


class NullCooccurrenceBackend:
    """Placeholder: no hay datos de co-ocurrencia (PNPMI=0)."""

    def pnpmi(self, a: str, b: str) -> float:
        return 0.0


class SqliteCooccurrenceBackend:
    """
    Backend de co-ocurrencia basado en SQLite.

    Modelo (replicable):
    - Contexto = párrafo (separado por líneas en blanco) en un corpus.
    - count(term) = nº de párrafos donde aparece el término (presencia/ausencia).
    - count(term1,term2) = nº de párrafos donde aparecen ambos.
    - PNPMI = max(0, NPMI), con:
        PMI = log2(p(a,b)/(p(a)p(b)))
        NPMI = PMI / -log2(p(a,b))
      donde p(x)=count(x)/N, p(a,b)=count(a,b)/N y N = nº de párrafos.

    Para MWEs (términos multi-palabra): aproxima como en el paper,
    promediando PNPMI entre palabras constituyentes.
    """

    def __init__(self, db_path: str | Path):
        import sqlite3

        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        try:
            tables = {
                r["name"]
                for r in self.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        except sqlite3.DatabaseError as e:
            raise ValueError(
                f"DB de co-ocurrencia inválida (no es SQLite o está corrupta): {self.db_path}"
            ) from e

        required = {"meta", "unigram", "bigram"}
        missing = sorted(required - tables)
        if missing:
            raise ValueError(
                "DB de co-ocurrencia inválida: faltan tablas "
                f"{', '.join(missing)} en {self.db_path}. "
                "Regenera la DB con `python3 scripts/build_cooc_db_es.py --out ... --inputs ...`."
            )

        row = self.conn.execute("SELECT value FROM meta WHERE key='N'").fetchone()
        if row is None:
            raise ValueError("DB de co-ocurrencia inválida: falta meta.N")
        self.N = float(row["value"])

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _count1(self, t: str) -> float:
        row = self.conn.execute("SELECT c FROM unigram WHERE term=?", (t,)).fetchone()
        return float(row["c"]) if row else 0.0

    def _count2(self, a: str, b: str) -> float:
        t1, t2 = (a, b) if a <= b else (b, a)
        row = self.conn.execute(
            "SELECT c FROM bigram WHERE term1=? AND term2=?",
            (t1, t2),
        ).fetchone()
        return float(row["c"]) if row else 0.0

    def _pnpmi_single(self, a: str, b: str) -> float:
        import math

        if not a or not b or a == b:
            return 0.0
        ca = self._count1(a)
        cb = self._count1(b)
        cab = self._count2(a, b)
        if ca <= 0 or cb <= 0 or cab <= 0 or self.N <= 0:
            return 0.0
        pa = ca / self.N
        pb = cb / self.N
        pab = cab / self.N
        # seguridad numérica
        if pab <= 0 or pa <= 0 or pb <= 0:
            return 0.0
        pmi = math.log2(pab / (pa * pb))
        npmi = pmi / (-math.log2(pab))
        if npmi <= 0:
            return 0.0
        return float(npmi)

    def pnpmi(self, a: str, b: str) -> float:
        a = normalize_term(a.lower())
        b = normalize_term(b.lower())
        if not a or not b:
            return 0.0

        a_parts = a.split(" ")
        b_parts = b.split(" ")
        # Si ambos son unigramas: directo
        if len(a_parts) == 1 and len(b_parts) == 1:
            return self._pnpmi_single(a, b)

        # Para MWEs: promedio de pares palabra-palabra
        vals: list[float] = []
        for aw in a_parts:
            for bw in b_parts:
                v = self._pnpmi_single(aw, bw)
                if v > 0:
                    vals.append(v)
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))


class DifficultyEstimator(Protocol):
    """
    Estimación de "dificultad" de un término para separar formas fácil/difícil.

    En el paper usan VXGL (grade level). Para español, dejamos interfaz.
    """

    def difficulty(self, term: str, lang: str) -> float:  # pragma: no cover
        raise NotImplementedError


class ZipfDifficultyEstimator:
    """
    Heurístico: dificultad ≈ inversa de la frecuencia Zipf.

    - Valores mayores => más difícil.
    - Requiere `wordfreq` (opcional).
    """

    def __init__(self) -> None:
        try:
            from wordfreq import zipf_frequency  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Para usar ZipfDifficultyEstimator instala extras: pip install -e '.[es]'"
            ) from e
        self._zipf_frequency = zipf_frequency

    def difficulty(self, term: str, lang: str) -> float:
        z = float(self._zipf_frequency(term, lang))
        # Zipf típico ~ 1..7 (más alto = más frecuente = más fácil).
        # Convertimos a dificultad (más alto = más difícil).
        return 7.0 - z


@dataclass(frozen=True)
class Lexicon:
    """
    Léxico externo (TOD/NT).

    Para replicar el paper más fielmente, este léxico debería contener
    principalmente sustantivos/NPs y, idealmente, MWEs.
    """

    terms: list[str]

    @staticmethod
    def from_file(path: str | Path) -> "Lexicon":
        p = Path(path)
        raw = p.read_text(encoding="utf-8").splitlines()
        terms: list[str] = []
        for line in raw:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            terms.append(t)
        return Lexicon(terms=terms)

    @staticmethod
    def tiny_default_es() -> "Lexicon":
        # Placeholder mínimo para que el pipeline funcione out-of-the-box.
        # Recomendado: pasar `--lexicon path.txt` con un léxico grande y filtrado.
        return Lexicon(
            terms=[
                "atmósfera",
                "oxígeno",
                "planeta",
                "galaxia",
                "tormenta",
                "célula",
                "energía",
                "ecosistema",
                "fósil",
                "sedimento",
                "electrón",
                "fuerza",
                "temperatura",
                "radiación",
                "migración",
                "carretera",
                "pelota",
                "montaña",
                "cuchara",
                "sillón",
                "cinta",
                "alfombra",
                "ventana",
                "zapato",
            ]
        )


def normalize_term(t: str) -> str:
    return " ".join(t.strip().split())


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out

