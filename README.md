# ktool-es (replicación del pipeline de K-tool)

Este repo implementa un **intento reproducible** del pipeline descrito en el artículo *“Towards an automatic method for generating topical vocabulary test forms for specific reading passages”* (Flor et al., 2025), adaptado para **español**.

La meta es: dado un **pasaje** (un texto suelto), **detectar su(s) tópico(s)** y **generar una prueba de vocabulario temático** (términos TID/TOD y distractores NT), usando embeddings tipo BERT y clustering.

## Qué está implementado

- **Vectorización del documento** con Sentence Transformers (modelo configurable).
  - Para textos largos: chunking y promedio de embeddings (como en el paper).
- **Extracción de candidatos del texto**:
  - Tokens de contenido con spaCy (por defecto: `NOUN/ADJ/VERB` para clustering).
  - MWEs nominales aproximadas con `noun_chunks` (según el modelo).
  - MWEs verbales heurísticas (perífrasis contiguas; opcional).
  - Filtrado de stopwords y categorías no útiles.
- **Normalización para scores**:
  - Los ítems conservan una forma “visible” (sobre todo en MWEs), pero internamente se usa una
    **clave lematizada** para alinear **embeddings** y **PNPMI** con la DB de co-ocurrencias (lemas).
- **Clustering de términos del documento** con **Affinity Propagation** sobre similitud coseno.
- **Selección de clusters centrales** por similitud (centroide del cluster vs vector del documento).
- **Construcción de pools y ensamblaje de formas**:
  - TID (topical in-document): desde clusters centrales.
  - TOD (topical out-of-document): desde un léxico externo configurable.
  - NT (non-topical): distractores desde el léxico externo con baja similitud.
- **“Fácil” vs “difícil” (heurístico)**:
  - Sustituye la idea de *grade level* (VXGL) por una aproximación con **frecuencia Zipf** (`wordfreq`) si está disponible.

## Qué queda como placeholder (por falta de recursos del paper)

El paper usa recursos internos/propietarios. Aquí quedan como **interfaces** y están documentados para que puedas enchufarlos:

- **Lexicón de MWEs (68K nominales)**:
  - Placeholder: detección de sintagmas nominales con spaCy.
  - Para replicar más fielmente: añadir un lexicón de MWEs y un detector por lookup.
- **PNPMI / base de co-ocurrencias** (modelo sobre >2B palabras, asociación dentro de párrafos):
  - Implementación básica: podés **construir** una co-oc en SQLite (por unidades de texto) y pasar
    `--cooc-db` para activar el término PNPMI.
  - Lo que *no* replica el paper 1:1: el “corpus gigante de co-oc y parsing” interno; el tamaño/calidad
    depende de **tu** corpus.
- **VXGL (mapeo palabra→grade level)**:
  - Placeholder: dificultad ~ (1 / frecuencia) por Zipf.
  - Para replicar: añadir un recurso español equivalente (o un mapeo calibrado a cursos).

## Instalación

Requisitos: Python 3.10+.

En Ubuntu/Debian, si tu Python no trae `venv`/`pip` (muy común), primero:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
```

Luego:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[es]"
python3 -m spacy download es_core_news_sm
```

## Uso rápido

Generar una forma (y opcionalmente fácil/difícil) a partir de un archivo de texto:

```bash
ktool-es generate --input examples/ejemplo.txt --lang es
```

Elegir modelo de embeddings (opcional):

```bash
ktool-es generate --input ejemplo.txt --model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

Salida: JSON con pools, formas y metadatos.

### MWEs (nominal / verbal) y filtrado del léxico

Por defecto, el generador intenta extraer **MWEs nominales** (`noun_chunks`) y **MWEs verbales**
(heurística). Podés desactivar cada tipo:

```bash
ktool-es generate --input examples/ejemplo.txt --no-np-mwe
ktool-es generate --input examples/ejemplo.txt --no-vp-mwe
```

Para el **léxico externo** (TOD/NT), las entradas multi-palabra se pueden filtrar como MWE
**nominal** y/o **verbal** (heurístico). Default: ambas:

```bash
ktool-es generate --input examples/ejemplo.txt --lexicon data/lexicon_es.txt --lex-mwe-allow nominal,verbal
```

## Generar un léxico en español (reemplazar placeholder)

El paper usa un léxico grande externo (sustantivos + MWEs) para TOD/NT. Aquí puedes generar uno
automáticamente con `wordfreq` + un filtro de POS con spaCy:

```bash
python3 scripts/build_lexicon_es.py --out data/lexicon_es.txt --top-n 50000
```

Opcionalmente, puedes **añadir sintagmas nominales (NPs)** extraídos desde un conjunto de textos
(útil como aproximación de MWEs):

```bash
python3 scripts/build_lexicon_es.py --out data/lexicon_es.txt --top-n 50000 --add-nps-from examples/ejemplo.txt
```

Y luego úsalo así:

```bash
ktool-es generate --input examples/ejemplo.txt --lexicon data/lexicon_es.txt
```

## Construir co-ocurrencias (PNPMI) y usarlas

El paper combina **coseno de embeddings** + **PNPMI** (co-ocurrencia por párrafos). Aquí puedes
construir una base SQLite de co-ocurrencias desde un corpus de `.txt`:

```bash
python3 scripts/build_cooc_db_es.py --out data/cooc_es.sqlite --inputs RUTA1.txt RUTA2.txt
```

Por defecto, la ingesta toma términos cuyo `token.pos_` (etiqueta universal de spaCy) esté
en **`NOUN,PROPN,VERB,AUX,ADJ`**. Si querés excluir nombres propios o ajustar el conjunto, usá
`--pos` (separado por comas), por ejemplo **solo** sustantivos comunes y verbos:

```bash
python3 scripts/build_cooc_db_es.py --out data/cooc_es.sqlite --inputs data/sbwcle.clean.txt --sentence-per-line --pos NOUN,VERB
```

Si tu corpus está **segmentado por oraciones** (1 oración por línea), puedes agrupar en
“pseudo-párrafos” con ventanas de \(N\) oraciones:

```bash
python3 scripts/build_cooc_db_es.py --out data/cooc_es.sqlite --inputs data/sbwcle.clean.txt --sentence-per-line --window-sentences 5 --step-sentences 5
```

En corpus **muy grandes**, el script actualiza SQLite **en streaming** (no acumula conteos
completos en RAM) y hace `COMMIT` periódico. Si querés ver progreso con más frecuencia o
reducir el tamaño de transacción:

```bash
python3 scripts/build_cooc_db_es.py --out data/cooc_es.sqlite --inputs data/sbwcle.clean.txt --sentence-per-line --window-sentences 5 --step-sentences 5 --progress-every 1000 --commit-every 2000
```

Si querés acotar el **vocabulario** (muy común: ~50k–60k términos frecuentes) para bajar
tamaño de disco y de la tabla de pares, podés quedarte con los lemas (unigramas) más
frecuentes y recortar pares afectados al final de la ingesta:

```bash
python3 scripts/build_cooc_db_es.py --out data/cooc_es.sqlite --inputs data/sbwcle.clean.txt --sentence-per-line --window-sentences 5 --step-sentences 5 --keep-top-terms 50000
```

Y activarlo en la generación:

```bash
ktool-es generate --input examples/ejemplo.txt --lexicon data/lexicon_es.txt --cooc-db data/cooc_es.sqlite
```

El peso del término PNPMI se controla con `--alpha-pnpmi` (por defecto 1.0):

```bash
ktool-es generate --input examples/ejemplo.txt --lexicon data/lexicon_es.txt --cooc-db data/cooc_es.sqlite --alpha-pnpmi 1.0
```

Si ves que los TOD quedan **demasiado laxamente vinculados**, podés aplicar cortes opcionales
contra las **anclas** (términos TID centrales) para exigir algo de señal:

```bash
ktool-es generate --input examples/ejemplo.txt --lexicon data/lexicon_es.txt --cooc-db data/cooc_es.sqlite --tod-min-cos 0.25 --tod-min-pnpmi 0.05
```

Si querés auditar por qué entran/salen términos, podés pedir un **informe** con la
distribución de `support`, `max_cosine_to_anchor` y `max_pnpmi_to_anchor` para los TOD/NT
seleccionados:

```bash
ktool-es generate --input examples/ejemplo.txt --lexicon data/lexicon_es.txt --cooc-db data/cooc_es.sqlite --report reports/diagnostics.md
```

## Modelos recomendados (español / multilingüe)

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingüe, buen baseline)
- `sentence-transformers/distiluse-base-multilingual-cased-v2` (multilingüe)

Puedes pasar cualquier modelo compatible con `sentence-transformers`.

## Notas de diseño (alineación con el paper)

- El paper usa Sentence-BERT MiniLM-L6-v2 y embeddings estáticos precomputados para términos externos.
  - Aquí: para TOD/NT se embeben candidatos del léxico bajo demanda y se cachean localmente.
- El paper usa Affinity Propagation con coseno y centroides ponderados por tf.
  - Aquí: centroides ponderados por frecuencia del término en el documento, y similitud coseno.

## Estructura

- `src/ktool_es/cli.py`: CLI
- `src/ktool_es/pipeline.py`: pipeline principal
- `src/ktool_es/resources.py`: hooks/placeholders (léxico, PNPMI, dificultad)
- `src/ktool_es/nlp.py`: extracción de candidatos (spaCy)

