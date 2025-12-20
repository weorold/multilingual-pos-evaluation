# Data Sources

This project uses test sets from Universal Dependencies v2.17.

## Download Instructions

1. Visit: https://universaldependencies.org/
2. Download the following treebanks:
   - **English-EWT** (English Web Treebank)
   - **Spanish-GSD** (Spanish Google Universal Dependencies)
   - **Yoruba-YTB** (Yoruba Treebank)

3. Extract the test files:
   - `en_ewt-ud-test.conllu`
   - `es_gsd-ud-test.conllu`
   - `yo_ytb-ud-test.conllu`

4. Place them in this `data/` directory

## Dominican Spanish Corpus

The Dominican Spanish evaluation corpus is generated programmatically using `code/dominican_corpus.py`. No download needed.

---

Expected file structure:
```
data/
├── en_ewt-ud-test.conllu
├── es_gsd-ud-test.conllu
├── yo_ytb-ud-test.conllu
└── README.md (this file)
```
