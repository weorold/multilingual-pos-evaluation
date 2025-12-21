# Multilingual POS Tagging Evaluation

**Examining NLP Performance Disparities Across High-Resource and Low-Resource Languages**

Author: Justin "Aurelio" Fernandez Sanchez  
Date: December 16 2025  
Course: LNG 3430: Internet Linguistics, CUNY Lehman College

---

## Overview

This project evaluates part-of-speech (POS) tagging accuracy across languages with different resource availability levels using Stanza, a neural NLP toolkit. The research reveals systematic performance disparities between high-resource languages (English, Standard Spanish) and low-resource varieties (Yoruba, Dominican Spanish).

**Key Finding:** 78.2 percentage point accuracy gap between English (96.5%) and Yoruba (18.3%), with Dominican Spanish achieving only 73.4% compared to Standard Spanish's 92.9%.

---

## Research Questions

1. How do POS tagging models perform across languages with varying resource availability?
2. Do dialectal varieties experience degraded performance compared to standard varieties?
3. Are performance disparities due to data quantity or structural infrastructure bias?

---

## Methodology

- **Languages Evaluated:** English, Standard Spanish, Yoruba, Dominican Spanish
- **Tool:** Stanza v1.11.0 (neural POS tagger)
- **Data:** Universal Dependencies v2.17 test sets
- **Metrics:** Accuracy, macro F1, weighted F1, confusion matrices, per-POS accuracy
- **Custom Corpus:** Programmatically generated 870-token Dominican Spanish evaluation set

---

## Key Results

| Language | Accuracy | F1 (Weighted) | Tokens |
|----------|----------|---------------|--------|
| English | 96.5% | 0.965 | 25,094 |
| Spanish (Standard) | 92.9% | 0.928 | 12,002 |
| **Yoruba** | **18.3%** | **0.057** | 8,243 |
| **Dominican Spanish** | **73.4%** | **0.752** | 870 |

**Error Analysis:** Dominican Spanish errors concentrated on dialectal features:
- Phonological contractions (pa', ta): High failure rate
- Code-switches: Systematic misclassification
- Regional vocabulary: Unrecognized as valid forms

---

## Technologies Used

- **Python 3.10**
- **NLP:** Stanza, spaCy (evaluation), CoNLL-U parsing
- **Data Analysis:** pandas, NumPy, scikit-learn
- **Visualization:** matplotlib, seaborn

---

## Repository Contents

- `Fernandez_Sanchez_NLP_Evaluation.pdf` - Full research paper (52 pages)
- `code/pos_evaluation.py` - Main evaluation script (~600 lines)
- `code/dominican_corpus.py` - Dominican Spanish corpus generator (~150 lines)
- `code/requirements.txt` - Python dependencies
- `results/` - Confusion matrices and visualizations

---

## How to Run

### Prerequisites
```bash
pip install -r code/requirements.txt
```

### Download Universal Dependencies Data
1. Visit [Universal Dependencies](https://universaldependencies.org/)
2. Download v2.17 test sets for:
   - English-EWT
   - Spanish-GSD
   - Yoruba-YTB
3. Place in `data/` directory

### Run Evaluation
```bash
python code/pos_evaluation.py
```

This will generate:
- Accuracy metrics for all languages
- Confusion matrices (saved as PNG files)
- Per-POS accuracy breakdowns (CSV files)

---

## Significance

This research quantifies systematic exclusion of low-resource languages and dialectal varieties from NLP infrastructure. Despite Yoruba's 48 million speakers and available training data, major NLP libraries provide no pretrained models, demonstrating that infrastructure gaps are policy choices and not technical limitations.

The findings have implications for:
- Language technology equity
- Digital inclusion for Global South languages
- Bias in AI systems
- NLP research priorities

---

## Future Work

- Train actual Yoruba POS tagging models using available UD data
- Expand dialectal corpus with naturalistic data sources
- Evaluate multilingual transformers (mBERT, XLM-R) on same language sample
- Develop architectural interventions to reduce dialectal bias

---

## Citation

If you reference this work, please cite:
```
Fernandez Sanchez, J. (2025). Multilingual POS Tagging Evaluation: 
Examining NLP Performance Disparities Across High-Resource and 
Low-Resource Languages. CUNY Lehman College.
```

---

## Contact

Justin Fernandez Sanchez  
Email: justinfernandez777@gmail.com  
LinkedIn: [linkedin.com/in/justinfernandezsanchez](https://linkedin.com/in/justinfernandezsanchez)

---

## Acknowledgments

Special thanks to Matthew Malone (CUNY Graduate Center/CUNY Lehman College) for guidance on this project, and to the Universal Dependencies community for maintaining multilingual treebanks.

---

## License

This project is for academic and educational purposes.
