"""
Multilingual POS Tagging Evaluation System
Author: Justin "Aurelio" Fernandez Sanchez
Date: December 16 2025

Evaluates part-of-speech tagging performance across languages with varying 
resource availability using Stanza and Universal Dependencies corpora.

Languages evaluated:
- English (high-resource)
- Standard Spanish (high-resource)
- Yoruba (low-resource, no pretrained model)
- Dominican Spanish (dialectal variety)

Main functions:
- load_conllu_file(): Parse Universal Dependencies test files
- tag_with_stanza(): Apply Stanza POS tagger to tokens
- evaluate_tagging(): Calculate accuracy, F1 scores, per-POS metrics
- create_confusion_matrix(): Visualize error patterns
- run_experiment_pretrained(): Test real-world pretrained models
- run_experiment_controlled(): Compare with equal-sized datasets

Part of: Multilingual POS Tagging Evaluation Project
Full paper: github.com/weorold/multilingual-pos-evaluation/blob/main/Fernandez_Sanchez_NLP_Evaluation.pdf
"""

import spacy
import stanza
from conllu import parse_incr
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------------------
# SECTION 1: DATA LOADING AND PREPROCESSING
# -----------------------------------------

def load_conllu_file(filepath, max_sentences=None):
  
  """
  Load CoNLL-U formatted file and extract tokens with gold POS tags.
  CoNLL-U is the standard format for Universal Dependencies treebanks.
  Each token has fields: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
  
  Args:
    filepath (str): Path to .conllu file
    max_sentences (int, optional): Limit number of sentences to load
  
  Returns:
    list: List of (token, pos_tag) tuples
  
  Example:
    >>> tokens_and_tags = load_conllu_file('en_ewt-ud-test.conllu')
    Loaded 25094 tokens from 2077 sentences
  """
  
  tokens_and_tags = []
  with open(filepath, 'r', encoding='utf-8') as f:
    sentence_count = 0
    for sentence in parse_incr(f):
      if max_sentences and sentence_count >= max_sentences:
        break
      for token in sentence:
        # Skip multi-word tokens (IDs like "1-2") and empty nodes (IDs like "1.1")
        if isinstance(token['id'], int):
          word = token['form']
          pos = token['upos']
          tokens_and_tags.append((word, pos))
      sentence_count += 1

  print(f"Loaded {len(tokens_and_tags)} tokens from {sentence_count} sentences")
  return tokens_and_tags

def create_equal_sized_corpora(corpora_paths, target_size=5000):

  """
  Create equal-sized datasets for controlled experiment.
  Limits each corpus to target_size tokens to isolate the effect of 
  linguistic properties from data quantity.
  
  Args:
    corpora_paths (dict): {language_name: filepath} mapping
    target_size (int): Target number of tokens per language
  
  Returns:
    dict: {language_name: [(token, pos), ...]} with equal-sized datasets
  """

  equal_corpora = {}
  for lang_name, path in corpora_paths.items():
    tokens_and_tags = []
    with open(path, 'r', encoding='utf-8') as f:
      token_count = 0
      for sentence in parse_incr(f):
        for token in sentence:
          if isinstance(token['id'], int):
            tokens_and_tags.append((token['form'], token['upos']))
            token_count += 1
        # Stop when we reach target size
        if token_count >= target_size:
          break
    equal_corpora[lang_name] = tokens_and_tags[:target_size]

    print(f"{lang_name}: {len(equal_corpora[lang_name])} tokens")

  return equal_corpora

# -----------------------------------------
# SECTION 2: POS TAGGING WITH NEURAL MODELS
# -----------------------------------------

def tag_with_spacy(tokens, model_name):

  """
  Tag tokens using spaCy's pretrained model.
  Note: spaCy tokenization may differ from UD tokenization, causing
  alignment issues. This function is included for comparison but
  may produce unreliable results due to tokenization mismatches.
  
  Args:
    tokens (list): List of token strings
    model_name (str): spaCy model name (e.g., 'en_core_web_sm')
  
  Returns:
    list: Predicted POS tags
  """

  try:
    nlp = spacy.load(model_name)
  except OSError:
    print(f"Downloading spaCy model {model_name}...")
    spacy.cli.download(model_name)
    nlp = spacy.load(model_name)

  # Reconstruct text and process
  text = " ".join(tokens)
  doc = nlp(text)
  predicted_tags = [token.pos_ for token in doc]
  return predicted_tags

def tag_with_stanza(tokens, lang_code):

  """
  Tag tokens using Stanza's pretrained model.
  Stanza uses tokenize_pretokenized=True to respect UD tokenization,
  ensuring alignment between predictions and gold tags.
  Special case: Yoruba (lang_code='yo') has no Stanza model, so a
  naive baseline (predict NOUN for all tokens) is used instead.
  
  Args:
    tokens (list): List of token strings
    lang_code (str): Stanza language code (e.g., 'en', 'es', 'yo')
  
  Returns:
    list: Predicted POS tags (aligned with input tokens)
  
  Example:
    >>> tags = tag_with_stanza(['The', 'cat', 'sleeps'], 'en')
    >>> print(tags)
    ['DET', 'NOUN', 'VERB']
  """

  # Handle Yoruba infrastructure gap
  if lang_code == 'yo':
    print("Yoruba not supported by Stanza! Using baseline tagger")
    # Naive baseline: predict NOUN for all tokens (most frequent POS)
    # This quantifies the cost of infrastructure exclusion
    return ['NOUN'] * len(tokens)

  # Initialize Stanza pipeline
  try:
    nlp = stanza.Pipeline(lang_code, processors='tokenize,pos', tokenize_pretokenized=True, verbose=False)
  except:
    print(f"Downloading Stanza model for {lang_code}...")
    stanza.download(lang_code)
    nlp = stanza.Pipeline(lang_code, processors='tokenize,pos', tokenize_pretokenized=True, verbose=False)

  # Process tokens in batches for efficiency
  batch_size = 100
  all_tags = []
  for i in range(0, len(tokens), batch_size):
    batch = [tokens[i:i + batch_size]]
    doc = nlp(batch)
    for sentence in doc.sentences:
      for word in sentence.words:
        all_tags.append(word.upos)
  return all_tags

# -----------------------------------------
# SECTION 3: EVALUATION METRICS
# -----------------------------------------

def evaluate_tagging(gold_tags, predicted_tags, language_name):

  """
  Calculate comprehensive evaluation metrics for POS tagging.
  Metrics computed:
  - Overall accuracy: % of tokens tagged correctly
  - F1 macro: Average F1 across all POS categories (equal weight)
  - F1 weighted: Average F1 weighted by category frequency
  - Per-POS accuracy: Accuracy for each individual POS tag
  
  Args:
    gold_tags (list): True POS tags from corpus
    predicted_tags (list): Model predictions
    language_name (str): Name for display/reporting
  
  Returns:
    dict: Comprehensive results including all metrics
  
  Example output:
    'language': 'English (Stanza)',
    'overall_accuracy': 0.9654,
    'f1_macro': 0.9169,
    'f1_weighted': 0.9649,
    'per_pos_accuracy': {'NOUN': {'accuracy': 0.92, 'count': 5000}, ...},
    'total_tokens': 25094
  """

  # Overall metrics
  accuracy = accuracy_score(gold_tags, predicted_tags)
  f1_macro = f1_score(gold_tags, predicted_tags, average='macro', zero_division=0)
  f1_weighted = f1_score(gold_tags, predicted_tags, average='weighted', zero_division=0)

  # Per-POS metrics
  per_pos_results = {}
  unique_tags = set(gold_tags)

  for tag in unique_tags:
    # Find all occurrences of this tag
    tag_indices = [i for i, t in enumerate(gold_tags) if t == tag]
    if tag_indices:
      tag_gold = [gold_tags[i] for i in tag_indices]
      tag_pred = [predicted_tags[i] for i in tag_indices]
      tag_acc = accuracy_score(tag_gold, tag_pred)
      per_pos_results[tag] = {
        'accuracy': tag_acc,
        'count': len(tag_indices)
      }

  results = {
    'language': language_name,
    'overall_accuracy': accuracy,
    'f1_macro': f1_macro,
    'f1_weighted': f1_weighted,
    'per_pos_accuracy': per_pos_results,
    'total_tokens': len(gold_tags)
  }

  # Print summary
  print(f"\n{'=' * 60}")
  print(f"Results for {language_name}")
  print(f"{'=' * 60}")
  print(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
  print(f"F1 Macro:         {f1_macro:.4f}")
  print(f"F1 Weighted:      {f1_weighted:.4f}")
  print(f"Total Tokens:     {len(gold_tags)}")
  return results

def analyze_errors(gold_tags, predicted_tags, tokens, language_name):

  """
  Perform detailed error analysis to identify systematic failure patterns.
  Identifies:
  - Total error count and rate
  - Most common confusion pairs (Gold → Predicted)
  - Most frequently misclassified tokens
  This is particularly important for dialectal varieties where errors
  may concentrate on specific features (e.g., contractions, code-switches).
  
  Args:
    gold_tags (list): True POS tags
    predicted_tags (list): Model predictions
    tokens (list): Original token strings
    language_name (str): Name for display
  
  Returns:
    dict: Error analysis results with confusion pairs and problematic tokens
  """

  errors = []

  # Collect all errors with context
  for i, (gold, pred, token) in enumerate(zip(gold_tags, predicted_tags, tokens)):
    if gold != pred:
      errors.append({
        'token': token,
        'gold': gold,
        'predicted': pred,
        'index': i
      })

  # Analyze error patterns
  confusion_pairs = Counter([(e['gold'], e['predicted']) for e in errors])
  error_tokens = Counter([e['token'] for e in errors])

  # Print analysis
  print(f"\n{'=' * 60}")
  print(f"Error Analysis for {language_name}")
  print(f"{'=' * 60}")
  print(f"Total Errors: {len(errors)} / {len(gold_tags)} ({len(errors) / len(gold_tags) * 100:.2f}%)")
  print(f"\nTop 10 Confusion Pairs (Gold -> Predicted):")
  for (gold, pred), count in confusion_pairs.most_common(10):
    print(f"  {gold:10s} -> {pred:10s}: {count:4d} times")
  print(f"\nTop 10 Most Misclassified Tokens:")
  for token, count in error_tokens.most_common(10):
    print(f"  '{token}': {count} times")
  return {
    'errors': errors,
    'confusion_pairs': confusion_pairs,
    'error_tokens': error_tokens
  }

def create_results_table(all_results):

  """
  Create pandas DataFrame with all results for easy comparison.
  
  Args:
    all_results (dict): {language_name: results_dict} mapping
  
  Returns:
    pd.DataFrame: Formatted table with key metrics
  """

  rows = []
  for lang, result in all_results.items():
    rows.append({
      'Language': lang,
      'Accuracy': result['overall_accuracy'],
      'F1 Macro': result['f1_macro'],
      'F1 Weighted': result['f1_weighted'],
      'Total Tokens': result['total_tokens']
    })
  df = pd.DataFrame(rows)
  return df

# -----------------------------------------
# SECTION 4: VISUALIZATION FUNCTIONS
# -----------------------------------------

def plot_accuracy_comparison(results_dict, output_file='accuracy_comparison.png'):

  """
  Create bar plot comparing accuracy and F1 scores across languages.
  Generates a side-by-side bar chart showing overall accuracy and
  weighted F1 score for all evaluated languages.
  
  Args:
    results_dict (dict): {language: results} mapping
    output_file (str): Path to save plot
  """

  languages = list(results_dict.keys())
  accuracies = [results_dict[lang]['overall_accuracy'] for lang in languages]
  f1_scores = [results_dict[lang]['f1_weighted'] for lang in languages]
  x = np.arange(len(languages))
  width = 0.35
  fig, ax = plt.subplots(figsize=(12, 7))
  bars1 = ax.bar(x - width / 2, accuracies, width, label='Accuracy', alpha=0.8, color='steelblue')
  bars2 = ax.bar(x + width / 2, f1_scores, width, label='F1 (Weighted)', alpha=0.8, color='coral')
  ax.set_xlabel('Language', fontsize=12)
  ax.set_ylabel('Score', fontsize=12)
  ax.set_title('POS Tagging Performance Across Languages', fontsize=14, fontweight='bold')
  ax.set_xticks(x)
  ax.set_xticklabels(languages, rotation=15, ha='right')
  ax.legend(fontsize=10)
  ax.set_ylim([0, 1])
  ax.grid(axis='y', alpha=0.3)

  # Add value labels on bars
  for bars in [bars1, bars2]:
    for bar in bars:
      height = bar.get_height()
      ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
  plt.tight_layout()
  plt.savefig(output_file, dpi=300, bbox_inches='tight')
  print(f"\nSaved plot to {output_file}")
  plt.close()

def plot_per_pos_accuracy(results_dict, output_file='per_pos_accuracy.png'):

  """
  Create heatmap showing per-POS accuracy across all languages.
  Helps identify which POS categories are problematic for each language.
  For example, reveals that ADP (prepositions) fail completely for
  Dominican Spanish due to contracted forms like pa'.
  
  Args:
    results_dict (dict): {language: results} mapping
    output_file (str): Path to save heatmap
  """

  # Collect all POS tags that appear in any language
  all_pos_tags = set()
  for result in results_dict.values():
    all_pos_tags.update(result['per_pos_accuracy'].keys())
  all_pos_tags = sorted(all_pos_tags)
  languages = list(results_dict.keys())
  matrix = []

  # Build accuracy matrix
  for lang in languages:
    row = []
    for pos in all_pos_tags:
      if pos in results_dict[lang]['per_pos_accuracy']:
        acc = results_dict[lang]['per_pos_accuracy'][pos]['accuracy']
        row.append(acc)
      else:
        row.append(np.nan)
    matrix.append(row)

  # Create heatmap
  fig, ax = plt.subplots(figsize=(14, len(languages) * 0.5 + 2))
  sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn', xticklabels=all_pos_tags, yticklabels=languages, vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'})
  plt.title('Per-POS Accuracy Across Languages', fontsize=14, fontweight='bold')
  plt.xlabel('POS Tag', fontsize=12)
  plt.ylabel('Language', fontsize=12)
  plt.tight_layout()
  plt.savefig(output_file, dpi=300, bbox_inches='tight')
  print(f"Saved per-POS accuracy heatmap to {output_file}")
  plt.close()

def plot_confusion_matrix(gold_tags, predicted_tags, language_name, output_file=None):

  """
  Create confusion matrix heatmap showing error patterns.
  Rows represent true labels, columns represent predictions.
  Diagonal = correct predictions, off-diagonal = errors.
  
  Args:
    gold_tags (list): True POS tags
    predicted_tags (list): Model predictions
    language_name (str): Language name for title
    output_file (str, optional): Path to save plot
  """

  unique_tags = sorted(list(set(gold_tags + predicted_tags)))
  cm = confusion_matrix(gold_tags, predicted_tags, labels=unique_tags)
  plt.figure(figsize=(14, 12))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_tags, yticklabels=unique_tags, cbar_kws={'label': 'Count'})
  plt.title(f'Confusion Matrix - {language_name}', fontsize=14, fontweight='bold')
  plt.ylabel('True Label', fontsize=12)
  plt.xlabel('Predicted Label', fontsize=12)
  plt.tight_layout()
  if output_file:
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {output_file}")
    plt.close()
  else:
    plt.show()

# -----------------------------------------
# SECTION 5: MAIN EXPERIMENTS
# -----------------------------------------

def run_experiment_pretrained(corpora_config, tool='stanza'):

  """
  Run experiment using pretrained models as deployed in real-world usage.
  This represents actual user experience with existing NLP tools.
  Evaluates models on full test sets with no modifications.
  
  Args:
    corpora_config (dict): Configuration for each language including:
      - path: filepath to .conllu test set
      - stanza_lang: language code for Stanza
      - spacy_model: model name for spaCy (optional)
    tool (str): 'stanza', 'spacy', or 'both'
  
  Returns:
    tuple: (all_results, all_errors) dictionaries
  """

  print("\n" + "=" * 70)
  print("EXPERIMENT 1: PRETRAINED MODELS (Real-World Performance)")
  print("=" * 70)
  all_results = {}
  all_errors = {}

  for lang_name, config in corpora_config.items():
    print(f"\n{'=' * 70}")
    print(f"Processing {lang_name}")
    print(f"{'=' * 70}")

    # Load corpus
    tokens_and_tags = load_conllu_file(config['path'])
    tokens = [t[0] for t in tokens_and_tags]
    gold_tags = [t[1] for t in tokens_and_tags]

    # Evaluate with Stanza
    if tool in ['stanza', 'both']:
      print(f"\n--- Tagging with Stanza ---")
      stanza_predictions = tag_with_stanza(tokens, config['stanza_lang'])
      stanza_results = evaluate_tagging(gold_tags, stanza_predictions, f"{lang_name} (Stanza)")
      stanza_errors = analyze_errors(gold_tags, stanza_predictions, tokens, f"{lang_name} (Stanza)")
      all_results[f"{lang_name}-Stanza"] = stanza_results
      all_errors[f"{lang_name}-Stanza"] = stanza_errors
      # Generate confusion matrix
      plot_confusion_matrix(gold_tags, stanza_predictions, f"{lang_name} (Stanza)", f"confusion_matrix_{lang_name.lower().replace(' ', '_')}_stanza.png")

    # Evaluate with spaCy (if model available)
    if tool in ['spacy', 'both'] and config.get('spacy_model'):
      print(f"\n--- Tagging with spaCy ---")
      spacy_predictions = tag_with_spacy(tokens, config['spacy_model'])
      spacy_results = evaluate_tagging(gold_tags, spacy_predictions, f"{lang_name} (spaCy)")
      spacy_errors = analyze_errors(gold_tags, spacy_predictions, tokens, f"{lang_name} (spaCy)")
      all_results[f"{lang_name}-spaCy"] = spacy_results
      all_errors[f"{lang_name}-spaCy"] = spacy_errors
      plot_confusion_matrix(gold_tags, spacy_predictions, f"{lang_name} (spaCy)", f"confusion_matrix_{lang_name.lower().replace(' ', '_')}_spacy.png")

  # Generate comparative visualizations
  plot_accuracy_comparison(all_results, 'pretrained_comparison.png')
  plot_per_pos_accuracy(all_results, 'per_pos_accuracy.png')
  # Save results table
  results_df = create_results_table(all_results)
  results_df.to_csv('pretrained_results.csv', index=False)
  print(f"\n{'=' * 70}")
  print("SUMMARY TABLE")
  print(f"{'=' * 70}")
  print(results_df.to_string(index=False))
  print(f"\nResults saved to pretrained_results.csv")
  return all_results, all_errors

def run_experiment_controlled(test_corpora_config, target_size=5000):

  """
  Run controlled experiment with equal-sized datasets.
  Isolates the effect of corpus size by limiting all languages to
  the same number of tokens. Tests whether performance disparities
  persist when evaluation data quantity is equalized.
  Note: This controls EVALUATION set size, not training data.
  Models are still pretrained on different amounts of data.
  
  Args:
    test_corpora_config (dict): Same format as pretrained experiment
    target_size (int): Number of tokens to use per language
  
  Returns:
    tuple: (all_results, all_errors) dictionaries
  """

  print("\n" + "=" * 70)
  print("EXPERIMENT 2: CONTROLLED (Equal Data Size)")
  print("=" * 70)
  print(f"Token limit per language: {target_size}")
  print("NOTE: Using pretrained models on equal-sized test sets")
  print("=" * 70)
  all_results = {}
  all_errors = {}

  for lang_name, config in test_corpora_config.items():
    print(f"\n{'=' * 70}")
    print(f"Processing {lang_name}")
    print(f"{'=' * 70}")

    # Load and limit corpus
    tokens_and_tags = load_conllu_file(config['path'])
    tokens_and_tags = tokens_and_tags[:target_size]
    tokens = [t[0] for t in tokens_and_tags]
    gold_tags = [t[1] for t in tokens_and_tags]
    print(f"Using {len(tokens)} tokens (limited to {target_size})")

    # Evaluate with Stanza
    print(f"\n--- Tagging with Stanza ---")
    stanza_predictions = tag_with_stanza(tokens, config['stanza_lang'])
    stanza_results = evaluate_tagging(gold_tags, stanza_predictions,
                                      f"{lang_name} (Controlled)")
    stanza_errors = analyze_errors(gold_tags, stanza_predictions, tokens,
                                   f"{lang_name} (Controlled)")
    all_results[f"{lang_name}-Controlled"] = stanza_results
    all_errors[f"{lang_name}-Controlled"] = stanza_errors
    # Generate confusion matrix
    plot_confusion_matrix(gold_tags, stanza_predictions, f"{lang_name} (Controlled)", f"confusion_matrix_{lang_name.lower().replace(' ', '_')}_controlled.png")

  # Generate comparative visualizations
  plot_accuracy_comparison(all_results, 'controlled_comparison.png')
  plot_per_pos_accuracy(all_results, 'controlled_per_pos_accuracy.png')

  # Save results table
  results_df = create_results_table(all_results)
  results_df.to_csv('controlled_results.csv', index=False)
  print(f"\n{'=' * 70}")
  print("CONTROLLED EXPERIMENT SUMMARY")
  print(f"{'=' * 70}")
  print(results_df.to_string(index=False))
  print(f"\nResults saved to controlled_results.csv")
  return all_results, all_errors

# -----------------------------------------
# SECTION 6: MAIN EXECUTION
# -----------------------------------------

if __name__ == "__main__":

  """
  Main execution script for multilingual POS tagging evaluation.
  Workflow:
  1. Generate Dominican Spanish corpus
  2. Configure paths to UD test sets
  3. Run pretrained model experiment
  4. Run controlled experiment with equal corpus sizes
  5. Save all results and visualizations
  
  Output files:
  - pretrained_results.csv: Main results table
  - controlled_results.csv: Controlled experiment results
  - *.png: Confusion matrices and comparison plots
  """

  # Generate Dominican Spanish corpus
  print("=" * 70)
  print("SETUP: Creating Dominican Spanish corpus")
  print("=" * 70)
  from dominican_spanish_corpus import DominicanSpanishCorpusGenerator
  corpus = DominicanSpanishCorpusGenerator.expand_corpus(target_sentences=150)
  DominicanSpanishCorpusGenerator.to_conllu(corpus, 'dominican_spanish_test.conllu')

  # Configuration: Corpus paths and model settings
  corpora_config = {
    'English': {
      'path': 'en_ewt-ud-test.conllu',
      'spacy_model': 'en_core_web_sm',
      'stanza_lang': 'en'
    },
    'Spanish': {
      'path': 'es_gsd-ud-test.conllu',
      'spacy_model': 'es_core_news_sm',
      'stanza_lang': 'es'
    },
    'Yoruba': {
      'path': 'yo_ytb-ud-test.conllu',
      'spacy_model': None,
      'stanza_lang': 'yo'
    },
    'Dominican Spanish': {
      'path': 'dominican_spanish_test.conllu',
      'spacy_model': 'es_core_news_sm',
      'stanza_lang': 'es'
    }
  }

  # Run experiments
  print("\n" + "=" * 70)
  print("RUNNING EXPERIMENTS")
  print("=" * 70)

  # Experiment 1: Pretrained models (real-world deployment)
  results, errors = run_experiment_pretrained(corpora_config, tool='stanza')
  # Experiment 2: Controlled comparison (equal corpus sizes)
  controlled_results, controlled_errors = run_experiment_controlled(
    corpora_config, 
    target_size=5000
  )

  # Summary
  print("\n" + "=" * 70)
  print("ALL EXPERIMENTS COMPLETE!")
  print("=" * 70)
  print("\nPretrained experiment files:")
  print("  - pretrained_results.csv")
  print("  - pretrained_comparison.png")
  print("  - per_pos_accuracy.png")
  print("  - confusion_matrix_*_stanza.png")
  print("\nControlled experiment files:")
  print("  - controlled_results.csv")
  print("  - controlled_comparison.png")
  print("  - controlled_per_pos_accuracy.png")
  print("  - confusion_matrix_*_controlled.png")