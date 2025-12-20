"""
Dominican Spanish Corpus Generator
Author: Justin "Aurelio" Fernandez Sanchez
Date: December 16 2025

Generates programmatic evaluation corpus for Dominican Spanish dialectal variety.
Creates sentences with phonological contractions (pa', ta), code-switches, and
distinctive Dominican lexical items for POS tagging evaluation.

Part of: Multilingual POS Tagging Evaluation Project
"""

import random
from typing import List, Tuple, Dict

class DominicanSpanishCorpusGenerator:

  """
  Generator for Dominican Spanish dialect corpus with POS tags.
  Creates sentences featuring:
  - Phonological contractions: pa' (para), ta (está)
  - Code-switches: ticket, meeting, party, full
  - Distinctive vocabulary: vaina, tiguere, enratiao, colmado
  - Regional expressions: guagua, motoconcho, sancocho
  """

  # -----------------------------------------
  # DIALECTAL FEATURE INVENTORY
  # -----------------------------------------

  DIALECTAL_FEATURES = {
    "phonological": ["ta", "mai", "pai", "guagua", "colmado", "loma", 
                    "motoconcho", "chin"],
    "lexical": ["vaina", "tiguere", "pana", "jevo", "enratiao", 
               "diache", "chele"],
    "code_switch": ["ticket", "meeting", "party", "full", "parking", 
                   "iPhone", "ride", "play"],
    "contractions": ["pa'", "t'", "d'"]
  }

  # -----------------------------------------
  # SENTENCE TEMPLATES
  # -----------------------------------------

  # Each template is a list of (word_slot, pos_tag) pairs
  # Slots are filled from WORD_BANK, literals are used directly

  TEMPLATES = [

    # Question patterns
    [("¿", "PUNCT"), ("QU_WORD", "PRON/ADV"), ("PRON", "PRON"), 
     ("ta", "VERB"), ("ADJ", "ADJ"), ("?", "PUNCT")],
    [("¿", "PUNCT"), ("QU_WORD", "PRON/ADV"), ("lo", "PRON"), 
     ("que", "PRON"), ("ta", "VERB"), ("?", "PUNCT")],

    # Statement patterns with dialectal features
    [("PRON", "PRON"), ("ta", "AUX"), ("ADJ", "ADJ"), (".", "PUNCT")],
    [("Vamos", "VERB"), ("pa'", "ADP"), ("DET", "DET"), ("NOUN", "NOUN"), 
     ("TIME", "NOUN/ADV"), (".", "PUNCT")],
    [("Esa", "DET"), ("vaina", "NOUN"), ("no", "ADV"), ("VERB", "VERB"), 
     ("bien", "ADV"), (".", "PUNCT")],
    [("Ese", "DET"), ("tiguere", "NOUN"), ("ta", "AUX"), 
     ("enratiao", "ADJ"), (".", "PUNCT")],

    # Exclamation patterns
    [("¡", "PUNCT"), ("INTJ", "INTJ"), ("!", "PUNCT")],

    # Code-switching patterns
    [("Necesito", "VERB"), ("un", "DET"), ("CODE_NOUN", "NOUN"), 
     ("pa'", "ADP"), ("el", "DET"), ("NOUN", "NOUN"), (".", "PUNCT")],
    [("Ese", "DET"), ("party", "NOUN"), ("estuvo", "AUX"), 
     ("brutal", "ADJ"), (".", "PUNCT")],
  ]

  # -----------------------------------------
  # WORD INVENTORY BY CATEGORY
  # -----------------------------------------

  WORD_BANK: Dict[str, List[Tuple[str, str]]] = {
    "QU_WORD": [
      ("qué", "PRON"), ("cómo", "ADV"), ("dónde", "ADV"),
      ("cuándo", "ADV"), ("por qué", "ADV")
    ],
    "PRON": [
      ("tú", "PRON"), ("él", "PRON"), ("ella", "PRON"),
      ("usted", "PRON"), ("nosotros", "PRON"), ("ellos", "PRON")
    ],
    "DET": [
      ("la", "DET"), ("el", "DET"), ("un", "DET"),
      ("una", "DET"), ("ese", "DET"), ("esa", "DET")
    ],
    "NOUN": [
      ("playa", "NOUN"), ("calle", "NOUN"), ("casa", "NOUN"),
      ("trabajo", "NOUN"), ("fiera", "NOUN"), ("sancocho", "NOUN"),
      ("colmado", "NOUN"), ("guagua", "NOUN"), ("motoconcho", "NOUN")
    ],
    "VERB": [
      ("funciona", "VERB"), ("cocina", "VERB"), ("corre", "VERB"),
      ("trabaja", "VERB"), ("canta", "VERB"), ("baila", "VERB")
    ],
    "ADJ": [
      ("bien", "ADJ"), ("rápido", "ADV"), ("duro", "ADJ"),
      ("enratiao", "ADJ"), ("full", "ADV"), ("brutal", "ADJ")
    ],
    "INTJ": [
      ("Diache", "INTJ"), ("Wey", "INTJ"), ("Dale", "INTJ"),
      ("Klk", "INTJ"), ("Manito", "INTJ")
    ],
    "TIME": [
      ("hoy", "NOUN"), ("mañana", "NOUN"), ("ahora", "ADV"),
      ("este fin de semana", "NOUN")
    ],
    "CODE_NOUN": [
      ("ticket", "NOUN"), ("meeting", "NOUN"), ("ride", "NOUN"),
      ("texto", "NOUN"), ("parking", "NOUN"), ("iPhone", "PROPN")
    ]
  }

  # -----------------------------------------
  # CORPUS GENERATION METHODS
  # -----------------------------------------

  @staticmethod
  def create_sentences() -> List[List[Tuple[str, str]]]:

    """
    Create base seed sentences for the corpus.
    
    Returns:
      List of sentences, where each sentence is a list of (token, pos) tuples
    """

    return [
      # Seed sentence: "Hola, ¿cómo tú ta?"
      [("Hola", "INTJ"), (",", "PUNCT"), ("¿", "PUNCT"), ("cómo", "ADV"),
       ("tú", "PRON"), ("ta", "VERB"), ("?", "PUNCT")],
    ]

  @classmethod
  def generate_sentence(cls) -> List[Tuple[str, str]]:

    """
    Generate one new sentence from templates using random word selection.
    Process:
    1. Randomly select a sentence template
    2. For each slot in template, fill with random word from that category
    3. Keep literal words (punctuation, fixed items) as-is
    
    Returns:
      List of (token, pos_tag) tuples representing one sentence
    """

    template = random.choice(cls.TEMPLATES)
    sentence = []
    for slot, pos in template:
      if slot in cls.WORD_BANK:
        # Fill slot with random word from appropriate category
        word, correct_pos = random.choice(cls.WORD_BANK[slot])
        sentence.append((word, correct_pos))
      else:
        # Use slot value as literal word (handles punctuation, fixed words)
        sentence.append((slot, pos))
    return sentence
  
  @classmethod
  def expand_corpus(cls, target_sentences: int = 150) -> List[List[Tuple[str, str]]]:

    """
    Expand corpus to target number of sentences by generating variations.
    
    Args:
      target_sentences: Desired number of sentences (default: 150)
    
    Returns:
      List of sentences with POS-tagged tokens
    
    Example:
      >>> corpus = DominicanSpanishCorpusGenerator.expand_corpus(150)
      Expanded corpus to 150 sentences
        Total tokens: 870
    """

    base = cls.create_sentences()
    # Generate new sentences until we reach target
    while len(base) < target_sentences:
      new_sentence = cls.generate_sentence()
      # Avoid exact duplicates
      if new_sentence not in base:
        base.append(new_sentence)
    # Report statistics
    total_tokens = sum(len(s) for s in base)
    print(f"  Expanded corpus to {len(base)} sentences")
    print(f"  Total tokens: {total_tokens}")
    return base
  
  @staticmethod
  def to_conllu(sentences: List[List[Tuple]], output_file: str):

    """
    Export corpus to CoNLL-U format for Universal Dependencies compatibility.
    
    Args:
      sentences: List of sentences with (token, pos) tuples
      output_file: Path to output .conllu file
    
    Format:
      # sent_id = 1
      # text = Hola, ¿cómo tú ta?
      1   Hola    _   INTJ    _   _   _   _   _   _
      2   ,       _   PUNCT   _   _   _   _   _   _
      ...
    """

    with open(output_file, 'w', encoding='utf-8') as f:
      for sent_id, sentence in enumerate(sentences, 1):
        # Reconstruct text from tokens
        text = ' '.join([token for token, pos in sentence])
        # Write sentence metadata
        f.write(f"# sent_id = {sent_id}\n")
        f.write(f"# text = {text}\n")
        # Write token lines (CoNLL-U has 10 tab-separated fields)
        for token_id, (token, pos) in enumerate(sentence, 1):
          f.write(f"{token_id}\t{token}\t_\t{pos}\t_\t_\t_\t_\t_\t_\n")
        # Blank line separates sentences
        f.write("\n")
    print(f"Saved {len(sentences)} sentences to {output_file}")

# -----------------------------------------
# COMMAND-LINE USAGE
# -----------------------------------------

if __name__ == "__main__":

  """
  Generate Dominican Spanish corpus and export to CoNLL-U format.
  
  Usage:
    python dominican_spanish_corpus.py
  
  Output:
    dominican_spanish_expanded.conllu (150 sentences, ~870 tokens)
  """

  # Generate corpus
  corpus = DominicanSpanishCorpusGenerator.expand_corpus(target_sentences=150)
  # Export to CoNLL-U format
  DominicanSpanishCorpusGenerator.to_conllu(
    corpus, 
    'dominican_spanish_expanded.conllu'
  )

  # Display feature statistics
  print("\n--- Feature Statistics ---")

  # Count dialectal features
  all_tokens = [token for sent in corpus for token, _ in sent]
  pa_count = all_tokens.count("pa'")
  ta_count = all_tokens.count("ta")
  vaina_count = all_tokens.count("vaina")
  tiguere_count = all_tokens.count("tiguere")
  print(f"Contractions: pa' ({pa_count}), ta ({ta_count})")
  print(f"Lexical items: vaina ({vaina_count}), tiguere ({tiguere_count})")
  print(f"Code-switches: {len([t for t in all_tokens if t in DominicanSpanishCorpusGenerator.DIALECTAL_FEATURES['code_switch']])}")