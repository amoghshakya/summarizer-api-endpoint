# TODO: Implement a dyslexia-friendly text preprocessor

import re
from collections import defaultdict
from heapq import nsmallest
from typing import Dict, List, Tuple

import nltk
import spacy
from nltk.corpus import wordnet
from syllapy import count as syllable_count

# Download required NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


class DyslexiaFriendlyPreprocessor:
    def __init__(self):
        # Load SpaCy's English model
        self.nlp = spacy.load("en_core_web_sm")

        # Configuration
        self.max_sentence_length = 15  # in words
        self.max_syllables_per_word = 3
        self.max_word_length = 8  # characters
        self.target_grade_level = 6  # approximate reading level

        # Conjunctions to split sentences at
        self.split_conjunctions = {
            'and', 'but', 'or', 'which', 'that', 'because',
            'however', 'although', 'while', 'so'
        }

        # Cache for synonyms to avoid repeated lookups
        self.synonym_cache = defaultdict(list)

        # Common word exceptions that shouldn't be replaced
        self.word_blacklist = {
            'people', 'world', 'important', 'science', 'research'
        }

    def _get_simplest_synonym(self, word: str, pos: str | None) -> Tuple[str, int]:
        """
        Get the simplest synonym for a word based on:
        - Syllable count
        - Word length
        - Frequency (approximated by length)
        Returns (synonym, syllable_count)
        """
        if word.lower() in self.word_blacklist:
            return word, syllable_count(word)

        # Check cache first
        cache_key = f"{word.lower()}_{pos}" if pos else word.lower()
        if cache_key in self.synonym_cache:
            return self.synonym_cache[cache_key][0]

        # Get all synonyms from WordNet with matching POS if provided
        synonyms = set()
        for syn in wordnet.synsets(word, pos=self._convert_pos(pos) if pos else None):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if ' ' not in synonym:  # Only single-word synonyms
                    synonyms.add(synonym.lower())

        # Include the original word for comparison
        synonyms.add(word.lower())

        # Rank synonyms by simplicity (syllables, then length)
        ranked = []
        for synonym in synonyms:
            syl_count = syllable_count(synonym)
            ranked.append((syl_count, len(synonym), synonym))

        # Get the top 3 simplest by syllable count, then choose shortest
        simplest = nsmallest(3, ranked, key=lambda x: (x[0], x[1]))
        best_synonym = simplest[0][2]
        best_syllables = simplest[0][0]

        # Only use synonym if it's actually simpler
        original_syllables = syllable_count(word)
        if best_syllables >= original_syllables and len(best_synonym) >= len(word):
            best_synonym = word
            best_syllables = original_syllables

        # Cache the result
        self.synonym_cache[cache_key].append((best_synonym, best_syllables))

        return best_synonym, best_syllables

    def _convert_pos(self, spacy_pos: str) -> str | None:
        """Convert SpaCy POS tags to WordNet POS tags"""
        if spacy_pos.startswith('V'):
            return wordnet.VERB
        elif spacy_pos.startswith('N'):
            return wordnet.NOUN
        elif spacy_pos.startswith('J'):
            return wordnet.ADJ
        elif spacy_pos.startswith('R'):
            return wordnet.ADV
        return None

    def _should_simplify(self, word: str, syllable_count: int) -> bool:
        """Determine if a word should be simplified"""
        return (syllable_count > self.max_syllables_per_word or
                len(word) > self.max_word_length)

    def _simplify_sentence(self, sentence: str) -> str:
        """Simplify words in a sentence while preserving meaning"""
        doc = self.nlp(sentence)
        simplified_tokens = []

        for token in doc:
            # Skip punctuation and spaces
            if token.is_punct or token.is_space:
                simplified_tokens.append(token.text)
                continue

            # Check if word needs simplification
            syl_count = syllable_count(token.text)
            if not self._should_simplify(token.text, syl_count):
                simplified_tokens.append(token.text)
                continue

            # Get simplest synonym with matching POS
            synonym, new_syl_count = self._get_simplest_synonym(
                token.text,
                token.pos_
            )

            # Maintain original capitalization
            if token.text[0].isupper():
                synonym = synonym.capitalize()

            simplified_tokens.append(synonym)

        return ''.join([
            (' ' + tok if not tok.startswith("'") and
             tok not in {'', '.', ',', '!', '?'} else tok)
            for tok in simplified_tokens
        ]).strip()

    def _split_long_sentences(self, text: str) -> List[str]:
        """Split complex sentences into simpler ones using syntactic analysis"""
        doc = self.nlp(text)
        new_sentences = []

        for sent in doc.sents:
            # Skip if sentence is already short enough
            if len(sent) <= self.max_sentence_length:
                new_sentences.append(sent.text)
                continue

            # Find good splitting points using dependency parsing
            split_points = []
            for token in sent:
                if token.text.lower() in self.split_conjunctions:
                    # Only split if the conjunction connects substantial clauses
                    if (token.head.dep_ in {'ROOT', 'cc'} and
                        len(list(token.lefts)) > 3 and
                            len(list(token.rights)) > 3):
                        split_points.append(token.i)

            # Split at the best point found
            if split_points:
                # Choose split point closest to middle
                mid = len(sent) // 2
                best_split = min(split_points, key=lambda x: abs(x - mid))
                left = sent[:best_split].text
                right = sent[best_split:].text

                # Clean up the split
                left = left.rstrip(' ,')
                right = right.lstrip(' ,')

                # Ensure the second part starts with capital
                if right and not right[0].isupper():
                    right = right[0].upper() + right[1:]

                new_sentences.extend([left + '.', right])
            else:
                # Fallback: split at clauses
                clauses = []
                current_clause = []
                for token in sent:
                    current_clause.append(token.text)
                    if token.dep_ in {'cc', 'mark', 'advcl'} and len(current_clause) > 5:
                        clauses.append(' '.join(current_clause))
                        current_clause = []
                if current_clause:
                    clauses.append(' '.join(current_clause))

                if len(clauses) > 1:
                    new_sentences.extend(clauses)
                else:
                    # Final fallback: split in half
                    mid = len(sent) // 2
                    left = sent[:mid].text
                    right = sent[mid:].text
                    new_sentences.extend([left + '-', right])

        return new_sentences

    def _postprocess(self, text: str) -> str:
        """Clean up the final text output"""
        # Fix spaces before punctuation
        text = re.sub(r'\s([?.!,])', r'\1', text)
        # Fix double spaces
        text = re.sub(r' +', ' ', text)
        # Ensure proper capitalization
        sentences = [s[0].upper() + s[1:] for s in text.split('. ')]
        text = '. '.join(sentences)
        return text

    def process(self, text: str) -> str:
        """Main processing method"""
        # Step 1: Split long sentences
        sentences = self._split_long_sentences(text)

        # Step 2: Simplify words in each sentence
        simplified = [self._simplify_sentence(s) for s in sentences]

        # Step 3: Join and post-process
        result = ' '.join(simplified)
        return self._postprocess(result)


# Example usage
if __name__ == "__main__":
    preprocessor = DyslexiaFriendlyPreprocessor()

    complex_text = """
    The transformer is a deep learning architecture that was developed by researchers 
    at Google and is based on the multi-head attention mechanism, which was proposed 
    in the 2017 paper "Attention Is All You Need". This innovative approach 
    revolutionized natural language processing by eliminating the need for recurrent 
    connections and instead relying entirely on self-attention to model relationships 
    between all words in a sequence.
    """

    print("Original text:")
    print(complex_text)

    simplified_text = preprocessor.process(complex_text)

    print("\nSimplified text:")
    print(simplified_text)
