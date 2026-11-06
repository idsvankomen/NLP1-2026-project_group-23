from tokenizers import Tokenizer, Encoding
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import json

class FrequencyEncoding():

    def __init__(self, tokens, ids):
        self.tokens = tokens
        self.ids = ids

class FrequencyTokenizer:

    def __init__(self, max_vocab_size):
        self.max_vocab_size = max_vocab_size
        self.vocab = {}
        self.idfs = None


    def train(self, corpus):

        n = len(corpus)
        word_counts = Counter()
        idfs =  {}

        for text in corpus:
            words = re.sub(r'([^\w\s]|_)',r'', text.lower()).split()
            word_counts.update(words)
            for word in set(words):
                if word in idfs:
                    idfs[word] += 1
                else:
                    idfs[word] = 1

        self.vocab = {"[UNK]": 0}

        # Keep the most frequent words
        most_common_words = word_counts.most_common(
            self.max_vocab_size - 1
        )

        for word, _ in most_common_words:
            self.vocab[word] = len(self.vocab)

        self.idfs = np.array([0] + [np.log(n/idfs[x]) for x in self.vocab if x != "[UNK]"])

        return self

    def encode(self, text):
        words = re.sub(r'([^\w\s]|_)',r'', text.lower()).split()

        ids = [
            self.vocab.get(word, self.vocab["[UNK]"])
            for word in words
        ]

        tokens = [
            word if word in self.vocab else "[UNK]"
            for word in words
        ]

        encoding = FrequencyEncoding(tokens, ids)

        return encoding

    def get_vocab(self):
        return self.vocab

    def load_from_file(self, json_file):

        with open(json_file, 'r') as f:
            d = json.load(f)

        self.vocab = d["vocab"]
        self.idfs = d["idfs"]

    def save(self, filename):
        tokenizer_file = {}
        tokenizer_file["vocab"] = self.vocab
        tokenizer_file["idfs"] = [float(x) for x in self.idfs]

        with open(filename, "w") as f:
            json.dump(tokenizer_file, f, indent=4)

class Encoder:

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        raise NotImplementedError("Concrete implementations will be discussed later")

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        raise NotImplementedError("Concrete implementations will be discussed later")


class TokenIndicator(Encoder):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        unique_token_ids = list(set(self.tokenizer.encode(x).ids))

        # A 0 vector with the same length as the vocabulary except in positions corresponding
        # to tokens appearing in the document
        vector = np.zeros(len(self))
        vector[unique_token_ids] = 1
        return vector


class BagOfWords(Encoder):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Dimensionality of the representation space."""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a vector of token frequencies."""

        # Tokenize the input text
        token_ids = self.tokenizer.encode(x).ids

        # Initialise a zero vector
        vector = np.zeros(len(self), dtype=int)

        # Count the occurrences of each token
        np.add.at(vector, token_ids, 1)

        return vector


class TF_IDF(Encoder):

    def __init__(self, tokenizer: FrequencyTokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Dimensionality of the representation space."""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a vector of token frequencies."""

        # Tokenize the input text
        token_ids = self.tokenizer.encode(x).ids

        # Initialise a zero vector
        vector = np.zeros(len(self), dtype=float)

        # Count the occurrences of each token
        np.add.at(vector, token_ids, 1)
        vector *= self.tokenizer.idfs
        return vector