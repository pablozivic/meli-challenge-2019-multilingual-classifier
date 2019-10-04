import io
import unidecode
import numpy as np
from collections import defaultdict

from typing import Dict

from multilingual_title_classifier.src.path_helpers import get_resources_path

_EMBEDDING_DIM = 300  # TODO: Try to train and rectify embeddings, playing with this dimension too.
_EMBEDDING_FILES = {'es': 'wiki.multi.es.vec',
                    'pt': 'wiki.multi.pt.vec',
                    'en': 'wiki.multi.en.vec'}


def get_embeddings_vocabulary() -> dict:
    """
    Extract vocabulary from embeddings files.

    :return: Mapping from word to embedding.
    """
    vocabulary = defaultdict(set)
    for language, emb_path in _EMBEDDING_FILES.items():
        with io.open(get_resources_path(emb_path), 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            next(f)
            for i, line in enumerate(f):
                word, vect = line.rstrip().split(' ', 1)
                vocabulary[language].add(unidecode.unidecode(word))
    return vocabulary


def load_embeddings(vocabulary: list) -> dict:
    """
    Get dictionary containing pre-trained word embeddings.

    :param vocabulary: List of words we want embeddings for.
    :return: Dictionary with embeddings (word, embedding)
    """
    embeddings = {}

    for language, emb_path in _EMBEDDING_FILES.items():
        with io.open(get_resources_path(emb_path), 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            next(f)
            for i, line in enumerate(f):
                word, vect = line.rstrip().split(' ', 1)
                word = unidecode.unidecode(word)

                if word not in vocabulary:
                    continue
                else:
                    vect = np.fromstring(vect, sep=' ')

                # TODO: Consider language in mapping?
                embeddings[word] = vect

                if len(embeddings) == len(vocabulary):
                    break

    return embeddings


def get_embeddings_matrix(word_index: Dict[str, int]) -> np.array:
    """
    Create embeddings matrix from a word index.

    :param word_index: Dictionary mapping words to indexes.
    :return: Numpy array containing embeddings matrix.
    """
    embeddings = load_embeddings(vocabulary=list(word_index.keys()))

    # 0 is reserved for _pad_ token.
    embedding_matrix = np.zeros((len(word_index) + 1, _EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            # TODO: maybe we can find a better initialization strategy?
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
