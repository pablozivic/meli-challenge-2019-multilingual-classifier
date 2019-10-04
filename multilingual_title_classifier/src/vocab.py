import re
import unidecode
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import multilingual_title_classifier.src.config as config


# TODO: Improve vocabulary generation, this is really messy and can be improved with a sklearn Pipeline. Among other
#  things, you may add lemmatization, stemming and a better spell correction system that considers context.

def clean_numbers(title: str) -> str:
    title = re.sub('(\d+(\.|,)+\d+)', ' ######## ', title)
    title = re.sub('[0-9]{7,}', ' ####### ', title)
    title = re.sub('[0-9]{6}', ' ###### ', title)
    title = re.sub('[0-9]{5}', ' ##### ', title)
    title = re.sub('[0-9]{4}', ' #### ', title)
    title = re.sub('[0-9]{3}', ' ### ', title)
    title = re.sub('[0-9]{2}', ' ## ', title)
    title = re.sub('([0-9]{1})', r" \1 ", title)
    return title


def clean_symbols(text, filters='!¡"@$%&*,.:;<=>?@[\\]^_`{|}~\t\n') -> str:
    for c in filters:
        text = text.replace(c, ' ')
    text = text.replace('\u007f', '')  # for some reason this is not being cleaned up
    return text


def space_characters(text: str, chars: str = '+()-\'') -> str:
    for c in chars:
        text = text.replace(c, ' {} '.format(c))
    return text


def encode_contractions(text: str) -> str:
    text = text + ' '
    text = text.replace(' c/u ', ' cada uno ')
    text = text.replace(' c/', ' con ')
    text = text.replace(' p/', ' para ')
    return text


def _process_title(title: str) -> str:
    title = unidecode.unidecode(title)
    title = title.lower()
    title = clean_symbols(title, filters='#')
    title = clean_numbers(title)
    title = clean_symbols(title)
    title = space_characters(title)
    title = encode_contractions(title)
    title = clean_symbols(title, filters='/')
    title = title.replace('########', '#,#')
    return str(title)


def pre_process_titles(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: Do this with multiprocessing
    df['title'] = df['title'].apply(_process_title)
    return df


def get_tokenizer(titles: pd.Series, vocabulary: dict) -> Tokenizer:
    """
    Get title tokenizer.

    :param titles: Series of titles to train the tokenizer on.
    :param vocabulary: Dictionary mapping (language, vocabulary)
    :return: Tokenizer object
    """
    # TODO: Try Bigrams and Trigrams instead of just using word level.
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(titles)

    oov_words = set()
    inv_words = set()
    final_words = set()

    for w, freq in sorted(tokenizer.word_counts.items(), key=lambda p: p[1], reverse=True):

        # Add any word we have an embedding for to the vocabulary ignoring MIN_FREQ.
        for l, lan_words in vocabulary.items():

            if w in lan_words:
                final_words.add(w)
                inv_words.add(w)
                break
        else:
            if freq > config.MIN_FREQ:
                final_words.add(w)

            oov_words.add((w, freq))

    print('Matched:', len(inv_words), len(tokenizer.word_counts))
    print('Final vocabulary', len(final_words))

    word_index = {e: i for i, e in enumerate(final_words, 1)}
    tokenizer.word_index = word_index

    return tokenizer


def get_padded_sequences(titles: pd.Series, tokenizer: Tokenizer) -> np.array:
    sequences = tokenizer.texts_to_sequences(titles)
    padded_sequences = pad_sequences(sequences, maxlen=config.MAX_SEQUENCE_LENGTH)
    return padded_sequences


def get_label_encoder(categories: pd.Series) -> preprocessing.LabelEncoder:
    le = preprocessing.LabelEncoder()
    le.classes_ = np.array(sorted(categories.unique()))
    return le
