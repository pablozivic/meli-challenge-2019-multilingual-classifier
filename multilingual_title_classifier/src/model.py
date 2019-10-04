import numpy as np
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization

import multilingual_title_classifier.src.config as config


def get_model(embedding_matrix: np.array, num_classes: int) -> keras.Model:
    embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                                output_dim=embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                trainable=False)

    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.45))  # TODO: Use SpatialDropout1D instead
    model.add(BatchNormalization())
    model.add(LSTM(256, recurrent_dropout=0.1, unroll=True))
    # TODO: Try to use return_sequences with pooling layers to reduce dimensionality
    #  See: https://sigir-ecom.github.io/ecom18DCPapers/ecom18DC_paper_9.pdf
    model.add(Dropout(0.35))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  # TODO: Try other learning schedules, see: https://arxiv.org/abs/1708.07120
                  optimizer=Adam(lr=config.LR,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 amsgrad=config.AMSGRAD),
                  metrics=['acc'])

    return model


def make_embedding_trainable(model) -> keras.Model:
    model.layers[0].trainable = True
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(lr=config.LR,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 amsgrad=config.AMSGRAD),
                  metrics=['acc'])
    return model
