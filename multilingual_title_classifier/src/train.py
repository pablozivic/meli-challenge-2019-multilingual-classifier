import os
import gc
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras_preprocessing.text import tokenizer_from_json
from sklearn.model_selection import KFold

import multilingual_title_classifier.src.config as config
import multilingual_title_classifier.src.embeddings as embeddings
import multilingual_title_classifier.src.vocab as vocab
import multilingual_title_classifier.src.dataset as data
import multilingual_title_classifier.src.model as text_model
import multilingual_title_classifier.src.metrics as metrics

from multilingual_title_classifier.src.path_helpers import get_path
import multilingual_title_classifier.src.utils as utils

utils.check_gpu_usage()
utils.make_deterministic()

# load and pre-process data
print('Loading dataset')
dataset, submission_set = data.get_training_test_set()

print('Pre-processing titles')
dataset = vocab.pre_process_titles(dataset)
submission_set = vocab.pre_process_titles(submission_set)

# get vocabulary and encoders
vocabulary = embeddings.get_embeddings_vocabulary()
label_encoder = vocab.get_label_encoder(dataset['category'])

# try to load tokenizer from file
tokenizer_path = get_path('tokenizer.json')
if os.path.exists(tokenizer_path):
    with open(tokenizer_path) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    print('Loaded tokenizer from file')
else:
    tokenizer = vocab.get_tokenizer(pd.concat([dataset['title'], submission_set['title']]), vocabulary)
    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

# pre-process dataset for training
labels = label_encoder.transform(dataset['category'])

# encode and pad sequences
sequences_dataset = vocab.get_padded_sequences(dataset['title'], tokenizer)
sequences_submission = vocab.get_padded_sequences(submission_set['title'], tokenizer)

print('Shape of data tensor:', sequences_dataset.shape)
print('Shape of label tensor:', labels.shape)

# load pre-traianed embeddings and then model
pretrained_embeddings = embeddings.get_embeddings_matrix(tokenizer.word_index)

# split the data into a training set and a validation set
kf = KFold(n_splits=config.N_SPLITS, random_state=42, shuffle=False)  # TODO: Try a StratifiedShuffle
num_fold = 1

for train_index, validation_index in kf.split(sequences_dataset):
    stage_2_path = get_path('stage_2_fold_{}.ckpt'.format(num_fold), dirs=['checkpoints'])
    if os.path.exists(stage_2_path):
        print('Checkpoint for fold {} already exists'.format(num_fold))
        num_fold += 1
        continue

    x_train = sequences_dataset[train_index]
    y_train = np.array(labels)[train_index]
    x_val = sequences_dataset[validation_index]
    y_val = np.array(labels)[validation_index]
    x_submission = sequences_submission

    assert (len(y_train) != len(labels))

    # keep reliability array to compute metrics later
    reliability_val = np.array(dataset['label_quality'][validation_index]) == 'reliable'

    model = text_model.get_model(pretrained_embeddings, num_clasases=len(label_encoder.classes_))

    print('Running model for fold {}'.format(num_fold))
    start_time = time.time()

    # stage 1: fit model with frozen multilingual pre-trained embeddings.
    print('Running stage 1')
    stage_1_path = get_path('stage_1_fold_{}.ckpt'.format(num_fold), dirs=['checkpoints'])
    if os.path.exists(stage_1_path):
        print('Checkpoint found for stage 1')
        model = load_model(stage_1_path)
    else:
        model = text_model.get_model(pretrained_embeddings, num_clasases=len(label_encoder.classes_))
        # TODO: Try Stratified custom sampler for each batch
        model.fit(x_train, y_train,
                  validation_data=(x_val, y_val),
                  batch_size=config.BATCH_SIZE,
                  epochs=25,
                  verbose=2,
                  callbacks=[
                      EarlyStopping(monitor='val_acc', patience=5, verbose=2, mode='max', restore_best_weights=True)
                  ])
        model.save(stage_1_path)

    # stage 2: fine tune pre-trained embedding and learn missing embeddings.
    print('Running stage 2')
    model = text_model.make_embedding_trainable(model)
    model.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              batch_size=config.BATCH_SIZE,
              epochs=1000,
              verbose=2,
              callbacks=[
                  EarlyStopping(monitor='val_acc', patience=5, verbose=2, mode='max', restore_best_weights=True)
              ])
    model.save(stage_2_path)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print('Calculating balanced accuracy')

    m = metrics.get_balanced_accuracy(model, x_val, y_val, reliability_val)
    m['elapsed_time'] = str(timedelta(seconds=int(elapsed_time)))
    print('Metrics:', m)

    m['history'] = model.history.history
    m = utils.dict_to_json_serializable(m)

    with open(get_path('metrics_fold_{}.json'.format(num_fold), dirs=['metrics']), 'w', encoding='utf-8') as f:
        f.write(json.dumps(m))

    # save predictions
    print('Computing predictions')
    yhat = model.predict(x_submission, verbose=2)
    predicted_classes = np.argmax(yhat, axis=1)
    submission_set['category'] = label_encoder.inverse_transform(predicted_classes)

    timestamp = datetime.now().strftime("%d_%m_%Y_%Hh_%Mm_%Ss")
    filename = 'submission_{}_fold_{}.csv'.format(timestamp, num_fold)
    submission_set[['id', 'category']].to_csv(get_path(filename, dirs=['submissions']),
                                              index=False,
                                              header=True)

    num_fold += 1

    # run garbage collector
    del model
    gc.collect()
