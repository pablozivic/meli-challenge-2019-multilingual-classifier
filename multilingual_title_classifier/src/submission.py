import os
import json
import numpy as np
from datetime import datetime
from scipy.stats import entropy
from keras.models import load_model
from keras_preprocessing.text import tokenizer_from_json

import multilingual_title_classifier.src.config as config
import multilingual_title_classifier.src.vocab as vocab
import multilingual_title_classifier.src.dataset as data
from multilingual_title_classifier.src.path_helpers import get_path
import multilingual_title_classifier.src.utils as utils

utils.check_gpu_usage()
utils.make_deterministic()
timestamp = datetime.now().strftime("%d_%m_%Y_%Hh_%Mm_%Ss")

# load and pre-process data
dataset, submission_set = data.get_training_test_set()
forensics = submission_set.copy()
submission_set = vocab.pre_process_titles(submission_set)

# load category encoder
label_encoder = vocab.get_label_encoder(dataset['category'])
labels = label_encoder.transform(dataset['category'])

# reload tokenizer
with open(get_path('tokenizer.json')) as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

# tokenize submission set
sequences_submission = vocab.get_padded_sequences(submission_set['title'], tokenizer)

# ensemble prediction
yhat_ensemble = None
qty_ensembled_models = 0
for i in range(1, config.N_SPLITS + 1):
    model_path = get_path('stage_2_fold_{}.ckpt'.format(i), dirs=['checkpoints'])

    if not os.path.exists(model_path):
        print('Checkpoint does not exist: {}'.format(model_path))
        continue

    print('Ensembling fold {}'.format(i))

    model = load_model(model_path)

    fold_probs = model.predict_proba(sequences_submission)

    if yhat_ensemble is None:
        yhat_ensemble = fold_probs.astype('float64')
    else:
        yhat_ensemble += fold_probs

    qty_ensembled_models += 1

yhat_ensemble /= qty_ensembled_models

# forensics
entropy = [round(entropy(row), 5) for row in yhat_ensemble]
predicted_classes = np.argmax(yhat_ensemble, axis=1)
forensics['transformed_title'] = submission_set['title']
forensics['category'] = label_encoder.inverse_transform(predicted_classes)
forensics['entropy'] = entropy
forensics = forensics.sort_values('entropy', ascending=False)
cols = ['id', 'title', 'transformed_title', 'category', 'entropy']
filename = 'forensics_{}_ensemble_{}.csv'.format(timestamp, qty_ensembled_models)
forensics[cols].to_csv(get_path(filename, dirs=['forensics']),
                       index=False,
                       header=True)

# saving final submission
predicted_classes = np.argmax(yhat_ensemble, axis=1)
submission_set['category'] = label_encoder.inverse_transform(predicted_classes)

filename = 'submission_{}_ensemble_{}.csv'.format(timestamp, qty_ensembled_models)
submission_set[['id', 'category']].to_csv(get_path(filename, dirs=['submissions']),
                                          index=False,
                                          header=True)
