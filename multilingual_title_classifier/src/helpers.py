import os
import numpy as np
import pandas as pd
import keras.backend as K
from sklearn.metrics import balanced_accuracy_score
from multilingual_title_classifier.src.path_helpers import get_path, get_resources_path


def predict_with_uncertainty(model, x, y, num_classes, beta=0.4, n_iter=100) -> None:
    """
    Predict using dropout ensemble, failed attempt to reduce variance.

    Motivated by:
    https://github.com/keras-team/keras/issues/9412
    https://arxiv.org/pdf/1506.02142.pdf
    """
    f = K.function(model.inputs + [K.learning_phase()], model.outputs)
    preds = model.predict(x, verbose=1)
    avg_preds = np.zeros((x.shape[0], num_classes))

    for i in range(n_iter):
        avg_preds += np.concatenate([f((s, 1))[0] for s in np.array_split(x, 300)])
        final_preds = beta * preds + (1 - beta) * avg_preds / (i + 1)
        predicted_class = np.argmax(final_preds, axis=1)
        print(balanced_accuracy_score(y, predicted_class))


def get_transfer_categories(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Get categories where we will probably require transfer learning between languages.
    """
    category_counts = (dataset
                       .groupby(['category', 'language'])
                       .count()
                       .reset_index()
                       .pivot(index='category',
                              columns='language',
                              values='title').reset_index())
    return category_counts[category_counts['portuguese'].isna() | category_counts['spanish'].isna()]


def distribution_analysis(submission_fname: str) -> pd.DataFrame:
    """
    Analyze distribution mismatch between training and test set.

    :param submission_fname: Filename of submission.
    :return: Pandas dataframe with distribution analysis.
    """
    training_distribution = pd.read_csv(get_resources_path('train.csv')).groupby('category').count()[['title']]
    training_distribution = training_distribution.rename(columns={"title": "train_count"})
    training_distribution['pmf_train'] = training_distribution[
                                             'train_count'] / training_distribution.train_count.sum() * 100

    submission_distribution = pd.read_csv(get_path(submission_fname, dirs=['submissions'])).groupby('category').count()
    submission_distribution = submission_distribution.rename(columns={"id": "val_count"})
    submission_distribution['pmf_val'] = submission_distribution[
                                             'val_count'] / submission_distribution.val_count.sum() * 100

    dist_comp = submission_distribution.join(training_distribution)
    dist_comp['dif'] = dist_comp['pmf_val'] - dist_comp['pmf_train']
    return dist_comp.sort_values('dif')


def ensemble_analyzer() -> None:
    """
    Ensemble analyzer, useful to avoid multiple submissions where not many predictions change.

    :return: Pandas dataframe with observations that have changed.
    """
    test_set = pd.read_csv(get_resources_path('test.csv'))
    base_ensemble = None
    for filepath in sorted(list(os.walk(get_path(dirs=['submissions'])))[0][2]):
        if 'ensemble' in filepath:
            if base_ensemble is None:
                base_ensemble = pd.read_csv(get_path(filepath, dirs=['submissions']))
            else:
                print('Analyzing ensemble {}'.format(filepath))
                current_ensemble = pd.read_csv(get_path(filepath, dirs=['submissions']))
                merged_df = pd.merge(base_ensemble, current_ensemble, suffixes=('_base', '_curr'), on='id', how='inner')
                dif = merged_df.category_base != merged_df.category_curr
                print('Different predictions:', np.sum(dif))
                base_ensemble = current_ensemble

    return pd.merge(merged_df[merged_df.category_base != merged_df.category_curr], test_set, on='id', how='inner')
