import pandas as pd
from typing import Tuple

import multilingual_title_classifier.src.config as config
from multilingual_title_classifier.src.path_helpers import get_resources_path


def get_training_test_set() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get pandas dataframes of the training and test sets.

    :return: Pair of pandas dataframes.
    """
    training_set = pd.read_csv(get_resources_path('train.csv'))
    test_set = pd.read_csv(get_resources_path('test.csv'))

    if config.DATASET_SIZE:
        training_set = training_set.sample(config.DATASET_SIZE)
    else:
        training_set = training_set.sample(frac=1, random_state=42)

    print('Training set dimensions:', training_set.shape)
    print('Test set dimensions:', test_set.shape)

    return training_set, test_set
