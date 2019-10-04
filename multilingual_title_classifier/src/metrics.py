import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from multilingual_title_classifier.src.path_helpers import get_path


def get_balanced_accuracy(model, x_val: np.array, y_val: np.array, reliability: np.array) -> dict:
    """
    Returns the balanced accuracy for different model configurations.

    :param model: Model with .predict method.
    :param x_val: Input data
    :param y_val: Expected output
    :param reliability: Boolean matrix with observation reliability.
    :return: Dictionary with metrics.
    """
    # get predictions
    yhat_val = model.predict(x_val, verbose=1)

    # balanced accuracy
    yhat_class = np.argmax(yhat_val, axis=1)
    val_accuracy = accuracy_score(y_val, yhat_class)
    ba_unweighted = balanced_accuracy_score(y_val, yhat_class)

    # balanced accuracy reliable
    ba_unweighted_reliable = balanced_accuracy_score(y_val[reliability], yhat_class[reliability])

    return {
        'accuracy': val_accuracy,
        'balanced_accuracy': ba_unweighted,
        'balanced_accuracy_reliable': ba_unweighted_reliable
    }


def get_model_metrics() -> pd.DataFrame:
    metrics_path = get_path(dirs=['metrics'])

    dfs = []
    for filename in sorted(os.listdir(metrics_path)):
        with open(get_path(filename, dirs=['metrics']), 'r') as fp:
            d = json.load(fp)
            d = {filename: d}
            dfs.append(pd.DataFrame.from_dict(d))

    return pd.concat(dfs, axis=1, sort=False)
