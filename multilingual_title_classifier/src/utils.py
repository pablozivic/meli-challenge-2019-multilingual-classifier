import os
import random
import numpy as np
import tensorflow as tf
from keras import backend as K


def check_gpu_usage() -> None:
    available_gpus = K.tensorflow_backend._get_available_gpus()

    if not len(available_gpus):
        raise Warning('No GPUs were found, using CPU instead')


def use_cpu() -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def make_deterministic() -> None:
    # Taken from: https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras
    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value = 0

    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


def dict_to_json_serializable(d: dict) -> dict:
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_json_serializable(v)
        elif isinstance(v, (list, np.ndarray)):
            d[k] = list(map(lambda i: round(float(i), 5), v))
        elif isinstance(v, float):
            d[k] = round(v, 5)
        elif not isinstance(v, str):
            print(type(v))
    return d
