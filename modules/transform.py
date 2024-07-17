"""
Author: Tomb
Date: 3/27/2024
This is the transform.py module.
Usage:
- For Transorm Feature into integer. Headline string > integer
"""

import tensorflow as tf

LABEL_KEY = "clickbait"
FEATURE_KEY = "headline"


def transformed_name(key):
    """Transform feature key

    Args:
        key (str): the key to be transformed

    Returns:
        str: transformed key
    """

    return f"{key}_xf"


def preprocessing_fn(inputs):
    """Preprocess input features into transformed features

    Args:
        inputs (dict): map from feature keys to raw features

    Returns:
        dict: map from features keys to transformed features

    """

    outputs = {}

    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(
        inputs[FEATURE_KEY])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
