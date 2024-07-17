"""
Author: Tomb
Date: 3/30/2024
This module provides auto hyperparameter tuning for TensorFlow
models in TFX pipelines using KerasTuner.
Usage:
    Use within a TFX pipeline component
    for optimizing model
    parameters based on specified
    metrics and trials.
"""
from typing import NamedTuple, Dict, Text, Any
from tfx.components.trainer.fn_args_utils import FnArgs
from kerastuner.engine import base_tuner
from kerastuner import Hyperband, Objective
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_transform as tft
from transform import (LABEL_KEY, FEATURE_KEY, transformed_name)

# Constants
NUM_EPOCHS = 1
MAX_TOKEN = 5000
SEQUENCE_LENGTH = 100

# NamedTuple for Tuner Function Result
TunerResult = NamedTuple("TunerResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])

# Early stopping callback
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_binary_accuracy",
    mode="max",
    verbose=1,
    patience=5,
)


def gzip_reader(filenames):
    """
    Creates a TFRecordDataset to read TFRecord files with GZIP compression.
        This function takes a list of filenames pointing to
        TFRecord files compressed using GZIP and returns a `tf.data.TFRecordDataset`
        object configured to read them.

    Parameters:
    - filenames (list of str or str): A list of filenames (paths) of the
        TFRecord files to read. If a single filename is provided,
        it is wrapped in a list automatically.
    """
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_data(file_pattern, transform_output, num_epochs, batch_size=64):
    """
    Prepares a batched dataset from TFRecord files using the provided transformation schema.
     Parameters:
    - file_pattern (str): A string pattern that matches the TFRecord files to read.
    - transform_output: object containing the transformation schema 
    - num_epochs (int): The number of times the dataset should be repeated. 
    - batch_size (int): The number of records to combine in a single batch.
    """
    transform_feature_spec = (
        transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset

# Model Building Function


def build_model(hp, vectorizer_layer):
    """
    Create model based
    """
    num_hidden_layers = hp.Choice("num_hidden_layers", values=[1, 2, 3])
    embed_dims = hp.Int("embed_dims", min_value=16, max_value=256, step=32)
    lstm_units = hp.Int("lstm_units", min_value=32, max_value=256, step=32)
    dense_units = hp.Int("dense_units", min_value=64, max_value=512, step=64)
    dropout_rate = hp.Float(
        "dropout_rate",
        min_value=0.2,
        max_value=0.5,
        step=0.1)
    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4, 1e-5])

    inputs = tf.keras.Input(
        shape=(
            1,
        ),
        name=transformed_name(FEATURE_KEY),
        dtype=tf.string)

    x = vectorizer_layer(inputs)
    x = layers.Embedding(input_dim=5000, output_dim=embed_dims)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units))(x)

    for _ in range(num_hidden_layers):
        x = layers.Dense(dense_units, activation=tf.nn.relu)(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation=tf.nn.sigmoid)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["binary_accuracy"],
    )

    return model

# Tuner Function


def tuner_fn(fn_args: FnArgs):
    """
    main tuner function
    """
    transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_data = input_data(
        fn_args.train_files[0], transform_output, NUM_EPOCHS
    )
    eval_data = input_data(
        fn_args.eval_files[0], transform_output, NUM_EPOCHS
    )

    vectorizer_dataset = train_data.map(
        lambda f, l: f[transformed_name(FEATURE_KEY)]
    )

    vectorizer_layer = layers.TextVectorization(
        max_tokens=MAX_TOKEN,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
    )
    print("DEBUG: Start_vectorizer_adapt")
    vectorizer_layer.adapt(vectorizer_dataset)
    print("DEBUG: Finish Vectorizer Adapt")

    tuner = Hyperband(
        hypermodel=lambda hp: build_model(hp, vectorizer_layer),
        objective=Objective('val_binary_accuracy', direction='max'),
        max_epochs=NUM_EPOCHS,
        factor=3,
        directory=fn_args.working_dir,
        project_name="clickbait_detection_hyperband_kt",
    )

    return TunerResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [early_stopping_cb],
            "x": train_data,
            "validation_data": eval_data,
            "steps_per_epoch": 50,
            "validation_steps": 25,
        },
    )
