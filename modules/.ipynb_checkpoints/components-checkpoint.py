"""
Author: Tomb
Date: 3/30/2024
This is the components.py module.
Usage:
    this is the main of starting pipeline
    components for running
    pipeline.
"""
import os
import tfx
import tensorflow_model_analysis as tfma
from tfx.components import (CsvExampleGen, StatisticsGen,
                    SchemaGen, ExampleValidator, Transform,
                    Tuner, Trainer, Evaluator, Pusher)
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy
)

LABEL_KEY = "clickbait"
FEATURE_KEY = "headline"

def init_components(arguments):
    """Initiate tfx pipeline components

    arguments is dictionary include:
        data_dir (str): a path to the data
        transform_module (str): a path to the transform_module
        training_module (str): a path to the transform_module
        training_steps (int): number of training steps
        eval_steps (int): number of eval steps
        serving_model_dir (str): a path to the serving model directory

    Returns:
        TFX components
    """

    example_gen = CsvExampleGen(
        input_base=arguments["data_dir"]
    )

    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"]
    )

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath(arguments["transform_module"])
    )

    tuner = Tuner(
        module_file=os.path.abspath(arguments["tuner_module"]),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=tfx.v1.proto.TrainArgs(
            splits=["train"],
            num_steps=arguments["training_steps"],
        ),
        eval_args=tfx.v1.proto.EvalArgs(
            splits=["eval"],
            num_steps=arguments["eval_steps"],
        ),
    )

    trainer = Trainer(
        module_file=os.path.abspath(arguments["training_module"]),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        hyperparameters=tuner.outputs["best_hyperparameters"],
        train_args=tfx.v1.proto.TrainArgs(
            splits=['train'],
            num_steps=arguments["training_steps"]),
        eval_args=tfx.v1.proto.EvalArgs(
            splits=['eval'],
            num_steps=arguments["eval_steps"])
    )

    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')

    slicing_specs = [
        tfma.SlicingSpec(),
        tfma.SlicingSpec(feature_keys=FEATURE_KEY)
    ]

    metrics_specs = [
        tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='AUC'),
                tfma.MetricConfig(class_name="Precision"),
                tfma.MetricConfig(class_name="Recall"),
                tfma.MetricConfig(class_name="ExampleCount"),
                tfma.MetricConfig(class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.5}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': 0.0001})
                        )
                )
            ])
    ]

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key=LABEL_KEY)],
        slicing_specs=slicing_specs,
        metrics_specs=metrics_specs
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)

    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=tfx.v1.proto.PushDestination(
            filesystem=tfx.v1.proto.PushDestination.Filesystem(
                base_directory=arguments["serving_model_dir"]
            )
        ),
    )

    components = (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher
    )
    return components
