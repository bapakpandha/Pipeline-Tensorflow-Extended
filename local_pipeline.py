"""
Author: Tomb
Date: 3/30/2024
This is the local_pipeline.py module.
Usage:
    this is the main of starting pipeline for running
    pipelne. This module will create pipeline environtment
    and initialize it.
"""

import os
from typing import Text
from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

PIPELINE_NAME = "sitomb-pipeline"

# pipeline inputs
DATA_ROOT = "dataset"
TRANSFORM_MODULE_FILE = "modules/transform.py"
TRAINER_MODULE_FILE = "modules/trainer.py"
TUNER_MODULE_FILE = "modules/tuner.py"

# pipeline outputs
OUTPUT_BASE = "output"
serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")


def init_local_pipeline(component_arg, pipeline_root_arg: Text
                        ) -> pipeline.Pipeline:
    """
    init_local_pipeline that start everything

    Init local pipeline for initialize pipeline
    and defined environtment above (Output_Base dkk)

    Returns:
    TFX pipeline sent to apache beam
    """

    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        "--direct_running_mode=multi_processing"
        # 0 auto-detect based on on the number of CPUs available
        # during execution time.
        "----direct_num_workers=0"
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root_arg,
        components=component_arg,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        eam_pipeline_args=beam_args
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    from modules.components import init_components

    components = init_components({
        "data_dir": DATA_ROOT,
        "training_module": TRAINER_MODULE_FILE,
        "transform_module": TRANSFORM_MODULE_FILE,
        "tuner_module": TUNER_MODULE_FILE,
        "training_steps": 5000,
        "eval_steps": 1000,
        "serving_model_dir": serving_model_dir
    })

    pipeline = init_local_pipeline(components, pipeline_root)
    BeamDagRunner().run(pipeline=pipeline)
