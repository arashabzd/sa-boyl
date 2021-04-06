from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.evaluation.metrics import beta_vae
from disentanglement_lib.evaluation.metrics import dci
from disentanglement_lib.evaluation.metrics import downstream_task
from disentanglement_lib.evaluation.metrics import factor_vae
from disentanglement_lib.evaluation.metrics import irs
from disentanglement_lib.evaluation.metrics import mig
from disentanglement_lib.evaluation.metrics import modularity_explicitness
from disentanglement_lib.evaluation.metrics import reduced_downstream_task
from disentanglement_lib.evaluation.metrics import sap_score
from disentanglement_lib.evaluation.metrics import unsupervised_metrics
from disentanglement_lib.utils import results
import numpy as np
import tensorflow as tf
import gin.tf


def evaluate_with_gin(
    dataset,
    representation_fn,
    output_dir,
    overwrite,
    gin_config_files,
    gin_bindings
):
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
    results_dict = evaluate(dataset, representation_fn)
    gin.clear_config()
    if output_dir is not None:
        if tf.gfile.IsDirectory(output_dir):
            if overwrite:
                tf.gfile.DeleteRecursively(output_dir)
            else:
                raise ValueError('Directory already exists and overwrite is False.')
        results.update_result_directory(output_dir, 'evaluation', results_dict)
    return results_dict


@gin.configurable('evaluation', blacklist=['dataset', 'representation_fn'])
def evaluate(
    dataset,
    representation_fn,
    evaluation_fn=gin.REQUIRED,
    random_seed=gin.REQUIRED,
    name=""
):
    experiment_timer = time.time()
    results_dict = evaluation_fn(
        dataset,
        representation_fn,
        random_state=np.random.RandomState(random_seed)
    )
    results_dict["elapsed_time"] = time.time() - experiment_timer
    return results_dict

