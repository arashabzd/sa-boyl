from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import disentanglement_lib
from disentanglement_lib.config.unsupervised_study_v1 import sweep

from . import evaluate
disentanglement_lib.evaluation.evaluate = evaluate


def compute_metrics(
    dataset,
    representation_fn,
    metrics=None,
    output_dir=None,
    overwrite=False,
    seed=0
):
    study = sweep.UnsupervisedStudyV1()
    config_files = study.get_eval_config_files()
    irs_config_file = os.path.join(
        os.path.abspath('./'), 
        'src/evaluation/configs/irs.gin'
    )
    config_files.append(irs_config_file)
    if metrics is None:
        metrics = [
            'dci',
            'sap_score',
            'irs',
            'downstream_task_boosted_trees',
            'downstream_task_logistic_regression',
            'beta_vae_sklearn',
            'factor_vae_metric',
            'mig',
            'modularity_explicitness',
            'unsupervised'
        ]
    
    results_dicts = dict()
    for config_file in config_files:
        metric = config_file.split("/")[-1].replace(".gin", "")
        if  metric not in metrics:
            continue
        
        bindings = [
            "evaluation.random_seed = {}".format(seed),
            "evaluation.name = '{}'".format(metric)
        ]
        
        if output_dir is not None:
            metric_dir = os.path.join(output_dir, metric)
        else:
            metric_dir = None
            
        results_dict = evaluate.evaluate_with_gin(
            dataset,
            representation_fn,
            metric_dir,
            overwrite,
            [config_file],
            bindings
        )
        results_dicts[metric] = results_dict
    return results_dicts