import numpy as np
import optuna
from optuna.study import Study
from optuna.trial import Trial
from typing import Dict, Any
from ghosttuner.prompting import ConversationManager


class GhosttunerSampler(optuna.samplers.BaseSampler):
    def __init__(self, model_type: str, search_space: str, budget: int = 10, use_cot: bool = False):
        self.manager = ConversationManager(
            model_type="mlp", search_space=search_space, budget=budget, use_cot=use_cot
        )
        self.initial_params = None

    def sample_relative(self, study: Study, trial: Trial, search_space: Dict) -> Dict[str, Any]:
        if search_space == {}:
            return {}

        try:
            previous_trial = study.get_trials(deepcopy=False)[-2]
            previous_loss = previous_trial.value
        except IndexError:
            previous_loss = None

        return self.manager.sample_hyperparameters(previous_loss)

    def sample_independent(self, study, trial, param_name, param_distribution):
        """
        During the execution of the objective function, sample_independent() is used to
        sample parameters that don’t belong to the relative search space.

        Typically this happens in the first trial of the study, for example. In the case of
        GhostTuner, we dont really want this to happen... Need to find a workaround.
        """
        if trial.number == 0:
            if not self.initial_params:
                self.initial_params = self.manager.sample_hyperparameters()
            return self.initial_params[param_name]

        independent_sampler = optuna.samplers.RandomSampler()
        return independent_sampler.sample_independent(study, trial, param_name, param_distribution)

    def infer_relative_search_space(self, study, trial):
        """Return the intersection of the parameter distributions that have been suggested in prior completed trials"""
        return optuna.search_space.intersection_search_space(study.get_trials(deepcopy=False))
