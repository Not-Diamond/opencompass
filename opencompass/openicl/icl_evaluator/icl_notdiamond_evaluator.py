import os
import random
from typing import List

import evaluate
import numpy as np

from opencompass.registry import ICL_EVALUATORS
from opencompass.utils.text_postprocessors import general_postprocess

from .icl_base_evaluator import BaseEvaluator


@ICL_EVALUATORS.register_module()
class NDAccEvaluator(BaseEvaluator):
    """Use huggingface evaluate module to calculate the target metrics.

    Args:
        metric (str): Metric name in evaluate module.
        seed (int): There exists some randomness during the calculation of some
            metrics, thus we set a fixed random seed for reproducing. Defaults
            to 0.
    """

    def __init__(self, seed: int = 0) -> None:
        self.metric = 'accuracy'
        self.seed = seed
        super().__init__()

    def _preprocess(self, predictions: List, references: List) -> dict:
        """Preprocess the final predictions and references to needed format.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: preprocessed results.
        """
        mapping_to_int_dict = {
            label: idx
            for idx, label in enumerate(set(map(str, references)))
        }
        pred_set = set(predictions)
        for pred in pred_set:
            if str(pred) not in mapping_to_int_dict.keys():
                mapping_to_int_dict[str(pred)] = len(mapping_to_int_dict)
        golds = [mapping_to_int_dict[str(gold)] for gold in references]
        preds = [mapping_to_int_dict[str(pred)] for pred in predictions]
        return {
            'predictions': preds,
            'references': golds,
        }

    def _postprocess(self, scores: dict) -> dict:
        """Postprocess for final scores.

        Args:
            scores (dict): Dict of calculated scores of metrics.

        Returns:
            dict: postprocessed scores.
        """
        scores['accuracy'] *= 100
        return scores

    def _compute_sample_accuracy(self, predictions: List, references: List, sample_ids: List) -> dict:
        sample_accuracy = []
        for pred, ref, id in zip(predictions, references, sample_ids):
            score = 1. if pred == ref else 0.
            sample_result = {
                "sample_id": id,
                "score": score
            }
            sample_accuracy.append(sample_result)
        return {"sample_score": sample_accuracy}

    def score(self, predictions: List, references: List, sample_ids: List) -> dict:
        """Calculate scores.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: calculated scores.
        """
        random_state = random.getstate()
        np_random_state = np.random.get_state()

        random.seed(self.seed)
        np.random.seed(self.seed)
        if len(predictions) != len(references):
            return {
                'error':
                'predictions and references have different '
                f'length. len(predictions): {len(predictions)}, '
                f'len(references): {len(references)}'
            }
        # use codes pre-downloaded to opencompass repo, avoid downloading
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'hf_metrics', self.metric + '.py')
        if os.path.exists(local_path):
            metric = evaluate.load(local_path)
        else:
            metric = evaluate.load(self.metric)

        preprocessed = self._preprocess(predictions, references)

        scores = metric.compute(**preprocessed)
        sample_accuracy = self._compute_sample_accuracy(**preprocessed, sample_ids=sample_ids)

        scores = {**scores, **sample_accuracy}
        result = self._postprocess(scores)

        random.setstate(random_state)
        np.random.set_state(np_random_state)
        return result


@ICL_EVALUATORS.register_module()
class NDEDAccEvaluator(NDAccEvaluator):
    """Edit distance based accuracy evaluator.

    This implementation requires the un-postprocessed outputs from the model,
    and the reference list where each item is structured as:

    .. code-block:: python

        {
            'candidates': [],  # a list of informative answer candidates
            'label': 0,  # the index of the gold answer
        }

    It always matches the model's output to a valid answer with the citerion
    as the minimum editing distance.
    """

    def __init__(self) -> None:
        super().__init__()
        from rapidfuzz.distance import Levenshtein
        self.dist = Levenshtein.distance

    def _preprocess(self, predictions: List, references: List) -> dict:
        """Preprocess the final predictions and references to needed format.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: preprocessed results.
        """

        preds = []
        golds = []

        for i in range(len(predictions)):
            pred, ref = predictions[i], references[i]
            dists = []
            for cands in ref['candidates']:
                if isinstance(cands, str):
                    d = self.dist(pred, cands)
                else:
                    d = np.min([self.dist(pred, cand) for cand in cands])
                dists.append(d)
            preds.append(np.argmin(dists))
            golds.append(ref['label'])

        return {
            'predictions': preds,
            'references': golds,
        }


@ICL_EVALUATORS.register_module()
class NDEMEvaluator(BaseEvaluator):
    """Exact match evaluator."""

    def __init__(self) -> None:
        self.metric = 'accuracy'
        super().__init__()

    def score(self, predictions, references, sample_ids):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        predictions = [
            general_postprocess(prediction) for prediction in predictions
        ]
        processed_answers = [[general_postprocess(j) for j in i]
                             for i in references]

        cnt = 0
        details = []
        sample_accuracy = []
        for pred, ans, origin_ans, id in zip(predictions, processed_answers,
                                             references, sample_ids):
            answers = list(set(ans + origin_ans))
            detail = {'pred': pred, 'answer': answers}
            if pred in ans or pred in origin_ans:
                cnt += 1
                detail['correct'] = True
                score = 1.
            else:
                detail['correct'] = False
                score = 0.

            sample_result = {
                "sample_id": id,
                "score": score
            }
            sample_accuracy.append(sample_result)
            details.append(detail)

        score = cnt / len(predictions) * 100

        return {'score': score, 'details': details, 'sample_score': sample_accuracy}


@ICL_EVALUATORS.register_module()
class NDRougeEvaluator(BaseEvaluator):
    """Rouge evaluator.

    Note: this evaluator is not suitable for chinese datasets.
    """

    def __init__(self, seed: int = 0) -> None:
        self.metric = 'rouge'
        self.seed = seed
        super().__init__()

    def _preprocess(self, predictions: List, references: List) -> dict:
        """Preprocess the final predictions and references to needed format.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: preprocessed results.
        """
        return {
            'predictions': predictions,
            'references': references,
        }

    def _postprocess(self, scores: dict) -> dict:
        """Postprocess for final scores.

        Args:
            scores (dict): Dict of calculated scores of metrics.

        Returns:
            dict: postprocessed scores.
        """
        result = {k: np.mean(v) * 100 for k, v in scores.items()}
        return result

    def _compute_sample_score(self, scores: dict, sample_ids: List) -> dict:
        sample_scores = {}
        for k, v in scores.items():
            sample_accuracy = []
            for id, s in zip(sample_ids, v):
                sample_result = {
                    "sample_id": id,
                    "score": s
                }
                sample_accuracy.append(sample_result)
            sample_scores[k] = sample_accuracy
        return {"sample_score": sample_scores}

    def score(self, predictions: List, references: List, sample_ids: List) -> dict:
        """Calculate scores.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.
            sample_ids (List): List of sample ids in db.

        Returns:
            dict: calculated scores.
        """
        random_state = random.getstate()
        np_random_state = np.random.get_state()

        random.seed(self.seed)
        np.random.seed(self.seed)
        if len(predictions) != len(references):
            return {
                'error':
                'predictions and references have different '
                f'length. len(predictions): {len(predictions)}, '
                f'len(references): {len(references)}'
            }
        # use codes pre-downloaded to opencompass repo, avoid downloading
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'hf_metrics', self.metric + '.py')
        if os.path.exists(local_path):
            metric = evaluate.load(local_path)
        else:
            metric = evaluate.load(self.metric)

        preprocessed = self._preprocess(predictions, references)

        scores = metric.compute(**preprocessed, use_aggregator=False)
        sample_accuracy = self._compute_sample_score(scores, sample_ids=sample_ids)

        scores = self._postprocess(scores)
        result = {**scores, **sample_accuracy}

        random.setstate(random_state)
        np.random.set_state(np_random_state)
        return result
