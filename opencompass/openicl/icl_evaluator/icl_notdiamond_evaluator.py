import logging
import os
import copy
import random
from typing import List

import evaluate
import numpy as np

from opencompass.registry import ICL_EVALUATORS
from opencompass.utils.text_postprocessors import general_postprocess

from .icl_base_evaluator import BaseEvaluator

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
        self.metric = "accuracy"
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
            label: idx for idx, label in enumerate(set(map(str, references)))
        }
        pred_set = set(predictions)
        for pred in pred_set:
            if str(pred) not in mapping_to_int_dict.keys():
                mapping_to_int_dict[str(pred)] = len(mapping_to_int_dict)
        golds = [mapping_to_int_dict[str(gold)] for gold in references]
        preds = [mapping_to_int_dict[str(pred)] for pred in predictions]
        return {
            "predictions": preds,
            "references": golds,
        }

    def _postprocess(self, scores: dict) -> dict:
        """Postprocess for final scores.

        Args:
            scores (dict): Dict of calculated scores of metrics.

        Returns:
            dict: postprocessed scores.
        """
        scores["accuracy"] *= 100
        return scores

    def _compute_sample_accuracy(
        self, predictions: List, references: List, sample_ids: List
    ) -> dict:
        sample_accuracy = []
        for pred, ref, id in zip(predictions, references, sample_ids):
            score = 1.0 if pred == ref else 0.0
            sample_result = {"sample_id": id, "score": score}
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
                "error": "predictions and references have different "
                f"length. len(predictions): {len(predictions)}, "
                f"len(references): {len(references)}"
            }
        # use codes pre-downloaded to opencompass repo, avoid downloading
        local_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "hf_metrics",
            self.metric + ".py",
        )
        if os.path.exists(local_path):
            metric = evaluate.load(local_path)
        else:
            metric = evaluate.load(self.metric)

        preprocessed = self._preprocess(predictions, references)

        scores = metric.compute(**preprocessed)
        sample_accuracy = self._compute_sample_accuracy(
            **preprocessed, sample_ids=sample_ids
        )

        scores = {**scores, **sample_accuracy}
        result = self._postprocess(scores)

        random.setstate(random_state)
        np.random.set_state(np_random_state)
        return result


@ICL_EVALUATORS.register_module()
class NDMgsmEvaluator(NDAccEvaluator):
    def score(self, predictions: List, references: List, sample_ids: List) -> dict:
        try:
            numeric_preds = [int(pred) if pred else 0 for pred in predictions]
            numeric_references = [int(ref.replace(",", "")) for ref in references]
        except TypeError as terr:
            LOGGER.error(f"Error converting predictions and references to numeric.")
            LOGGER.error(f"PREDICTIONS: \n{predictions}")
            LOGGER.error(f"REFERENCES: \n{references}")
            raise terr
        return super().score(numeric_preds, numeric_references, sample_ids)


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
            for cands in ref["candidates"]:
                if isinstance(cands, str):
                    d = self.dist(pred, cands)
                else:
                    d = np.min([self.dist(pred, cand) for cand in cands])
                dists.append(d)
            preds.append(np.argmin(dists))
            golds.append(ref["label"])

        return {
            "predictions": preds,
            "references": golds,
        }


@ICL_EVALUATORS.register_module()
class NDEMEvaluator(BaseEvaluator):
    """Exact match evaluator."""

    def __init__(self) -> None:
        self.metric = "accuracy"
        super().__init__()

    def _get_processed_answers(self, references):
        return [[general_postprocess(j) for j in i] for i in references]

    def score(self, predictions, references, origin_prompt, sample_ids):
        if len(predictions) != len(references):
            return {"error": "predictions and references have different " "length"}
        origin_predictions = copy.deepcopy(predictions)
        print(f"predictions {predictions} references {references}")
        predictions = [general_postprocess(prediction) for prediction in predictions]
        processed_answers = self._get_processed_answers(references)
        if len(references) != len(processed_answers):
            raise AssertionError(
                f"NDEMEvaluator expected postprocessing of {references} to produce the same # of answers, but found {processed_answers} instead."
            )

        cnt = 0
        details = {}
        sample_accuracy = []
        for i, (pred, ans, origin_ans, origin_pred, prompt, id) in enumerate(
            zip(
                predictions,
                processed_answers,
                references,
                origin_predictions,
                origin_prompt,
                sample_ids,
            )
        ):
            try:
                answers = list(set(ans + origin_ans))
            except TypeError as terr:
                if isinstance(ans, list) and isinstance(origin_ans, str):
                    answers = list(set(ans + [origin_ans]))
                elif isinstance(ans, str) and isinstance(origin_ans, list):
                    answers = list(set([ans] + origin_ans))
                else:
                    raise terr
            detail = {
                "prompt": prompt,
                "pred": pred,
                "answer": answers,
                "origin_prediction": origin_pred,
            }
            if self._check_answer_match(pred, ans, origin_ans):
                cnt += 1
                detail["correct"] = True
                score = 1.0
            else:
                detail["correct"] = False
                score = 0.0

            sample_result = {"sample_id": id, "score": score}
            sample_accuracy.append(sample_result)
            details[f"{i}"] = detail

        score = cnt / len(predictions) * 100

        return {"score": score, "details": details, "sample_score": sample_accuracy}

    def _check_answer_match(self, pred, ans, origin_ans) -> bool:
        return pred in ans or pred in origin_ans


@ICL_EVALUATORS.register_module()
class NDRougeEvaluator(BaseEvaluator):
    """Rouge evaluator.

    Note: this evaluator is not suitable for chinese datasets.
    """

    def __init__(self, seed: int = 0) -> None:
        self.metric = "rouge"
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
            "predictions": predictions,
            "references": references,
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
                sample_result = {"sample_id": id, "score": s}
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
                "error": "predictions and references have different "
                f"length. len(predictions): {len(predictions)}, "
                f"len(references): {len(references)}"
            }
        # use codes pre-downloaded to opencompass repo, avoid downloading
        local_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "hf_metrics",
            self.metric + ".py",
        )
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


@ICL_EVALUATORS.register_module()
class NDMATHEvaluator(BaseEvaluator):

    def __init__(self, *args, **kwargs):
        super(NDMATHEvaluator, self).__init__(*args, **kwargs)
        self.metric = "accuracy"

    def score(self, predictions, references, sample_ids: List[str]) -> dict:
        """
        Calculate sample-level scores (instead of averaged scores) for math
        """
        if len(predictions) != len(references):
            return {"error": "predictions and references have different " "length"}

        results = []

        for sample_id, pred, ref in zip(sample_ids, predictions, references):
            if self.is_equiv(pred, ref):
                score = 1.0
            else:
                score = 0.0
            results.append({"sample_id": sample_id, "score": score})

        return {"sample_score": results}


@ICL_EVALUATORS.register_module()
class NDDropEvaluator(NDEMEvaluator):
    """
    Change scoring fn in NDEMEvaluator to check whether the answer is referenced
    in the LLM prediction.
    """

    def _check_answer_match(self, pred, ans, origin_ans) -> bool:
        answers_found = [
            answer in pred or origin_answer in pred
            for (answer, origin_answer) in zip(ans, origin_ans)
        ]
        return all(answers_found)


@ICL_EVALUATORS.register_module()
class NDBleuEvaluator(BaseEvaluator):
    """Bleu evaluator."""

    def __init__(self, seed: int = 0) -> None:
        self.metric = "sacrebleu"
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
        return scores

    def _compute_sample_accuracy(
        self, predictions: List, references: List, sample_ids: List, metric: evaluate
    ) -> dict:
        sample_accuracy = []
        for pred, ref, id in zip(predictions, references, sample_ids):
            score = metric.compute(**{"predictions": [pred,], "references": [ref,]})
            sample_result = {"sample_id": id, "score": score["score"]}
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
                "error": "predictions and references have different "
                f"length. len(predictions): {len(predictions)}, "
                f"len(references): {len(references)}"
            }
        # use codes pre-downloaded to opencompass repo, avoid downloading
        local_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "hf_metrics",
            self.metric + ".py",
        )
        if os.path.exists(local_path):
            metric = evaluate.load(local_path)
        else:
            metric = evaluate.load(self.metric)

        preprocessed = self._preprocess(predictions, references)

        scores = metric.compute(**preprocessed)
        sample_accuracy = self._compute_sample_accuracy(
            **preprocessed, sample_ids=sample_ids, metric=metric
        )

        scores = {**scores, **sample_accuracy}
        result = self._postprocess(scores)

        random.setstate(random_state)
        np.random.set_state(np_random_state)
        return result