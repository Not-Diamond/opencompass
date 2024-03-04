import json
import random
import os.path as osp
from typing import Union, List

from datasets import Dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, ICL_EVALUATORS

from .read_data import get_samples_from_local_dataset

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class NDGSM8KDataset(BaseDataset):

    @staticmethod
    def load(db_url: str, size: int, seed: Union[int, str]):
        random.seed(seed)
        eval_data_path = osp.join(db_url, "gsm8k.json")

        samples = get_samples_from_local_dataset(eval_data_path, size, seed)

        dataset = []
        for sample_id, sample in samples.items():
            dataset.append({
                'sample_id': sample_id,
                'prompt': sample["components"]["prompt"]["prompt"],
                'query': sample["components"]["query"]["query"],
                'label': sample["target"]['label'],
            })

        # engine = create_engine(db_url)

        # SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        # Base.metadata.create_all(bind=engine)

        # dataset = []
        # with SessionLocal() as db:
        #     db_samples = crud.get_samples_from_dataset("gsm8k", size, db, seed)

        #     for sample in db_samples:
        #         dataset.append({
        #             'sample_id': sample.id,
        #             'prompt': sample.components["prompt"].prompt,
        #             'query': sample.components["query"].query,
        #             'label': sample.target['label'],
        #         })

        dataset = Dataset.from_list(dataset)
        return dataset


@ICL_EVALUATORS.register_module()
class NDGSM8KEvaluator(BaseEvaluator):
    def __init__(self) -> None:
        self.metric = 'accuracy'
        super().__init__()

    def score(self, predictions: List, references: List, sample_ids: List) -> dict:
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        correct = 0
        count = 0
        details = []
        sample_accuracy = []
        for pred, ref, id in zip(predictions, references, sample_ids):
            detail = {'pred': pred, 'answer': ref, 'correct': False}
            count += 1
            if pred == ref:
                correct += 1
                detail['correct'] = True
                acc = 1.
            else:
                acc = 0.

            sample_result = {
                "sample_id": id,
                "score": acc
            }

            sample_accuracy.append(sample_result)
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details, 'sample_score': sample_accuracy}
        return result
