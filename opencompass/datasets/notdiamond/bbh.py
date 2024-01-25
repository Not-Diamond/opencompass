from typing import List, Union

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from notdiamond_server.database import crud
from notdiamond_server.database.initialize import Base

from ..base import BaseDataset
from ..bbh import bbh_freeform_postprocess


@LOAD_DATASET.register_module()
class NDBBHDataset(BaseDataset):

    @staticmethod
    def load(subset: str, db_url: str, size: int, seed: Union[int, str]):
        engine = create_engine(db_url)

        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)

        dataset = []
        with SessionLocal() as db:
            db_samples = crud.get_samples_from_dataset(f"bbh.{subset}", size, db, seed)

            for sample in db_samples:
                dataset.append({
                    'sample_id': sample.id,
                    'prompt': sample.components['prompt'].prompt,
                    'query': sample.components["query"].query,
                    'label': sample.target['label'],
                })

        dataset = Dataset.from_list(dataset)
        return dataset


@ICL_EVALUATORS.register_module()
class NDBBHEvaluator(BaseEvaluator):
    def __init__(self) -> None:
        self.metric = 'accuracy'
        super().__init__()

    def score(self, predictions: List, references: List, sample_ids: List) -> dict:
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        predictions = [bbh_freeform_postprocess(pred) for pred in predictions]

        details = []
        sample_accuracy = []
        cnt = 0
        for pred, ref, id in zip(predictions, references, sample_ids):
            detail = {'pred': pred, 'answer': ref, 'correct': False}
            if pred == ref:
                cnt += 1
                detail['correct'] = True
                acc = 1.
            else:
                acc = 0.

            sample_result = {
                "sample_id": id,
                "score": acc
            }

            details.append(detail)
            sample_accuracy.append(sample_result)

        score = cnt / len(predictions) * 100

        return {'score': score, 'details': details, "sample_score": sample_accuracy}


@ICL_EVALUATORS.register_module()
class NDBBHEvaluator_mcq(BaseEvaluator):
    def __init__(self) -> None:
        self.metric = 'accuracy'
        super().__init__()

    def score(self, predictions: List, references: List, sample_ids: List) -> dict:
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        details = []
        sample_accuracy = []
        cnt = 0
        for pred, ref, id in zip(predictions, references, sample_ids):
            detail = {'pred': pred, 'answer': ref, 'correct': False}
            if pred == ref:
                cnt += 1
                detail['correct'] = True
                acc = 1.
            else:
                acc = 0.

            sample_result = {
                "sample_id": id,
                "score": acc
            }

            details.append(detail)
            sample_accuracy.append(sample_result)

        score = cnt / len(predictions) * 100

        return {'score': score, 'details': details, "sample_score": sample_accuracy}
