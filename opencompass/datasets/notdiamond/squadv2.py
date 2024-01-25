from typing import Union

from datasets import Dataset

from opencompass.registry import LOAD_DATASET, ICL_EVALUATORS
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.utils.text_postprocessors import general_postprocess

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from notdiamond_server.database import crud
from notdiamond_server.database.initialize import Base

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class NDSQuADV2Dataset(BaseDataset):

    @staticmethod
    def load(db_url: str, size: int, seed: Union[int, str]):
        engine = create_engine(db_url)

        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)

        dataset = []
        with SessionLocal() as db:
            db_samples = crud.get_samples_from_dataset("squadv2", size, db, seed)

            for sample in db_samples:
                dataset.append({
                    'sample_id': sample.id,
                    'query': sample.components["query"].query,
                    'prompt': sample.components["prompt"].prompt,
                    'context': sample.components["context"].context,
                    'label': sample.target['label'],
                })

        dataset = Dataset.from_list(dataset)
        return dataset


@ICL_EVALUATORS.register_module()
class NDSQuADV2Evaluator(BaseEvaluator):
    def __init__(self) -> None:
        self.metric = 'accuracy'
        super().__init__()

    def score(self, predictions, references, sample_ids):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        processed_predictions = []
        for prediction in predictions:
            prediction = prediction.split('\n')[0].lower()
            if 'answer is' in prediction:
                prediction = prediction.split('answer is')[-1]
            prediction = general_postprocess(prediction)
            processed_predictions.append(prediction)
        processed_answers = [[general_postprocess(j).lower() for j in i]
                             for i in references]

        cnt = 0
        sample_accuracy = []
        for pred, cand_ans, id in zip(processed_predictions, processed_answers, sample_ids):
            correct = int(any([cand == pred for cand in cand_ans]))
            cnt += correct

            sample_result = {
                "sample_id": id,
                "score": float(correct)
            }
            sample_accuracy.append(sample_result)

        score = cnt / len(predictions) * 100
        return {'score': score, "sample_score": sample_accuracy}
