import json
import tempfile
import os.path as osp
from typing import List, Union

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET, ICL_EVALUATORS

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from notdiamond_server.database import crud
from notdiamond_server.database.initialize import Base

from ...base import BaseDataset
from .humaneval_execution import evaluate_functional_correctness


@LOAD_DATASET.register_module()
class NDHumanevalDataset(BaseDataset):

    @staticmethod
    def load(db_url: str, size: int, seed: Union[int, str]):

        engine = create_engine(db_url)

        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)

        dataset = []
        with SessionLocal() as db:
            db_samples = crud.get_samples_from_dataset("humaneval", size, db, seed)

            for sample in db_samples:
                dataset.append({
                    'sample_id': sample.id,
                    'problem': sample.components['problem'],
                    'query': sample.components["query"].query,
                    'task_id': sample.target['task_id'],
                })

        dataset = Dataset.from_list(dataset)
        return dataset


@ICL_EVALUATORS.register_module()
class NDHumanEvaluator(BaseEvaluator):
    """Evaluator for HumanEval or EvalPlus."""

    def __init__(self,
                 k: List[int] = [1, 10, 100],
                 dataset: str = 'humaneval') -> None:

        self.metric = "functional_correctness"
        self.dataset = dataset
        assert self.dataset in ['humaneval', 'evalplus']

        if self.dataset == 'humaneval':
            try:
                from human_eval.data import HUMAN_EVAL, write_jsonl
                # from human_eval.evaluation import evaluate_functional_correctness

                self.write_jsonl = write_jsonl
                self.HUMAN_EVAL = HUMAN_EVAL
                self.eval = evaluate_functional_correctness
            except ImportError:
                raise ImportError(
                    'Please install human_eval use following steps:\n'
                    'git clone git@github.com:open-compass/human-eval.git\n'
                    'cd human-eval && pip install -e .')
        else:
            raise NotImplementedError("evalplus is not implemented")
            try:
                from evalplus.data import write_jsonl
                from evalplus.evaluate import evaluate

                self.write_jsonl = write_jsonl
                self.eval = evaluate
            except ImportError:
                raise ImportError(
                    'Please install evalplus use following steps:\n'
                    'git clone --recurse-submodules git@github.com:open-compass/human-eval.git\n'  # noqa
                    'cd human-eval\n'
                    'pip install -e .\n'
                    'pip install -e evalplus\n')
        self.k = k
        super().__init__()

    def score(self, predictions: List, references: List, test_set: List, sample_ids: List) -> dict:
        prompts = [item['problem'] for item in test_set]
        humaneval_preds = []
        if self.dataset == 'humaneval':
            # create json file in human_eval format
            task_to_sample = {}
            for preds, ref, id in zip(predictions, references, sample_ids):
                # suits for two case
                # 1. use repeated dataset
                # 2. use `num_return_sequences` to generate multiple responses
                task_to_sample[ref] = id
                if not isinstance(preds, list):
                    preds = [preds]
                for pred in preds:
                    humaneval_preds.append({
                        'task_id': ref,
                        'completion': pred
                    })
            with tempfile.TemporaryDirectory() as tmp_dir:
                out_dir = osp.join(tmp_dir, 'human_eval.json')
                self.write_jsonl(out_dir, humaneval_preds)
                score = self.eval(out_dir,
                                  self.k,
                                  timeout=3.0,
                                  problem_file=self.HUMAN_EVAL)

                sample_correctness = self._get_sample_correctness(out_dir + "_results.jsonl", task_to_sample)
                result = {f'humaneval_{k}': score[k] * 100 for k in score}
                result['sample_score'] = sample_correctness
                return result
        else:
            for preds, refer, prompt in zip(predictions, references, prompts):
                if not isinstance(preds, list):
                    preds = [preds]
                for pred in preds:
                    humaneval_preds.append({
                        'task_id': refer,
                        'solution': prompt + pred
                    })
            with tempfile.TemporaryDirectory() as tmp_dir:
                out_dir = osp.join(tmp_dir, 'human_eval.jsonl')
                self.write_jsonl(out_dir, humaneval_preds)
                flags = dict(dataset='humaneval',
                             samples=out_dir,
                             base_only=None,
                             parallel=None,
                             i_just_wanna_run=None,
                             test_details=0.2,
                             min_time_limit=0.2,
                             gt_time_limit_factor=4.0,
                             mini=None)
                score = self.eval(flags)
                return {f'humaneval_plus_{k}': score[k] * 100 for k in score}

    def _get_sample_correctness(self, results_file: str, task_to_sample: dict):
        sample_correctness = []
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                assert sample['passed'] in (True, False)

                sample_id = task_to_sample[sample['task_id']]
                sample_result = {
                    "sample_id": sample_id,
                    "score": 1. if sample['passed'] else 0.
                }
                sample_correctness.append(sample_result)
        return sample_correctness
