import copy
import csv
import os.path as osp
import random

from datasets import Dataset

from opencompass.openicl.icl_evaluator import NDAccEvaluator
from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset

from .read_data import get_samples_from_local_dataset


@LOAD_DATASET.register_module()
class NDGPQADataset(BaseDataset):

    @staticmethod
    def load(db_url: str, size: int, seed: int | str):
        random.seed(seed)
        eval_data_path = osp.join(db_url, "gpqa.json")

        samples = get_samples_from_local_dataset(eval_data_path, size, seed)

        dataset = []
        for sample_id, sample in samples.items():
            dataset.append(
                {
                    "sample_id": sample_id,
                    "query": sample["components"]["query"]["query"],
                    "label": sample["target"]["label"],
                }
            )

        dataset = Dataset.from_list(dataset)
        return dataset


class NDGPQAEvaluator(NDAccEvaluator):
    pass
