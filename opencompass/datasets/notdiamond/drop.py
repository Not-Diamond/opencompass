import json
import os.path as osp
import random

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset
from .read_data import get_samples_from_local_dataset


@LOAD_DATASET.register_module()
class NDDropDataset(BaseDataset):

    @staticmethod
    def get_answers(validated_answers):
        answers = []
        for answer_item in validated_answers:
            if answer_item["number"]:
                answers.append(answer_item["number"])
            elif any(answer_item["date"][i] for i in ["day", "month", "year"]):
                d = [answer_item["date"][i] for i in ["day", "month", "year"]]
                answers.append(" ".join(d).strip())
            else:
                for span in answer_item["spans"]:
                    answers.append(span)
        answers = list(set(answers))
        return answers

    @staticmethod
    def load(db_url: str, size: int, seed: int | str):
        random.seed(seed)
        eval_data_path = osp.join(db_url, "drop.json")

        samples = get_samples_from_local_dataset(eval_data_path, size, seed)

        dataset = []
        for sample_id, sample in samples.items():
            dataset.append(
                {
                    "sample_id": sample_id,
                    "context": sample["components"]["context"]["context"],
                    "query": sample["components"]["query"]["query"],
                    "label": sample["target"]["label"],
                }
            )

        dataset = Dataset.from_list(dataset)
        return dataset
