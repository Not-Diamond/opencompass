import random
import os.path as osp
from typing import Union

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset

from .read_data import get_samples_from_local_dataset


@LOAD_DATASET.register_module()
class NDFlores100Dataset(BaseDataset):

    @staticmethod
    def load(subset: str, db_url: str, size: int, seed: Union[int, str]):
        random.seed(seed)
        eval_data_path = osp.join(db_url, f"flores100.{subset}.json")

        samples = get_samples_from_local_dataset(eval_data_path, size, seed)

        dataset = []
        for sample_id, sample in samples.items():
            dataset.append(
                {
                    "sample_id": sample_id,
                    "prompt": sample["components"]["prompt"]["prompt"],
                    "query": sample["components"]["query"]["query"],
                    "label": sample["target"]["label"],
                }
            )

        dataset = Dataset.from_list(dataset)
        return dataset