from abc import abstractstaticmethod
import json
import os
from typing import Dict, Optional, Union, List

from datasets import Dataset, DatasetDict

from opencompass.openicl import DatasetReader
from opencompass.utils import get_logger


class BaseDataset:

    def __init__(self, reader_cfg: Optional[Dict] = {}, **kwargs):
        self.dataset = self.load(**kwargs)
        self._init_reader(**reader_cfg)

    def _init_reader(self, **kwargs):
        self.reader = DatasetReader(self.dataset, **kwargs)

    @property
    def train(self):
        return self.reader.dataset["train"]

    @property
    def test(self):
        return self.reader.dataset["test"]

    @abstractstaticmethod
    def load(**kwargs) -> Union[Dataset, DatasetDict]:
        pass

    @staticmethod
    def build_denylist(training_path: str, dataset_abbr: str) -> Dict[str, List[str]]:
        """
        Take a file path to training samples for OOB, load them and identify previously-prompted samples.
        Do not re-prompt these samples.
        """
        logger = get_logger(log_level="INFO")
        model_to_denylist = {}
        for entry in os.scandir(training_path):
            for model in os.scandir(entry.path):

                if not model.is_dir():
                    logger.debug(f"Skipping non-directory model {model.path}")
                    continue

                for eval_dataset in os.scandir(model.path):
                    if dataset_abbr not in eval_dataset.name:
                        logger.debug(
                            f"Skipping dataset file {eval_dataset.path} - not a {dataset_abbr} file."
                        )
                        continue

                    with open(eval_dataset.path) as f:
                        sample_data = json.load(f)
                        if "denylist" in sample_data:
                            model_to_denylist[model.name] = sample_data["denylist"]

        logger.info(
            f"Identified denylist samples for {dataset_abbr} datasets: {model_to_denylist}"
        )
        return model_to_denylist
