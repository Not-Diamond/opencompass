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
        logger = get_logger(log_level="DEBUG")
        model_to_denylist = {}
        for dirpath, dirnames, filenames in os.walk(training_path, topdown=False):
            if len(dirnames) > 0:
                # We've exhausted all dirs, we're done
                break

            for filename in filenames:
                # Skip files for other training datasets
                if dataset_abbr not in filename:
                    continue

                _, model_name = os.path.split(dirpath)

                with open(f"{dirpath}/{filename}", "r") as f:
                    sample_data = json.load(f)
                    if "denylist" in sample_data:
                        model_to_denylist[model_name] = sample_data["denylist"]

        logger.info(
            f"Identified denylist samples for {dataset_abbr} datasets: {model_to_denylist}"
        )
        return model_to_denylist
