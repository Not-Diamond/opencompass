import os.path as osp
import random
from typing import Dict, List, Optional

from mmengine.config import Config, ConfigDict

from opencompass.registry import PARTITIONERS
from opencompass.utils import get_infer_output_path, get_logger

from .base import BasePartitioner


@PARTITIONERS.register_module()
class NDNaivePartitioner(BasePartitioner):
    """Naive task partitioner. This partitioner will generate a task for each n
    model-dataset pairs.

    Args:
        out_dir (str): The output directory of tasks.
        n (int): The number of model-dataset pairs in each task.
        keep_keys (List[str]): The keys to be kept from the experiment config
            to the task config.
    """

    def __init__(
        self, out_dir: str, n: int = 1, keep_keys: Optional[List[str]] = None, **kwargs
    ):
        super().__init__(out_dir=out_dir, keep_keys=keep_keys)
        self.n = n
        get_logger().warn(f"Ignoring kwargs passed to NDNaivePartitioner: {kwargs}")

    def partition(
        self,
        model_dataset_combinations: List[Dict[str, List[ConfigDict]]],
        work_dir: str,
        out_dir: str,
        add_cfg: Dict = {},
    ) -> List[Dict]:
        """Partition model-dataset pairs into tasks. Each task is defined as a
        dict and will run independently as a unit. Its structure is as
        follows:

        .. code-block:: python

            {
                'models': [],  # a list of model configs
                'datasets': [[]],  # a nested list of dataset configs, each
                                    list corresponds to a model
                'work_dir': '',  # the work dir
            }

        Unlike the base NaivePartitioner, this partitioner will shuffle the combination of models
        and datsets before creating each one.

        Args:
            model_dataset_combinations (List[Dict]): List of
                `{models: [...], datasets: [...]}` dicts. Each dict contains
                a list of model configs and a list of dataset configs.
            work_dir (str): The work dir for the task.
            out_dir (str): The full output path for the task, intended for
                Partitioners to check whether the task is finished via the
                existency of result file in this directory.

        Returns:
            List[Dict]: A list of tasks.
        """

        tasks = []
        for comb in model_dataset_combinations:
            for model in comb["models"]:
                chunks = []
                for dataset in comb["datasets"]:
                    filename = get_infer_output_path(model, dataset, out_dir)
                    if osp.exists(filename):
                        continue
                    chunks.append(dataset)

                for i in range(0, len(chunks), self.n):
                    task = Config(
                        {
                            "models": [model],
                            "datasets": [chunks[i : i + self.n]],
                            "work_dir": work_dir,
                            **add_cfg,
                        }
                    )
                    tasks.append(task)
        random.shuffle(tasks)
        return tasks
