import random
import os.path as osp
from typing import Union

from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset

from datasets import Dataset

from .read_data import get_samples_from_local_dataset


@LOAD_DATASET.register_module()
class NDXsumDataset(BaseDataset):

    @staticmethod
    def load(db_url: str, size: int, seed: Union[int, str]):
        random.seed(seed)
        eval_data_path = osp.join(db_url, "xsum.json")

        samples = get_samples_from_local_dataset(eval_data_path, size, seed)

        dataset = []
        for sample_id, sample in samples.items():
            dataset.append({
                'sample_id': sample_id,
                'context': sample["components"]["context"]["context"],
                'query': sample["components"]["query"]["query"],
                'label': sample["target"]['label'],
            })

        # engine = create_engine(db_url)

        # SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        # Base.metadata.create_all(bind=engine)

        # dataset = []
        # with SessionLocal() as db:
        #     db_samples = crud.get_samples_from_dataset("xsum", size, db, seed)

        #     for sample in db_samples:
        #         dataset.append({
        #             'sample_id': sample.id,
        #             'query': sample.components["query"].query,
        #             'context': sample.components["context"].context,
        #             'label': sample.target['label'],
        #         })

        dataset = Dataset.from_list(dataset)
        return dataset
