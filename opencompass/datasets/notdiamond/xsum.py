from typing import Union

from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from datasets import Dataset

from notdiamond_server.database import crud
from notdiamond_server.database.initialize import Base


@LOAD_DATASET.register_module()
class NDXsumDataset(BaseDataset):

    @staticmethod
    def load(db_url: str, size: int, seed: Union[int, str]):

        engine = create_engine(db_url)

        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)

        dataset = []
        with SessionLocal() as db:
            db_samples = crud.get_samples_from_dataset("xsum", size, db, seed)

            for sample in db_samples:
                dataset.append({
                    'sample_id': sample.id,
                    'query': sample.components["query"].query,
                    'context': sample.components["context"].context,
                    'label': sample.target['label'],
                })

        dataset = Dataset.from_list(dataset)
        return dataset
