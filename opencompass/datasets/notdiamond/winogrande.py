from typing import Union

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from notdiamond_server.database import crud
from notdiamond_server.database.initialize import Base

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class NDwinograndeDataset(BaseDataset):

    @staticmethod
    def load(db_url: str, size: int, seed: Union[int, str]):
        engine = create_engine(db_url)

        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)

        dataset = []
        with SessionLocal() as db:
            db_samples = crud.get_samples_from_dataset("winogrande", size, db, seed)

            for sample in db_samples:
                dataset.append({
                    'sample_id': sample.id,
                    'query': sample.components["query"].query,
                    'label': sample.target['label'],
                })

        dataset = Dataset.from_list(dataset)
        return dataset
