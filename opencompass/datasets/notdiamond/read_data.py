import json
import random
from typing import Union

def get_samples_from_local_dataset(path: str, size: int, seed: Union[int, str]) -> dict:
    random.seed(seed)

    with open(path) as f:
        eval_data = json.load(f)

    n_samples = len(eval_data.keys())
    if n_samples < size:
        size = n_samples
    sample_ids = random.sample(list(eval_data.keys()), size)
    samples = {}
    for sample_id in sample_ids:
        sample = eval_data[sample_id]
        samples[sample_id] = sample
    return samples
