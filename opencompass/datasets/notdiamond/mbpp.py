import json
import random

import io
import re
import signal
import tempfile
import itertools
import contextlib
import os.path as osp

from collections import defaultdict
from typing import List, Sequence, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker

from notdiamond_server.database import crud
# from notdiamond_server.database.initialize import Base

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class NDMBPPDataset(BaseDataset):

    @staticmethod
    def load(db_url: str, size: int, seed: Union[int, str]):
        random.seed(seed)
        eval_data_path = osp.join(db_url, "mbpp.json")

        samples = crud.get_samples_from_local_dataset(eval_data_path, size, seed)

        dataset = []
        for sample_id, sample in samples.items():
            dataset.append({
                'sample_id': sample_id,
                'prompt': sample["components"]["prompt"]["prompt"],
                'query': sample["components"]["query"]["query"],
                'tests': sample["target"]['tests'],
            })

        # engine = create_engine(db_url)

        # SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        # Base.metadata.create_all(bind=engine)

        # dataset = []
        # with SessionLocal() as db:
        #     db_samples = crud.get_samples_from_dataset("mbpp", size, db, seed)

        #     for sample in db_samples:
        #         dataset.append({
        #             'sample_id': sample.id,
        #             'prompt': sample.components["prompt"].prompt,
        #             'query': sample.components["query"].query,
        #             'tests': sample.target['tests'],
        #         })

        dataset = Dataset.from_list(dataset)
        return dataset


class TimeOutException(Exception):
    pass


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def time_limit(seconds: float):

    def signal_handler(signum, frame):
        raise TimeOutException('Time out!')

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from."""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


@ICL_EVALUATORS.register_module()
class NDMBPPEvaluator(BaseEvaluator):
    """Evaluator for MBPP or MBPPPlus."""

    def __init__(self, dataset: str = 'mbpp') -> None:
        self.metric = "functional_correctness"

        self.dataset = dataset
        assert self.dataset in ['mbpp', 'mbppplus']

    def score(self, predictions: List, references: List, sample_ids: List) -> dict:
        assert len(predictions) == len(references)

        if self.dataset == 'mbpp':
            result = {'pass': 0, 'timeout': 0, 'failed': 0, 'wrong_answer': 0}
            details = {}
            sample_correctness = []
            # change to thread pool for better killing blocked instance
            for i, (refer, pred, id) in enumerate(zip(references, predictions, sample_ids)):
                pred = self._process_answer(pred)
                programs = self._process_test(refer, pred)
                index, key = execution(programs, i, 3)
                result[key] += 1
                details[str(index)] = {
                    'prompt': predictions[index],
                    'origin_prediction': predictions[index],
                    'result': key
                }
                sample_result = {
                    "sample_id": id,
                    "score": 1. if key == "pass" else 0.
                }
                sample_correctness.append(sample_result)

            result['score'] = result['pass'] / len(predictions) * 100
            result['details'] = details
            result['sample_score'] = sample_correctness
            return result
        else:
            raise NotImplementedError("MBPPPlus is not implemented")
            try:
                from evalplus.data import write_jsonl
                from evalplus.evaluate import evaluate
                self.write_jsonl = write_jsonl
                self.eval = evaluate
            except ImportError:
                raise ImportError(
                    'Please install evalplus use following steps:\n'
                    'git clone --recurse-submodules git@github.com:open-compass/human-eval.git\n'  # noqa
                    'cd human-eval\n'
                    'pip install -e .\n'
                    'pip install -e evalplus\n')
            mbpp_preds = []
            for preds, refer in zip(predictions, references):
                if not isinstance(preds, list):
                    preds = [preds]
                for pred in preds:
                    mbpp_preds.append({'task_id': refer, 'solution': pred})
            with tempfile.TemporaryDirectory() as tmp_dir:
                out_dir = osp.join(tmp_dir, 'mbpp_eval.jsonl')
                self.write_jsonl(out_dir, mbpp_preds)
                flags = dict(dataset='mbpp',
                             samples=out_dir,
                             base_only=None,
                             parallel=None,
                             i_just_wanna_run=None,
                             test_details=0.2,
                             min_time_limit=0.2,
                             gt_time_limit_factor=4.0,
                             mini=None)
                score = self.eval(flags)
                return {f'mbpp_plus_{k}': score[k] * 100 for k in score}

    def _process_answer(self, text):
        try:
            # for chatGLM related text
            text = eval(text)
        except Exception:
            pass
        # deal with code block
        if '```' in text:
            blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
            if len(blocks) == 0:
                text = text.split('```')[1]  # fall back to default strategy
            else:
                text = blocks[0]  # fetch the first code block
                if not text.startswith('\n'):  # in case starting with ```xxx
                    text = text[max(text.find('\n') + 1, 0):]
        text = text.strip()
        match = re.search(r"('\s*|)(\[DONE\]|DONE)", text)
        if match:
            text = text[:match.start()]
        match = re.search(r"(\[BEGIN\]|BEGIN)('\s*|)", text)
        if match:
            text = text[match.end():]
        text = text.strip()
        if text.startswith("'"):
            text = text[1:]
        if text.endswith("'"):
            text = text[:-1]
        text = text.replace('\\', '')
        match = re.search(r'```python(.*)```', text, re.DOTALL)
        if match:
            text = match.group(1).strip().split('```')[0].strip()
        return text

    def _process_test(self, test_case, pred):
        formatted = pred + '\n'
        formatted += test_case
        return formatted


def execution(programs, task_id, timeout):
    """Execution function for running generation code.

    Args:
        programs(str): Python code to be executed.
        task_id(int): Task id of the current example.
        timeout(int): Time limit for execution, avoid unnecessary
            blocking.

    In pass@k scenario, a lot of programs should be executed.
    Some internal error cannot be handled properly, such as
    `RecursionError` might cause system break. It is better to
    separate the execution in thread or multiprocess to better
    control the process.
    """

    def _execution(programs, timeout):
        try:
            # Add exec globals to prevent the exec to raise
            # unnecessary NameError for correct answer
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(programs, exec_globals)
            key.append('pass')
        except TimeOutException:
            key.append('timeout')
        except AssertionError:
            key.append('wrong_answer')
        except BaseException as e:
            print(e)
            key.append('failed')

    key = []
    # `signal` cannot be used in child thread, therefore, we
    # need to create a process in the thread.
    _execution(programs, timeout - 1)

    # p = multiprocessing.Process(target=_execution,
    #                             args=(programs, timeout - 1))
    # p.start()
    # p.join(timeout=timeout)
    # if p.is_alive():
    #     p.kill()
    #     # key might not have value if killed
    #     return task_id, 'timeout'
    return task_id, key[0]
