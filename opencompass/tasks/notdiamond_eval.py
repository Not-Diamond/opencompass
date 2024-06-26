import os
import json
import argparse
import copy
import fnmatch
import math
import os.path as osp
import statistics
import time
from collections import Counter, defaultdict
from inspect import signature
from shutil import which
from typing import List, Optional
from datetime import datetime

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist
import sentry_sdk

from opencompass.registry import (ICL_EVALUATORS, MODELS, TASKS,
                                  TEXT_POSTPROCESSORS)
from opencompass.tasks.base import BaseTask
from opencompass.utils import (build_dataset_from_cfg, dataset_abbr_from_cfg,
                               get_infer_output_path, get_logger,
                               task_abbr_from_cfg)

# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker

import wandb

from opencompass.config import WANDB_API_KEY, EVALUATIONS_WORK_DIR, VALID_DATASET_ABBR, WITH_SUBSETS



def extract_role_pred(s: str, begin_str: Optional[str],
                      end_str: Optional[str]) -> str:
    """Extract the role prediction from the full prediction string. The role
    prediction may be the substring between the begin and end string.

    Args:
        s (str): Full prediction string.
        begin_str (str): The beginning string of the role
        end_str (str): The ending string of the role.

    Returns:
        str: The extracted role prediction.
    """
    start = 0
    end = len(s)

    if begin_str:
        begin_idx = s.find(begin_str)
        if begin_idx != -1:
            start = begin_idx + len(begin_str)

    if end_str:
        # TODO: Support calling tokenizer for the accurate eos token
        # and avoid such hardcode
        end_idx = s.find(end_str, start)
        if end_idx != -1:
            end = end_idx

    return s[start:end]


@TASKS.register_module(force=(__name__ == '__main__'))  # A hack for script run
class NDICLEvalTask(BaseTask):
    """NDICL Evaluation Task.

    This task is used to evaluate the metric between predictions and
    references.
    """

    name_prefix = 'NDICLEval'
    log_subdir = 'logs/eval'
    output_subdir = 'results'

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        self.num_gpus = 0
        self.logger = get_logger()
        self.dump_details = cfg.get('eval', {}).get('runner', {}).get(
            'task', {}).get('dump_details', False)

    def get_command(self, cfg_path, template):
        script_path = __file__
        python = 'python3' if which('python3') else 'python'
        command = f'{python} {script_path} {cfg_path}'
        return template.format(task_cmd=command)

    def run(self):
        sentry_sdk.init(
            dsn="https://6f02eb3c39ab2ba61a8e77c6e3590062@o4506820913201152.ingest.us.sentry.io/4507376224370688",
            traces_sample_rate=1.0,
            profiles_sample_rate=1.0,
        )
        for model_cfg, dataset_cfgs in zip(self.model_cfgs, self.dataset_cfgs):
            for dataset_cfg in dataset_cfgs:
                self.model_cfg = model_cfg
                self.dataset_cfg = dataset_cfg

                # Load Dataset
                self.eval_cfg = self.dataset_cfg.get('eval_cfg')
                self.output_column = dataset_cfg['reader_cfg']['output_column']

                # overwrite postprocessor if the model has specified one
                ds_abbr = dataset_abbr_from_cfg(self.dataset_cfg)
                model_postprocessors = self.model_cfg.get(
                    'pred_postprocessor', {})
                for pattern in model_postprocessors.keys():
                    if fnmatch.fnmatch(ds_abbr, pattern):
                        self.eval_cfg['pred_postprocessor'] = model_postprocessors[
                                pattern]  # noqa
                        break

                out_path = get_infer_output_path(
                    self.model_cfg, self.dataset_cfg,
                    osp.join(self.work_dir, 'results'))
                if osp.exists(out_path):
                    continue
                self._score()

    def _score(self):
        test_set = build_dataset_from_cfg(self.dataset_cfg).test
        # Postprocess dataset if necessary
        if 'dataset_postprocessor' in self.eval_cfg:
            proc = self.eval_cfg['dataset_postprocessor']['type']
            if isinstance(proc, str):
                proc = TEXT_POSTPROCESSORS.get(proc)

            def postprocess(sample):
                s = sample[self.output_column]
                sample[self.output_column] = proc(s)
                return sample

            test_set = test_set.map(postprocess)

        # Load predictions
        filename = get_infer_output_path(
            self.model_cfg, self.dataset_cfg,
            osp.join(self.work_dir, 'predictions'))
        # in case the prediction is partial
        root, ext = osp.splitext(filename)
        partial_filename = root + '_0' + ext

        # Get sc_size if use Self-Consistency
        sc_size = self.eval_cfg.get('sc_size')

        if not osp.exists(osp.realpath(filename)) and not osp.exists(
                osp.realpath(partial_filename)):
            result = {'error': 'No predictions found.'}
        else:
            if osp.exists(osp.realpath(filename)):
                preds = mmengine.load(filename)
                preds = [preds[str(i)] for i in range(len(preds))]
            else:
                filename = partial_filename
                preds = []
                i = 1
                while osp.exists(osp.realpath(filename)):
                    sub_preds = mmengine.load(filename)
                    preds.extend(
                        [sub_preds[str(i)] for i in range(len(sub_preds))])
                    filename = root + f'_{i}' + ext
                    i += 1
            pred_dicts = copy.deepcopy(preds)
            preds = {k: [pred.get(k) for pred in preds] for k in preds[0]}

            pred_strs = preds.pop('prediction', None)
            pred_list_flag = pred_strs is not None and isinstance(
                pred_strs[0], list)
            if ('pred_role' in self.eval_cfg
                    and 'meta_template' in self.model_cfg
                    and not MODELS.get(self.model_cfg['type']).is_api):
                # Create a prompt template for role config parsing
                from opencompass.models.base import LMTemplateParser
                parser = LMTemplateParser(self.model_cfg['meta_template'])
                role = parser.roles[self.eval_cfg['pred_role']]
                if sc_size is not None:
                    assert pred_list_flag, (
                        'The prediction for Self-Consistency'
                        'must be list.')
                if pred_list_flag:
                    pred_strs = [[
                        extract_role_pred(_pred, role.get('begin', None),
                                          role.get('end', None))
                        for _pred in pred
                    ] for pred in pred_strs]
                else:
                    pred_strs = [
                        extract_role_pred(pred, role.get('begin', None),
                                          role.get('end', None))
                        for pred in pred_strs
                    ]

            # Postprocess predictions if necessary
            if 'pred_postprocessor' in self.eval_cfg:
                kwargs = self.eval_cfg['pred_postprocessor']
                proc = kwargs.pop('type')
                if isinstance(proc, str):
                    proc = TEXT_POSTPROCESSORS.get(proc)
                if pred_list_flag:
                    pred_strs = [[proc(s, **kwargs) for s in preds]
                                 for preds in pred_strs]
                else:
                    pred_strs = [proc(s, **kwargs) for s in pred_strs]

            # Get majority voting predictions if use self-consistency
            if sc_size is not None:
                pred_strs = [
                    Counter(s).most_common(1)[0][0] for s in pred_strs
                ]

            icl_evaluator = ICL_EVALUATORS.build(self.eval_cfg['evaluator'])
            # need results dir to save other files
            out_path = get_infer_output_path(
                self.model_cfg, self.dataset_cfg,
                osp.join(self.work_dir, 'results'))
            icl_evaluator._out_dir = osp.splitext(out_path)[0]  # strip extension

            preds['predictions'] = pred_strs
            preds['references'] = (test_set[self.output_column]
                                   if self.output_column else None)
            preds['origin_prompt'] = [p['origin_prompt'] for p in pred_dicts]
            preds['test_set'] = test_set
            preds['sample_ids'] = test_set["sample_id"]
            preds = {
                k: preds[k]
                for k in signature(icl_evaluator.score).parameters
            }
            result = icl_evaluator.score(**preds)

            if self.dump_details:
                details = result.get('details', None)
                try:
                    result['details'] = self.format_details(
                        pred_strs, test_set[self.output_column], details,
                        pred_dicts)
                    result['type'] = result['details'].pop('type', None)

                    if 'PPL' in str(
                            self.dataset_cfg.infer_cfg.inferencer.type):
                        result['correct_bpb'], result['incorrect_bpb'] = \
                            self.calculate_bpb(pred_dicts)
                except Exception as e:
                    self.logger.warning(f'Skip dumping details due to: {e}.')
            else:
                result.pop('details', None)

        if 'error' in result:
            self.logger.error(
                f'Task {task_abbr_from_cfg(self.cfg)}: {result["error"]}')
            return
        else:
            result_wo_details = {
                i: result[i]
                for i in result if i != 'details'
            }
            self.logger.info(
                f'Task {task_abbr_from_cfg(self.cfg)}: {result_wo_details}')

        # Save result
        out_path = get_infer_output_path(self.model_cfg, self.dataset_cfg,
                                         osp.join(self.work_dir, 'results'))
        mkdir_or_exist(osp.split(out_path)[0])
        mmengine.dump(result, out_path, ensure_ascii=False, indent=4)

        # self._save_results_to_db(result, icl_evaluator.metric)
        self._log_eval_success_to_wandb(result, icl_evaluator.metric)
        self._dump_training_data(result, icl_evaluator.metric)

    def _dump_training_data(self, result: dict, metric: str):
        assert "db_url" in self.dataset_cfg
        assert "abbr" in self.model_cfg

        db_url = self.dataset_cfg["db_url"]
        source = self.dataset_cfg["abbr"]
        if source[-1].isdigit() and (source[:-2] in VALID_DATASET_ABBR or any([exc in source[:-2] for exc in WITH_SUBSETS])):
            dataset_name = source[:-2]
        else:
            dataset_name = source

        details = result["details"]
        out_path = get_infer_output_path(self.model_cfg, self.dataset_cfg,
                                         osp.join(self.work_dir, 'training_data'))
        eval_data_path = osp.join(db_url, f"{dataset_name}.json")
        with open(eval_data_path) as f:
            eval_data = json.load(f)

        training_data = {}
        sample_score = result["sample_score"]
        assert isinstance(sample_score, list) or isinstance(sample_score, dict)
        if isinstance(sample_score, list):
            for i, sample_result in enumerate(sample_score):
                sample_id = sample_result["sample_id"]
                assert sample_id not in training_data

                eval_sample = eval_data[sample_id]
                eval_details = details[f"{i}"]
                sample_result["metric"] = metric
                training_data[sample_id] = {
                    "sample_details": eval_sample,
                    "eval_details": eval_details,
                    "result": sample_result
                }

        elif isinstance(sample_score, dict):
            for sub, sub_score in sample_score.items():
                for i, sample_result in enumerate(sub_score):
                    sample_id = sample_result["sample_id"]
                    if sample_id in training_data:
                        eval_sample = eval_data[sample_id]
                        eval_details = details[f"{i}"]
                        sample_result["metric"] = f"{metric}.{sub}"
                        training_data[sample_id]["result"][f"{sub}"] = sample_result
                    else:
                        eval_sample = eval_data[sample_id]
                        eval_details = details[f"{i}"]
                        sample_result["metric"] = f"{metric}.{sub}"
                        training_data[sample_id] = {
                            "sample_details": eval_sample,
                            "eval_details": eval_details,
                            "result": {f"{sub}": sample_result}
                        }
        mkdir_or_exist(osp.split(out_path)[0])
        mmengine.dump(training_data, out_path, ensure_ascii=False, indent=4)

    def _log_eval_success_to_wandb(self, result: dict, metric: str):
        model = self.model_cfg["abbr"]
        dataset = self.dataset_cfg['abbr']
        timestamp = osp.basename(self.work_dir)
        columns = ["sample_id", "dataset", "metric", "success"]
        failure_found = False

        sample_score = result["sample_score"]
        details = result["details"]
        assert isinstance(sample_score, list) or isinstance(sample_score, dict)

        failures = defaultdict(dict)
        if isinstance(sample_score, list):
            for i, sample_result in enumerate(sample_score):
                sample_id = sample_result['sample_id']
                origin_prediction = details[f"{i}"]["origin_prediction"]
                if "### No response ###" in origin_prediction:
                    if not failure_found:
                        failure_found = True
                        eval_failure_table = wandb.Table(columns=columns)

                    success = False

                    eval_failure_table.add_data(sample_result['sample_id'],
                                                dataset,
                                                metric,
                                                success)
                    failures[sample_id] = self.get_sample_from_dataset_file(dataset, sample_id)
        elif isinstance(sample_score, dict):
            for sub, sub_score in sample_score.items():
                for i, sample_result in enumerate(sub_score):
                    sample_id = sample_result['sample_id']
                    origin_prediction = details[f"{i}"]["origin_prediction"]
                    if "### No response ###" in origin_prediction:
                        if not failure_found:
                            failure_found = True
                            eval_failure_table = wandb.Table(columns=columns)

                        success = False

                        eval_failure_table.add_data(sample_result['sample_id'],
                                                    dataset,
                                                    f"{metric}.{sub}",
                                                    success)
                        failures[sample_id] = self.get_sample_from_dataset_file(dataset, sample_id)
        if failure_found:
            with wandb.init(project=f"LLM Eval", name=f"{model}_{timestamp}", dir=self.work_dir) as run:
                run.log({"eval_fail": eval_failure_table})

            out_path = get_infer_output_path(self.model_cfg, self.dataset_cfg,
                                            osp.join(self.work_dir, 'failures'))
            mmengine.dump(failures, out_path, ensure_ascii=False, indent=4)
            exc = RuntimeError(f"Failed to evaluate {len(failures)} samples for {model} on {dataset}. See {out_path} for details, and try to re-run those samples using `db_url={out_path}.")
            sentry_sdk.capture_exception(exc)

    def get_sample_from_dataset_file(self, dataset_abbr: str, sample_id: str) -> dict:
        dataset_file = get_infer_output_path(self.model_cfg, self.dataset_cfg, osp.join(EVALUATIONS_WORK_DIR, f"{dataset_abbr}.json"))
        with open(dataset_file) as f:
            dataset = json.load(f)
        if sample_id not in dataset:
            raise AssertionError(f"Could not find {sample_id} in {dataset_file}")
        return dataset[sample_id]


    # def _save_results_to_db(self, result: dict, metric: str):
    #     raise RuntimeError("This method is deprecated.")
    #     assert "db_url" in self.dataset_cfg
    #     assert "abbr" in self.model_cfg

    #     model = self.model_cfg["abbr"]
    #     sample_score = result["sample_score"]
    #     assert isinstance(sample_score, list) or isinstance(sample_score, dict)
    #     # Check if sample_score is a dict and expect each key in dict to be the sub-metric name if so

    #     timestamp = datetime.now()

    #     engine = create_engine(self.dataset_cfg["db_url"])

    #     SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    #     Base.metadata.create_all(bind=engine)

    #     with SessionLocal() as db:
    #         if isinstance(sample_score, list):
    #             for sample_result in sample_score:
    #                 eval_result = schemas.EvaluationResultCreate(
    #                     sample_id=sample_result['sample_id'],
    #                     source=self.dataset_cfg['abbr'],
    #                     timestamp=timestamp,
    #                     model=model,
    #                     metric=metric,
    #                     score=sample_result['score']
    #                 )
    #                 crud.add_evaluation_result(eval_result, db)
    #         elif isinstance(sample_score, dict):
    #             for sub, sub_score in sample_score.items():
    #                 for sample_result in sub_score:
    #                     eval_result = schemas.EvaluationResultCreate(
    #                         sample_id=sample_result['sample_id'],
    #                         source=self.dataset_cfg['abbr'],
    #                         timestamp=timestamp,
    #                         model=model,
    #                         metric=f"{metric}.{sub}",
    #                         score=sample_result['score']
    #                     )
    #                     crud.add_evaluation_result(eval_result, db)

    def format_details(self, predictions, references, details, pred_dicts):
        """This function is responsible for formatting prediction details.

        Args:
            predictions (list): The prediction list.
            references (list): The reference list.
            details (list): Contains the 'pred' 'answer' and 'correct' for each
                sample. Such as `[{'pred': '光荣和ωforce',
                'answers': ['光荣和ω-force', '光荣和ωforce'], 'correct': True}]`
            pred_dicts (list): Contains a list of samples with the original
                prompts. Such as
                `[{'origin_prompt': '根据文章回答问题。你的答案应该尽可能3》…………',
                'prediction': ' 光荣和ω-force\n', 'gold': ['光荣和ω-force']}]`

        Returns:
            list: The formatted prediction details.
        """
        results = {}
        for i in range(len(predictions)):
            ppl_flag = False
            result = {}
            origin_prediction = copy.deepcopy(pred_dicts[i])
            origin_prediction.pop('in-context examples', None)
            origin_prediction.pop('prediction', None)
            keys = copy.deepcopy(list(origin_prediction.keys()))
            for key in keys:
                if key.startswith('label:'):
                    ppl_flag = True
                    origin_prediction[key].pop('testing input', None)
                    new_key = key.replace('label: ', '')
                    origin_prediction[new_key] = origin_prediction.pop(key)
            if ppl_flag:
                results['type'] = 'PPL'
                result['origin_prediction'] = origin_prediction
                result['predictions'] = str(predictions[i])
                result['references'] = str(references[i])
                result['correct'] = str(predictions[i]) == str(references[i])
            elif details is not None:
                results['type'] = 'GEN'
                result['prompt'] = origin_prediction['origin_prompt']
                result['origin_prediction'] = pred_dicts[i]['prediction']
                result['predictions'] = details[i]['pred']
                result['references'] = details[i]['answer']
                result['correct'] = details[i]['correct']
            else:
                results['type'] = 'GEN'
                result['prompt'] = origin_prediction['origin_prompt']
                result['origin_prediction'] = pred_dicts[i]['prediction']
                result['predictions'] = str(predictions[i])
                result['references'] = str(references[i])
            results[str(i)] = result
        return results

    def calculate_bpb(self, pred_dicts: List):
        """This function is used to calculate the BPB (Bits Per Byte) for the
        data. The correct BPB is obtained directly from the values in the
        'predictions' file. The incorrect BPB is the average of the remaining
        BPB values for each sample under different labels after subtracting the
        correct BPB. The calculation of BPB (Bits Per Byte) is similar to PPL,
        with the difference that it computes the additional bits needed on
        average, in terms of character length, to encode the true sequence
        based on the predictions. This calculation involves applying a
        weighting factor based on the ratio of words to characters.

        Args:
            pred_dicts (list): Contains a list of samples with each options
                and BPB scores.

        Returns:
            dict: Contains correct and incorrect bpb.
        """
        incorrect_bpb_list = []
        bpb_list = []
        for pred_dict in pred_dicts:
            preds = {
                key: value
                for key, value in pred_dict.items()
                if key.startswith('label: ')
            }
            values = []
            for item in preds.items():
                values.append(item[1])
            bpbs = [value['BPB'] for value in values]
            incorrect_bpb_list.append(
                (sum(bpbs) - min(bpbs)) / (len(bpbs) - 1))
            bpb_list.append(min(bpbs))

        def filters(origins):
            targets = [target for target in origins if not math.isnan(target)]
            return targets

        mean_incorrect = statistics.mean(filters(incorrect_bpb_list))
        mean_correct = statistics.mean(filters(bpb_list))
        return 100 * mean_correct, 100 * mean_incorrect


def parse_args():
    parser = argparse.ArgumentParser(description='Score Calculator')
    parser.add_argument('config', help='Config file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import os
    os.environ['WANDB_API_KEY'] = WANDB_API_KEY
    args = parse_args()
    cfg = Config.fromfile(args.config)
    start_time = time.time()
    inferencer = NDICLEvalTask(cfg)
    inferencer.run()
    end_time = time.time()
    get_logger().info(f'time elapsed: {end_time - start_time:.2f}s')
