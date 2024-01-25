import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from opencompass.registry import MODELS
from opencompass.utils import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class Gemini(BaseAPIModel):
    """Model wrapper around Google Gemini API.

    Args:
        key (str): Authorization key.
        path (str): The model to be used. Defaults to gemini-pro.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    def __init__(
        self,
        key: str,
        path: str = 'gemini-pro',
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)
        try:
            import google.generativeai
        except ImportError:
            raise ImportError('Import google.generativeai failed. Please install it '
                              'with "pip install -q -U google-generativeai" and try again.')

        try:
            import litellm
        except ImportError:
            raise ImportError('Import litellm failed. Please install it '
                              'with "pip install litellm" and try again.')

        os.environ["GEMINI_API_KEY"] = key
        self.litellm = litellm
        self.model = path

    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len: int = 512,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str or PromptList]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs)))
        return results

    def _generate(
        self,
        input: str or PromptList,
        max_out_len: int = 512,
    ) -> str:
        """Generate results given an input.

        Args:
            inputs (str or PromptList): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))

        if isinstance(input, str):
            messages = [{"role": "user", "content": f'{input}'}]
        else:
            messages = []
            for item in input:
                if item['role'] == 'HUMAN' or item['role'] == 'SYSTEM':
                    messages.append({"role": "user", "content": item["prompt"]})
                elif item['role'] == 'BOT':
                    messages.append({"role": "assistant", "content": item["prompt"]})
            assert messages[-1]["role"] == "user"

        num_retries = 0
        while num_retries < self.retry:
            self.wait()
            try:
                response = self.litellm.completion(
                    model=f"gemini/{self.model}",
                    messages=messages)
                content = response['choices'][0]['message']['content'].strip()
                return content
            except Exception as e:
                self.logger.error(e)
            num_retries += 1
        raise RuntimeError('Calling Gemini API failed after retrying for '
                           f'{self.retry} times. Check the logs for details.')
#!/usr/bin/env python3
