from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from mistralai.models.chat_completion import ChatMessage

from opencompass.registry import MODELS
from opencompass.utils import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class MistralAI(BaseAPIModel):
    """Model wrapper around Mistral AI API.

    Args:
        key (str): Authorization key.
        path (str): The model to be used. Defaults to claude-2.
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
            path: str,
            query_per_second: int = 2,
            max_seq_len: int = 2048,
            max_out_len: int = 1024,
            meta_template: Optional[Dict] = None,
            retry: int = 2,
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)
        try:
            from mistralai.client import MistralClient
            from mistralai.models.chat_completion import ChatMessage
        except ImportError:
            raise ImportError('Import together failed. Please install it '
                              'with "pip install together" and try again.')

        self.client = MistralClient(api_key=key)
        self.model = path
        self.max_out_len = max_out_len

    def generate(
        self,
        inputs: Union[List[str], List[PromptList]],
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
        input: Union[str, PromptList],
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
            messages = [ChatMessage(role="user", content=input)]
        else:
            messages = []
            for item in input:
                if item['role'] == 'HUMAN' or item['role'] == 'SYSTEM':
                    messages.append(ChatMessage(role="user", content=item["prompt"]))
                elif item['role'] == 'BOT':
                    messages.append(ChatMessage(role="assistant", content=item["prompt"]))

        num_retries = 0
        while num_retries < self.retry:
            self.wait()
            try:
                completion = self.client.chat(
                    messages=messages,
                    model=self.model,
                    max_tokens=self.max_out_len,
                    temperature=0,
                )
                response = completion.choices[0].message.content
                return response
            except Exception as e:
                self.logger.error(e)
                error_msg = str(e)
            num_retries += 1
        self.logger.error(f'Calling Mistral {self.model} API failed after retrying for '
                           f'{self.retry} times. Check the logs for details.')
        return f"### No response ### {error_msg}"
