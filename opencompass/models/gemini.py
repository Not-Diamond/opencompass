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
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel(f"models/{path}")
            self.genai = genai
        except ImportError:
            raise ImportError('Import google.generativeai failed. Please install it '
                              'with "pip install -q -U google-generativeai" and try again.')

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        }

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
            messages = [{"role": "user", "parts": [f'{input}']}]
        else:
            messages = []
            for item in input:
                if item['role'] == 'HUMAN' or item['role'] == 'SYSTEM':
                    messages.append({"role": "user", "parts": [item["prompt"]]})
                elif item['role'] == 'BOT':
                    messages.append({"role": "model", "parts": [item["prompt"]]})
            assert messages[-1]["role"] == "user"

        num_retries = 0
        while num_retries < self.retry:
            self.wait()
            try:
                response = self.model.generate_content(messages,
                                                       generation_config=self.genai.types.GenerationConfig(temperature=0.),
                                                       safety_settings=self.safety_settings)
                content_parts = response.candidates[0].content.parts
                content = ""
                for part in content_parts:
                    content += part.text
                    content += "\n"
                return content
            except Exception as e:
                self.logger.error(e)
                error_msg = str(e)
            num_retries += 1
        self.logger.error('Calling Gemini API failed after retrying for '
                           f'{self.retry} times. Check the logs for details.')
        return f"### No response ### {error_msg}"
        # raise RuntimeError('Calling Gemini API failed after retrying for '
        #                    f'{self.retry} times. Check the logs for details.')
