from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import tiktoken
from dotenv import load_dotenv
from notdiamond.llms.llm import NDLLM
from notdiamond.llms.provider import NDLLMProvider
from notdiamond.prompts.prompt import NDPromptTemplate

load_dotenv()

import hashlib
import shelve

from .base_api import BaseAPIModel


class NotDiamondModelSelect(BaseAPIModel):
    """
    Initial Reference: models.gemini_api.Gemini
    """

    is_api: bool = True

    def __init__(
        self,
        path: str,
        max_seq_len: int = 2048,
        query_per_second: int = 1,
        retry: int = 2,
        **kwargs,
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            retry=retry,
        )

        llm_providers = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "openai/gpt-4-1106-preview",
            "openai/gpt-4-turbo-preview",
            "anthropic/claude-2.1",
            "anthropic/claude-3-sonnet-20240229",
            "anthropic/claude-3-opus-20240229",
            "google/gemini-pro",
            "mistral/mistral-small-latest",
            "mistral/mistral-medium-latest",
            "mistral/mistral-large-latest",
            "mistral/open-mistral-7b",
            "mistral/open-mixtral-8x7b",
            "togetherai/Mistral-7B-Instruct-v0.2",
            "togetherai/Mixtral-8x7B-Instruct-v0.1",
            "togetherai/Phind-CodeLlama-34B-v2",
            "cohere/command",
        ]
        self.nd_llm = NDLLM(llm_providers=llm_providers)

    def _generate(self, prompt: str, max_out_len: int, temperature: float) -> str:
        """Generate results given an input."""

        try:
            prompt_template = NDPromptTemplate(prompt)
        except Exception as e:
            self.logger.error(e)

            # Fallback Model
            return "openai/gpt-4"

        # Call the ND API and check for a valid response
        num_retries = 0
        while num_retries < self.retry:
            self.wait()
            try:
                # After fuzzy hashing the inputs, the best LLM is determined by the ND API and the LLM is called client-side
                session_id, provider = self.nd_llm.model_select(
                    prompt_template=prompt_template
                )

                self.logger.debug(f"Session ID: {session_id}")
                self.logger.debug(f"Provider: {provider.model}")

                return provider.model

            except Exception as e:
                self.logger.error(e)
                error_msg = str(e)
            num_retries += 1

        self.logger.error(
            "Calling NotDiamond Model-Select API failed after retrying for "
            f"{self.retry} times. Check the logs for details."
        )

        # Fallback Model
        return "openai/gpt-4"

    def generate(
        self,
        inputs: List[str],
        max_out_len: int = 512,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate results given a list of inputs."""

        # with ThreadPoolExecutor() as executor:
        #     results = list(
        #         executor.map(
        #             self._generate,
        #             inputs,
        #             [max_out_len] * len(inputs),
        #             [temperature] * len(inputs),
        #         )
        #     )

        # For loop instead of ThreadPoolExecutor
        results = []
        for inp in inputs:
            results.append(self._generate(inp, max_out_len, temperature))

        self.flush()
        return results

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string."""
        # TODO: Done as a hack, what is the correct way to get the token length with the ND API?
        # Track last model used, and get the token length from the model's tokenizer

        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        # Tokenizing the text
        tokens = encoding.encode(prompt)

        # Counting the tokens
        return len(tokens)

        # prompt_template = NDPromptTemplate(prompt)
        # raise NotImplementedError
