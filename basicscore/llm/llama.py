#!/usr/bin/env python3
from basicscore.llm import LLM, EmbeddingProbabilities
import numpy as np
from typing import List
from llama_cpp import Llama as LlamaCpp

class Llama(LLM):
    def __init__(self, path, gpu_layers, threads, n_ctx):
        self._llama = LlamaCpp(
            model_path=str(path),
            n_ctx=n_ctx,
            verbose=False,
            n_threads=threads,
            n_gpu_layers=gpu_layers,
            logits_all=True
        )

    def prompt_probs(self, tokens: List[int]) -> List[EmbeddingProbabilities]:
        self._llama.reset()
        self._llama.eval(tokens)
        logits = self._llama._scores
        probs = np.exp(logits)
        return probs / np.sum(probs, axis=1)[:,None]

    def tokenize(self, text: bytes, add_bos=True) -> List[int]:
        return self._llama.tokenize(text, add_bos=add_bos)

    def detokenize(self, tokens: List[int]) -> bytes:
        return self._llama.detokenize(tokens)

    def n_ctx(self) -> int:
        return self._llama.n_ctx()
