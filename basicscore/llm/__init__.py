from typing import TypeAlias, Sequence, List
from abc import ABCMeta, abstractmethod


EmbeddingProbabilities: TypeAlias = Sequence[float]


class LLM(metaclass=ABCMeta):
    @abstractmethod
    def prompt_probs(self, tokens: List[int]) -> List[EmbeddingProbabilities]:
        pass

    @abstractmethod
    def tokenize(self, text: bytes, add_bos=True) -> List[int]:
        pass

    @abstractmethod
    def detokenize(self, tokens: List[int]) -> bytes:
        pass

    @abstractmethod
    def n_ctx(self) -> int:
        pass
