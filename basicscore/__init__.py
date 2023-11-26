from basicscore.llm.llama import Llama
from basicscore.llm import LLM
from tqdm import tqdm
import dataclasses
from pathlib import Path
from typing import List, Tuple


@dataclasses.dataclass
class Config:
    version: str
    model: Path
    threads: int
    gpu_layers: int
    score: str
    context_prompt: int
    n_ctx: int

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


class BacktrackPrompter():
    def __init__(self, n: int):
        self.window_context = n

    def __call__(self, path: Path, text_enc: List[int], progress: int) -> List[int]:
        if progress < self.window_context :
            return [text_enc[0]]
        context_content = text_enc[progress-self.window_context:progress]
        # Set first character to bos-token
        context_content[0] = text_enc[0]
        return context_content


class BasicScore():
    def __init__(self, llm: LLM, prompter, scorer):
        self.llm = llm
        self.prompter = prompter
        self.scorer = scorer
        self.probs :List[Tuple[bytes, float]] = []

    def score_unit(self, path, text: bytes, update_fn):
        self.probs = self.seq_probabilities(path, text, update_fn)
        return self.scorer(self.probs)

    def select(self, tokens, embs):
        return list((tok, float(embs[i][tok])) for i, tok in enumerate(tokens))

    def string_probs(self, token_probs, lstrip_first=False):
        out = []
        tokens = []
        probs = []
        for i, (token, prob) in enumerate(token_probs):
            tokens.append(token)
            probs.append(prob)
            try:
                tok_decode = str(self.llm.detokenize(tokens), "utf-8")
                if lstrip_first and i == 0:
                    tok_decode = tok_decode.lstrip()
                if len(tok_decode) > 0:
                    out.append((tok_decode, min(probs)))
                tokens = []
                probs = []
            except ValueError as e:
                pass

        return out

    def predict(self, text_enc, update_fn, context=None):
        if context is None:
            context = []

        tokens = context + text_enc
        assert self.llm.n_ctx() >= len(tokens)
        embedding_probs = self.llm.prompt_probs(tokens)
        for _ in text_enc:
            update_fn() #TODO: change this to take progress count
        return [emb for emb in embedding_probs[len(context)-1:]]

    def seq_probabilities(self, path, text: bytes, update_fn):
        n_ctx = self.llm.n_ctx()
        text_enc = self.llm.tokenize(text, add_bos=False)
        start = 0
        token_probs = []
        context_enc = [1]
        while True:
            end = start + n_ctx - len(context_enc)
            embs = self.predict(text_enc[start:end], update_fn, context_enc)
            tps = self.select(text_enc[start:end], embs)
            pps = self.string_probs(tps, text[0] != b' ')

            token_probs += pps
            start += len(tps)
            context_enc = self.prompter(path, text_enc, start)

            if start >= len(text_enc):
                break
        return token_probs


def average_prob_score(token_probs: List[Tuple[str, float]]) -> float:
    return sum(prob for token, prob in token_probs) / len(token_probs)


def from_config(config: Config, update_fn) -> BasicScore:
    assert(config.score == "avgprob")
    prompter = BacktrackPrompter(config.context_prompt)
    return BasicScore(
        Llama(config.model, config.gpu_layers, config.threads, config.n_ctx),
        prompter,
        average_prob_score
    )
