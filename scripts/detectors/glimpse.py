# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import numpy as np
import tqdm
import argparse
import json
import time
from os import path
from types import SimpleNamespace
from model_proxy import ResponseCache
from .detector_base import DetectorBase


class OpenAIGPT:
    def __init__(self, config):
        self.config = config
        self.cache = ResponseCache(path.join(config.cache_dir, f'{type(self).__name__}.hart.json'))
        if config.api_base.find('azure.com') > 0:
            self.client = self.create_client_azure()
        else:
            self.client = self.create_client_openai()
        # predefined prompts
        self.prompts = {
            "prompt0": "",
            "prompt1": f"You serve as a valuable aide, capable of generating clear and persuasive pieces of writing given a certain context. Now, assume the role of an author and strive to finalize this article.\n",
            "prompt2": f"You serve as a valuable aide, capable of generating clear and persuasive pieces of writing given a certain context. Now, assume the role of an author and strive to finalize this article.\nI operate as an entity utilizing GPT as the foundational large language model. I function in the capacity of a writer, authoring articles on a daily basis. Presented below is an example of an article I have crafted.\n",
            "prompt3": f"System:\nYou serve as a valuable aide, capable of generating clear and persuasive pieces of writing given a certain context. Now, assume the role of an author and strive to finalize this article.\nAssistant:\nI operate as an entity utilizing GPT as the foundational large language model. I function in the capacity of a writer, authoring articles on a daily basis. Presented below is an example of an article I have crafted.\n",
            "prompt4": f"Assistant:\nYou serve as a valuable aide, capable of generating clear and persuasive pieces of writing given a certain context. Now, assume the role of an author and strive to finalize this article.\nUser:\nI operate as an entity utilizing GPT as the foundational large language model. I function in the capacity of a writer, authoring articles on a daily basis. Presented below is an example of an article I have crafted.\n",
        }
        self.max_topk = 10

    def create_client_openai(self):
        from openai import OpenAI
        api_base = self.config.api_base
        api_key = self.config.api_key
        client = OpenAI(
            base_url=api_base,
            api_key=api_key)
        return client

    def create_client_azure(self):
        from openai import AzureOpenAI
        api_base = self.config.api_base
        api_key = self.config.api_key
        api_version = self.config.api_version
        client = AzureOpenAI(
            azure_endpoint=api_base,
            api_key=api_key,
            api_version=api_version)
        return client

    def _response_to_text(self, response):
        obj = vars(response)
        text = json.dumps(obj)
        return text

    def _response_from_text(self, text):
        obj = json.loads(text)
        response = SimpleNamespace(**obj)
        return response

    def evaluate(self, prompt, text):
        model_name = self.config.scoring_model_name
        kwargs = {"model": model_name,
                  "prompt": f"<|endoftext|>{prompt}{text}",
                  "max_tokens": 0, "echo": True, "logprobs": self.max_topk}
        key = self.cache.cachekey(kwargs)
        response_text = self.cache.get_cache(key)
        if response_text is None:
            # retry 1 time
            ntry = 2
            for idx in range(ntry):
                try:
                    response = self.client.completions.create(**kwargs)
                    response = response.choices[0].logprobs
                    break
                except Exception as e:
                    if idx < ntry - 1:
                        print(f'{model_name}, {kwargs}: {e}. Retrying ...')
                        time.sleep(5)
                        continue
                    self.cache.count_exception()
                    raise e
            self.cache.update_cache(key, self._response_to_text(response), model_name)
        else:
            response = self._response_from_text(response_text)
        return response

    def eval(self, text):
        prompt = self.prompts[self.config.prompt]
        # get top tokens
        result = self.evaluate(prompt, text)
        # decide the prefix length
        prefix = ""
        nprefix = 1
        while len(prefix) < len(prompt):
            prefix += result.tokens[nprefix]
            nprefix += 1
        assert prefix == prompt, f"Mismatch: {prompt} .vs. {prefix}"
        tokens = result.tokens[nprefix:]
        logprobs = result.token_logprobs[nprefix:]
        toplogprobs = result.top_logprobs[nprefix:]
        toplogprobs = [dict(item) for item in toplogprobs]
        assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"
        assert len(tokens) == len(toplogprobs), f"Expected {len(tokens)} toplogprobs, got {len(toplogprobs)}"
        return tokens, logprobs, toplogprobs

# probability distribution estimation
def safe_log(prob):
    return np.log(np.array(prob) + 1e-8)

class GeometricDistribution:
    '''
    Top-K probabilities: p_1, p_2, ..., p_K
    Estimated probabilities: Pr(X=k) = p_K * lambda ^ (k - K), for k > K.
    '''
    def __init__(self, top_k, rank_size):
        self.name = "GeometricDistribution"
        self.top_k = top_k
        self.rank_size = rank_size

    def estimate_distrib_token(self, toplogprobs):
        M = self.rank_size  # assuming rank list size
        K = self.top_k  # assuming top-K tokens
        assert K <= M
        toplogprobs = sorted(toplogprobs.values(), reverse=True)
        assert len(toplogprobs) >= K
        toplogprobs = toplogprobs[:K]
        probs = np.exp(toplogprobs)  # distribution over ranks
        if probs.sum() > 1.0:
            # print(f'Warnining: Probability {probs.sum()} excels 1.0')
            probs = probs / (probs.sum() + 1e-6)
        p_K = probs[-1]  # the k-th top token
        p_rest = 1 - probs.sum()  # the rest probability mass
        _lambda = p_rest / (p_K + p_rest)  # approximate the decay factor
        if _lambda ** (M - K + 1) > 1e-6:
            # If the condition was not satisfied, use the following code to calculate the decay factor iteratively
            _lambda_old = _lambda
            last_diff = 1.0
            while True:
                _lambda0 = _lambda
                minor = _lambda ** (M - K + 1)  # the minor part
                assert p_rest > 0, f'Error: Invalid p_rest={p_rest}'
                _lambda = 1 - (_lambda - minor) * p_K / p_rest
                # check convergence
                diff = abs(_lambda - _lambda0)
                if _lambda < 0 or diff < 1e-6 or diff >= last_diff:
                    _lambda = _lambda0
                    break
                last_diff = diff
            # print(f'Warnining: Invalid lambda={_lambda_old}, re-calculate lambda={_lambda}')
        assert p_rest >= 0, f'Error: Invalid p_rest={p_rest}'
        assert 0 <= _lambda <= 1, f'Error: Invalid lambda={_lambda} calculated by p_K={p_K} and p_rest={p_rest}.'
        # estimate the probabilities of the rest tokens
        probs_rest = np.exp(safe_log(p_K) + np.arange(1, M - K + 1) * safe_log(_lambda))
        probs = np.concatenate([probs, probs_rest])
        # check total probability
        # if abs(probs.sum() - 1.0) >= 1e-2:
            # print(f'Warnining: Invalid total probability: {probs.sum()}')
        probs = probs / probs.sum()
        return probs.tolist()

class PdeBase:
    def __init__(self, distrib):
        self.distrib = distrib

    def estimate_distrib_sequence(self, item):
        key = f'{self.distrib.name}-top{self.distrib.top_k}'
        if key in item:
            probs = item[key]
        else:
            toplogprobs = item["toplogprobs"]
            probs = [self.distrib.estimate_distrib_token(v) for v in toplogprobs]
            item[key] = probs
        return np.array(probs)

class PdeFastDetectGPT(PdeBase):
    def __call__(self, item):
        logprobs = item["logprobs"]
        probs = self.estimate_distrib_sequence(item)
        log_likelihood = np.array(logprobs)
        lprobs = np.nan_to_num(np.log(probs))
        mean_ref = (probs * lprobs).sum(axis=-1)
        lprobs2 = np.nan_to_num(np.square(lprobs))
        var_ref = (probs * lprobs2).sum(axis=-1) - np.square(mean_ref)
        discrepancy = (log_likelihood.sum(axis=-1) - mean_ref.sum(axis=-1)) / np.sqrt(var_ref.sum(axis=-1))
        discrepancy = discrepancy.mean()
        return discrepancy.item()


# the detector
class Glimpse(DetectorBase):
    def __init__(self, config_name):
        super().__init__(config_name)
        self.gpt = OpenAIGPT(self.config)
        self.criterion_fn = PdeFastDetectGPT(GeometricDistribution(self.config.top_k, self.config.rank_size))

    def compute_crit(self, text):
        try:
            tokens, logprobs, toplogprobs = self.gpt.eval(text)
            result = { 'text': text, 'tokens': tokens,
                       'logprobs': logprobs, 'toplogprobs': toplogprobs}
            crit = self.criterion_fn(result)
            return crit
        except Exception as e:
            print(e)
            return 0

