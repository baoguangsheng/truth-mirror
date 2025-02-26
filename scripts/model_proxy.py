# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import time
from collections import Counter
from utils import load_json, save_json
import hashlib

class ResponseCache:
    def __init__(self, cache_file, n_update=20):
        self.n_update = n_update
        self.cache_file = cache_file
        if os.path.exists(self.cache_file):
            self.cache = load_json(self.cache_file)
            self.n_saved = len(self.cache)
        else:
            self.cache = {}
            self.n_saved = 0
        self.counter = Counter()

    def __del__(self):
        save_json(self.cache_file, self.cache)
        self.n_saved = len(self.cache)
        if len(self.counter) > 0:
            print(f'{self.cache_file} new responses: {self.counter}')

    def cachekey(self, kwargs, identifier=None):
        key = json.dumps(kwargs) + (identifier if identifier else '')
        key = hashlib.md5(key.encode()).hexdigest()
        return key

    def update_cache(self, key, response, category):
        assert len(key) > 0 and len(response) > 0
        self.cache[key] = response
        self.counter[category] += 1
        if len(self.cache) >= self.n_saved + self.n_update:
            save_json(self.cache_file, self.cache)
            self.n_saved = len(self.cache)
            # print(f'Saved cache with {self.n_saved} items.')

    def count_exception(self):
        self.counter['exception'] += 1

    def get_cache(self, key):
        if key in self.cache:
            return self.cache[key]
        return None


class ModelProxy:
    def __init__(self, max_tokens, cache):
        self.max_tokens = max_tokens
        self.cache = cache
        # model clients
        self.clients = dict()
        self.clients.update(self.prepare_client())

    def prepare_client(self):
        from openai import OpenAI
        api_base = os.getenv('API_BASE')
        api_key = os.getenv('API_KEY')
        client = OpenAI(base_url=api_base, api_key=api_key)
        clients = {
            'gpt-4o-2024-11-20': ('gpt-4o', client),
        }
        return clients

    def prompt_generate(self, model_name, sys_prompt, task_prompt, identifier=None, temperature=1.0, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, validate_fn=None):
        kwargs = {"model": model_name, "max_tokens": self.max_tokens,
                  "temperature": temperature, "top_p": top_p,
                  'frequency_penalty': frequency_penalty, 'presence_penalty': presence_penalty,
                  "messages": [{"role": "system", "content": sys_prompt},
                               {"role": "user", "content": task_prompt}],
                  }
        key = self.cache.cachekey(kwargs, identifier)
        response = self.cache.get_cache(key)
        if response is None or not validate_fn(response):
            # mapping model to client and real model name
            if model_name not in self.clients:
                raise Exception('No client for model: {}'.format(model_name))
            model_name, client = self.clients[model_name]
            kwargs['model'] = model_name
            # retry 1 time
            ntry = 2
            for idx in range(ntry):
                try:
                    response = client.chat.completions.create(**kwargs)
                    response = response.choices[0].message.content
                    validate_fn(response, raise_exception=True)
                    break
                except Exception as e:
                    if idx < ntry - 1:
                        print(f'{model_name}, {identifier}, {kwargs}')
                        print(f'{model_name}: {e}. {response}. Retrying ...')
                        time.sleep(5)
                        continue
                    self.cache.count_exception()
                    raise e
            self.cache.update_cache(key, response, model_name)
        return response

