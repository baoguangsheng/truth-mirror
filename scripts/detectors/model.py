# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os

def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    local_path = os.path.join(cache_dir, model_name.replace("/", "_"))
    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    model = cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)
    model.save_pretrained(local_path)
    return model

# predefined models
model_fullnames = {  'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
                     'gpt-j-6B': 'EleutherAI/gpt-j-6B',
                     'falcon-7b': 'tiiuae/falcon-7b',
                     'falcon-7b-instruct': 'tiiuae/falcon-7b-instruct',
                     }
float16_models = ['gpt-j-6B', 'falcon-7b', 'falcon-7b-instruct']

cache_models = dict()
cache_tokenizers = dict()
def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def load_model(model_name, device, cache_dir):
    global cache_models
    if model_name in cache_models:
        return cache_models[model_name]
    # load model
    model_fullname = get_model_fullname(model_name)
    print(f'Loading model {model_fullname}...')
    model_kwargs = {}
    if model_name in float16_models:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in model_name:
        model_kwargs.update(dict(revision='float16'))
    model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')
    cache_models[model_name] = model
    return model

def load_tokenizer(model_name, cache_dir):
    global cache_tokenizers
    if model_name in cache_tokenizers:
        return cache_tokenizers[model_name]
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}
    optional_tok_kwargs['padding_side'] = 'right'
    base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    cache_tokenizers[model_name] = base_tokenizer
    return base_tokenizer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="bloom-7b1")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    load_tokenizer(args.model_name, 'xsum', args.cache_dir)
    load_model(args.model_name, 'cpu', args.cache_dir)
