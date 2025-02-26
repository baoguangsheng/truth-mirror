# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os.path as path
from model_proxy import ModelProxy, ResponseCache
from utils import get_lang, validate_common, validate_repetition, validate_garbage


SYSTEM_PROMPT = 'You are a professional writer.'

class ExtractorBase:
    def __init__(self, args, dataset):
        self.args = args
        self.cache = ResponseCache(path.join(args.cache_dir, f'{type(self).__name__}.{dataset}.json'))
        self.model_proxy = ModelProxy(4096, self.cache)
        # prompts for system role
        self.sys_prompt = SYSTEM_PROMPT
        # model settings
        self.model = args.model
        self.temperature = args.temperature
        self.top_p = 1.0
        self.lang = get_lang(dataset)
        self.gen_lens = []
        self.res_lens = []

    def _get_validate_fn(self, item, model_name):
        id = item['id']

        def _validate_response(response, raise_exception=False):
            # basic validation
            valid = validate_common(model_name, id, response, self.lang, raise_exception)
            if not valid:
                return False
            # check repetition
            valid = validate_repetition(model_name, id, response, self.lang, raise_exception)
            if not valid:
                return False
            # check garbage
            valid = validate_garbage(model_name, id, response, self.lang, raise_exception)
            if not valid:
                return False
            # pass validation
            return True

        return _validate_response