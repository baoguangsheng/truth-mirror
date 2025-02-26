# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from utils import load_json
from types import SimpleNamespace

class DetectorBase:
    def __init__(self, config_name):
        self.config_name = config_name
        self.config = self.load_config(config_name)

    def load_config(self, config_name):
        config = load_json(f'./scripts/detectors/configs/{config_name}.json')
        for key in config:
            val = config[key]
            if type(val) == str and val.startswith('${') and val.endswith('}'):
                var = val[2:-1]
                config[key] = os.getenv(var)
                print(f'Config entry solved: {key} -> {val}')
        return SimpleNamespace(**config)

    def compute_crit(self, text):
        raise NotImplementedError

    def __str__(self):
        return self.config_name
