# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F
from .detector_base import DetectorBase
from .model import from_pretrained
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Radar(DetectorBase):
    def __init__(self, config_name):
        super().__init__(config_name)
        self.detector = from_pretrained(AutoModelForSequenceClassification, self.config.scoring_model_name, {}, self.config.cache_dir)
        self.tokenizer = from_pretrained(AutoTokenizer, self.config.scoring_model_name, {}, self.config.cache_dir)
        self.detector.eval()
        self.detector.to(self.config.device)

    def compute_crit(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(text, padding=True, truncation=True, max_length=self.config.max_token_observed, return_tensors="pt")
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            output_probs = F.log_softmax(self.detector(**inputs).logits, -1)[:, 0].exp().tolist()
            return output_probs[0]

