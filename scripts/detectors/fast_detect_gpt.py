# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from .detector_base import DetectorBase
from .model import load_tokenizer, load_model

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()


class FastDetectGPT(DetectorBase):
    def __init__(self, config_name):
        super().__init__(config_name)
        self.criterion_fn = get_sampling_discrepancy_analytic
        self.scoring_tokenizer = load_tokenizer(self.config.scoring_model_name, self.config.cache_dir)
        self.scoring_model = load_model(self.config.scoring_model_name, self.config.device, self.config.cache_dir)
        self.scoring_model.eval()
        if self.config.reference_model_name != self.config.scoring_model_name:
            self.reference_tokenizer = load_tokenizer(self.config.reference_model_name, self.config.cache_dir)
            self.reference_model = load_model(self.config.reference_model_name, self.config.device, self.config.cache_dir)
            self.reference_model.eval()

    def compute_crit(self, text):
        tokenized = self.scoring_tokenizer(text, truncation=True, max_length=self.config.max_token_observed,
                                           return_tensors="pt", padding=True, return_token_type_ids=False).to(self.config.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.config.reference_model_name == self.config.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.reference_tokenizer(text, truncation=True, max_length=self.config.max_token_observed,
                                                     return_tensors="pt", padding=True, return_token_type_ids=False).to(self.config.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.reference_model(**tokenized).logits[:, :-1]
            crit = self.criterion_fn(logits_ref, logits_score, labels)
        return crit


