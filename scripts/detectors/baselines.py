# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F
from .detector_base import DetectorBase
from .model import load_tokenizer, load_model

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean().item()

def get_rank(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1 # convert to 1-indexed rank
    return -ranks.mean().item()

def get_logrank(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    # matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1  # convert to 1-indexed rank
    ranks = torch.log(ranks)
    return -ranks.mean().item()

def get_entropy(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
    entropy = -entropy.sum(-1)
    return entropy.mean().item()


class Baselines(DetectorBase):
    def __init__(self, name):
        super().__init__(name)
        self.scoring_tokenizer = load_tokenizer(self.config.scoring_model_name, self.config.cache_dir)
        self.scoring_model = load_model(self.config.scoring_model_name, self.config.device, self.config.cache_dir)
        self.scoring_model.eval()
        criterion_fns = {'likelihood': get_likelihood,
                         'rank': get_rank,
                         'logrank': get_logrank,
                         'entropy': get_entropy}
        self.criterion_fn = criterion_fns[self.config.baseline]

    def compute_crit(self, text):
        tokenized = self.scoring_tokenizer(text, truncation=True, max_length=self.config.max_token_observed,
                                           return_tensors="pt", padding=True, return_token_type_ids=False).to(self.config.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits = self.scoring_model(**tokenized).logits[:, :-1]
            crit = self.criterion_fn(logits, labels)
        return crit
