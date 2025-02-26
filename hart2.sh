#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

data_path=./benchmark/hart
result_path=./exp_main/hart
mkdir -p $result_path

# Multilingual datasets
datasets='news.dev,news.test,news-zh.dev,news-zh.test,news-fr.dev,news-fr.test,news-es.dev,news-es.test,news-ar.dev,news-ar.test'
detectors='roberta,radar,log_perplexity,log_rank,lrr,fast_detect,C(fast_detect),CT(fast_detect),binoculars,C(binoculars),CT(binoculars),glimpse,C(glimpse),CT(glimpse)'

python scripts/delegate_detector.py --data_path $data_path --result_path $result_path \
              --datasets $datasets --detectors $detectors --categorize level
