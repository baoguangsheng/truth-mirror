# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
from io import open
from collections import Counter
import numpy as np

def load_json(filename):
    with open(filename, encoding='utf-8') as f:
        return json.load(f)

def save_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_text(filename):
    with open(filename) as f:
        return f.read()

def save_text(filename, text):
    with open(filename, 'w') as f:
        f.write(text)

def get_lang(dataset):
    dataset = dataset.split('.')[0]
    parts = dataset.split('-')
    return parts[1] if len(parts) > 1 else 'en'

def full_lang(lang):
    langs = {'en': 'English',
             'zh': 'Chinese',
             'ar': 'Arabic',
             'fr': 'French',
             'es': 'Spanish',}
    return langs[lang]

def is_machine(source):
    source = source.split(':')[0]
    if source not in ['human', 'machine', 'rephrase', 'humanize']:
        raise Exception(f'Source {source} is not valid')
    return source in ['machine', 'rephrase']

def is_human(source):
    source = source.split(':')[0]
    if source not in ['human', 'machine', 'rephrase', 'humanize']:
        raise Exception(f'Source {source} is not valid')
    return source in ['human', 'humanize']

def get_risk_level(item):
    content_source = item['content_source']
    if 'language_source' in item:
        language_source = item['language_source']
        if is_human(content_source) and is_human(language_source):
            return 'level0'
        elif is_human(content_source) and is_machine(language_source):
            return 'level1'
        elif is_machine(content_source) and is_human(language_source):
            return 'level2'
        elif is_machine(content_source) and is_machine(language_source):
            return 'level3'
        else:
            raise Exception
    else:
        if is_human(content_source):
            return 'level0'
        elif is_machine(content_source):
            return 'level2'
        else:
            raise Exception

def split_by_level(items):
    level0 = [item for item in items if get_risk_level(item) == 'level0']
    level1 = [item for item in items if get_risk_level(item) == 'level1']
    level2 = [item for item in items if get_risk_level(item) == 'level2']
    level3 = [item for item in items if get_risk_level(item) == 'level3']
    return level0, level1, level2, level3

def validate_common(model, id, response, lang='en', raise_exception=False):
    if response is None:
        if raise_exception:
            raise Exception(f'{model}, {id}: None response.')
        return False
    # check empty
    response = response.strip()
    if len(response) == 0:
        if raise_exception:
            raise Exception(f'{model}, {id}: Empty response.')
        return False
    # check length: at least 30 characters for Chinese and 20 words for other languages
    n_units, min_units = (len(response), 30) if lang in ['zh'] else (len(response.split()), 5)
    if n_units < min_units:
        if raise_exception:
            raise Exception(f'{model}, {id}: Short response: {response}')
        return False
    return True

def validate_repetition(model, id, response, lang='en', raise_exception=False):
    if lang in ['zh']:  # 20 characters repeating 5 times
        span = 20
        repeat = 5
        ngrams = [response[i:i + span] for i in range(len(response))]
        counter = Counter(ngrams)
    else:  # 10 words repeating 5 times
        span = 10
        repeat = 5
        words = response.split()
        ngrams = [' '.join(words[i:i+span]) for i in range(len(words))]
        counter = Counter(ngrams)
    # check the most common case
    ngram, count = counter.most_common(1)[0]
    if count >= repeat:
        if raise_exception:
            raise Exception(f'{model}, {id}: Repetition ({ngram}, {count}) in response: {response}')
        return False
    return True

def validate_garbage(model, id, response, lang='en', raise_exception=False):
    if lang in ['zh']:
        span = np.mean([len(part) for part in response.split()])
        if span < 10:
            if raise_exception:
                raise Exception(f'{model}, {id}: Garbage response: {response}')
            return False
    else:
        span = np.mean([len(part) for part in response.split()])
        if span > 10:
            if raise_exception:
                raise Exception(f'{model}, {id}: Garbage response: {response}')
            return False
    return True


# record like "Generate:model(qwen2.5-72b-instruct),temperature(1.2),top_p(1.0),freq_penalty(0.0),presence_penalty(0.0),task_prompt(Write a news article (no title) in 6 paragraphs and 225 words based on the given title:\n{title}\n)."
# action like "Generate"
def parse_record(record, action):
    assert record.startswith(action), f'Wrong record: expect {action} but {record}'
    record = record[len(action) + 1:]
    record = record[:-1] if record.endswith(')') else record
    record = record[:-2] if record.endswith(').') else record
    record = [item.split('(') for item in record.split('),')]
    record = [item for item in record if len(item) == 2]
    return dict(record)
