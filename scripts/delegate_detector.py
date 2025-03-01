# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import os.path as path
import random
import numpy as np
import tqdm
import argparse
import json
import time
from copy import deepcopy
from detectors import get_detector
from utils import load_json, save_json, is_machine, is_human, get_lang, get_risk_level, parse_record
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVR, SVR
import joblib

def load_datasets(data_path, datasets):
    datasets = datasets.split(',') if type(datasets) == str else datasets
    items = []
    for dataset in datasets:
        dataset_file = path.join(data_path, f'{dataset}.json')
        items.extend(load_json(dataset_file))
    return items

def save_results(results, result_path, dataset, detector):
    result_file = path.join(result_path, f'{dataset}.{detector}.json')
    save_json(result_file, results)
    return result_file

def load_results(result_path, datasets, detector):
    datasets = datasets.split(',') if type(datasets) == str else datasets
    results = []
    for dataset in datasets:
        result_file = path.join(result_path, f'{dataset}.{detector}.json')
        items = load_json(result_file)
        lang = get_lang(dataset)
        if lang != 'en':
            for item in items:
                item['domain'] += f'-{lang}'
        results.extend(items)
    return results

def save_detector(config, model, result_path, category, detector):
    temp_path = path.join(result_path, 'temp')
    if not path.exists(temp_path):
        os.makedirs(temp_path)
    config_file = path.join(temp_path, f'{category}.{detector}.config.json')
    save_json(config_file, config)
    if model is not None:
        model_file = path.join(temp_path, f'{category}.{detector}.model.pkl')
        joblib.dump(model, model_file)

def load_detector(result_path, category, detector):
    temp_path = path.join(result_path, 'temp')
    config_file = path.join(temp_path, f'{category}.{detector}.config.json')
    config = load_json(config_file)
    model_file = path.join(temp_path, f'{category}.{detector}.model.pkl')
    model = joblib.load(model_file) if path.exists(model_file) else None
    return config, model

class DelegateModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = eval(f'{model_name}()')

    def fit(self, xs, ys):
        self.model = self.model.fit(xs, ys)
        return self

    def predict(self, xs):
        if self.model_name == 'LogisticRegression':
            proba = self.model.predict_proba(xs)
            return [v[1] for v in proba]
        ys = self.model.predict(xs)
        return ys

class DelegateDetector:
    def __init__(self, args, name):
        self.args = args
        self.name = name
        self.name2d, self.detector_name = self._split_name(name)
        self.detector = None
        # self.eval_fields = ['generation', 'content', 'language']  # for ablation
        self.eval_fields = ['generation', 'content']  # for final version
        self.feature_fields = self._get_feature_fields(self.name2d)

    def _get_feature_fields(self, name2d):
        abbr_fields = {
            'T': 'generation',  # original text (T)
            'C': 'content',     # content (C)
            'E': 'language',    # language expression (E)
        }
        if name2d is None:
            return [abbr_fields['T']]
        return [abbr_fields[ch] for ch in name2d]

    def _split_name(self, name):
        start = name.find('(')
        end = name.find(')')
        if start >= 0 and end >= 0:
            return name[:start], name[start + 1:end]
        return None, name

    def _prepare(self, dataset):
        # check and skip if result file exists
        result_file = path.join(self.args.result_path, f'{dataset}.{self.detector_name}.json')
        if path.exists(result_file):
            # print(f'Skip preparing, using existing file: {result_file}')
            return
        # initialize or use the cached detector
        if self.detector is None:
            self.detector = get_detector(self.detector_name)
        # compute detection criterion
        print(f'Preparing {dataset}.{self.detector_name}.json ...')
        items = load_datasets(self.args.data_path, dataset)
        results = []
        for item in tqdm.tqdm(items, desc=f"Computing {self.detector_name} criteria"):
            result = deepcopy(item)
            for field in self.eval_fields:
                if field in item:
                    target = item[field]
                    crit = self.detector.compute_crit(target)
                    result[f'{field}_crit'] = crit
            results.append(result)
        save_results(results, self.args.result_path, dataset, self.detector_name)

    def prepare(self, datasets):
        for dataset in datasets:
            self._prepare(dataset)

    def _classify(self, crit, threshold, pos_bigger):
        if pos_bigger:
            return crit >= threshold
        return crit <= threshold

    def _fit_threshold(self, datasets, category, label_fn):
        results = load_results(self.args.result_path, datasets, self.detector_name)
        random.shuffle(results)
        # prepare data
        assert len(self.feature_fields) == 1
        field = self.feature_fields[0]
        pairs = [(item[f'{field}_crit'], label_fn(item)) for item in results]
        pairs = [(0 if str(c) == 'nan' else c, l) for c, l in pairs if l is not None]
        pairs_pos = [(c, l) for c, l in pairs if l]
        pairs_neg = [(c, l) for c, l in pairs if not l]
        if self.args.ndev > 2:
            pairs_pos = pairs_pos[:self.args.ndev // 2]
            pairs_neg = pairs_neg[:self.args.ndev // 2]
            pairs = pairs_pos + pairs_neg
        print(f'Fit threshold on {len(pairs)} samples for {self.name}.')
        # identify direction
        crits_pos = [c for c, l in pairs_pos]
        crits_neg = [c for c, l in pairs_neg]
        pos_bigger = np.mean(crits_pos) > np.mean(crits_neg)
        # find threshold
        crits = [c for c, _ in pairs]
        labels = [l for _, l in pairs]
        precision, recall, thresholds = precision_recall_curve(labels, crits if pos_bigger else -np.array(crits))
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        index = np.argmax(f1)
        threshold = thresholds[index]
        f1 = f1[index]
        threshold = threshold if pos_bigger else -threshold
        # verify classifier
        preds =[self._classify(crit, threshold, pos_bigger) for crit in crits]
        f1_pred = f1_score(labels, preds)
        assert abs(f1_pred - f1) < 1e-2, f'f1_pred {f1_pred} != f1 {f1}'
        # save detector
        config = {'detector': self.detector_name,
                  'fit_data': {'samples': len(pairs), 'positives': int(np.sum(labels)), 'field': field,
                               'pos_mean': float(np.mean(crits_pos)), 'pos_std': float(np.std(crits_pos)),
                               'neg_mean': float(np.mean(crits_neg)), 'neg_std': float(np.std(crits_neg)),},
                  'threshold': threshold, 'pos_bigger': bool(pos_bigger)}
        save_detector(config, None, self.args.result_path, category, self.name)

    def _fit_model(self, datasets, category, label_fn):
        results = load_results(self.args.result_path, datasets, self.detector_name)
        random.shuffle(results)
        # prepare data
        pairs = [([item[f'{field}_crit'] for field in self.feature_fields], label_fn(item))
                 for item in results]
        pairs = [([0 if str(c) == 'nan' else c for c in cs], l) for cs, l in pairs if l is not None]
        pairs_pos = [(c, l) for c, l in pairs if l]
        pairs_neg = [(c, l) for c, l in pairs if not l]
        if self.args.ndev > 2:
            pairs_pos = pairs_pos[:self.args.ndev // 2]
            pairs_neg = pairs_neg[:self.args.ndev // 2]
            pairs = pairs_pos + pairs_neg
        print(f'Fit {self.args.model} on {len(pairs)} samples for {self.name}.')
        # fit a model
        features = [cs for cs, _ in pairs]
        labels = [l for _, l in pairs]
        model = DelegateModel(self.args.model).fit(features, labels)
        crits = model.predict(features)
        crits_pos = [c for c, l in zip(crits, labels) if l]
        crits_neg = [c for c, l in zip(crits, labels) if not l]
        # find threshold
        precision, recall, thresholds = precision_recall_curve(labels, crits)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        index = np.argmax(f1)
        threshold = thresholds[index]
        f1 = f1[index]
        # verify classifier
        preds =[self._classify(crit, threshold, True) for crit in crits]
        f1_pred = f1_score(labels, preds)
        assert abs(f1_pred - f1) < 1e-2, f'f1_pred {f1_pred} != f1 {f1}'
        # save detector
        fields = ','.join(self.feature_fields)
        config = {'detector': self.detector_name,
                  'fit_data': {'samples': len(pairs), 'positives': int(np.sum(labels)), 'fields': fields,
                               'pos_mean': float(np.mean(crits_pos)), 'pos_std': float(np.std(crits_pos)),
                               'neg_mean': float(np.mean(crits_neg)), 'neg_std': float(np.std(crits_neg)), },
                  'threshold': threshold, 'pos_bigger': True}
        save_detector(config, model, self.args.result_path, category, self.name)

    def fit(self, datasets, category, label_fn):
        if len(self.feature_fields) == 1:
            self._fit_threshold(datasets, category, label_fn)
        else:
            self._fit_model(datasets, category, label_fn)

    def _adjust(self, crit, threshold, pos_bigger):
        return (crit - threshold) * (1 if pos_bigger else -1) + threshold

    def _predict_threshold(self, config, model, item):
        assert model is None
        assert len(self.feature_fields) == 1
        field = self.feature_fields[0]
        crit = item[f'{field}_crit']
        crit = crit if crit == crit else 0
        threshold = config['threshold']
        pos_bigger = config['pos_bigger']
        pred = self._classify(crit, threshold, pos_bigger)
        crit = self._adjust(crit, threshold, pos_bigger)
        return crit, pred

    def _predict_model(self, config, model, item):
        features = [[item[f'{field}_crit'] for field in self.feature_fields]]
        features = np.nan_to_num(features, nan=0).tolist()
        crit = model.predict(features)[0]
        threshold = config['threshold']
        pos_bigger = config['pos_bigger']
        pred = self._classify(crit, threshold, pos_bigger)
        crit = self._adjust(crit, threshold, pos_bigger)
        return crit, pred

    def _eval(self, results, config, model, label_fn):
        # predict
        triples = []
        for item in results:
            label = label_fn(item)
            if label is None:
                continue
            if len(self.feature_fields) == 1:
                crit, pred = self._predict_threshold(config, model, item)
            else:
                crit, pred = self._predict_model(config, model, item)
            triples.append((crit, pred, label))
        pos_triples = [t for t in triples if t[2] == 1]
        neg_triples = [t for t in triples if t[2] == 0]
        if len(pos_triples) != len(neg_triples):
            print(f'WARNING: Eval with positive {len(pos_triples)} but negative {len(neg_triples)}')
            min_len = min(len(pos_triples), len(neg_triples))
            pos_triples = random.sample(pos_triples, min_len)
            neg_triples = random.sample(neg_triples, min_len)
            triples = pos_triples + neg_triples
        assert len(pos_triples) > 0
        assert len(neg_triples) > 0
        # auroc, f1, tpr@fpr5%
        crits = [t[0] for t in triples]
        preds = [t[1] for t in triples]
        labels = [t[2] for t in triples]
        fpr, tpr, thresholds = roc_curve(labels, crits)
        auroc = auc(fpr, tpr)
        f1 = f1_score(labels, preds)
        tpr05 = [t for f, t in zip(fpr, tpr) if f <= 0.05]
        tpr05 = tpr05[-1] if len(tpr05) > 0 else 0.0
        return auroc, f1, tpr05, int(np.sum(labels)), len(labels)

    def eval(self, datasets, category, label_fn, group_fn):
        results = load_results(self.args.result_path, datasets, self.detector_name)
        config, model = load_detector(self.args.result_path, category, self.name)
        fit_data = config['fit_data']
        groups = set([None] + [group_fn(item) for item in results])
        # print(f'Eval groups: {groups}')
        report = {}
        for group in groups:
            group_results = [item for item in results if group is None or group_fn(item) == group]
            auroc, f1, tpr05, npos, nsamples = self._eval(group_results, config, model, label_fn)
            report[group] = {'category': category, 'group': group,
                              'npos': npos, 'nsamples': nsamples,
                              'pos_mean': fit_data['pos_mean'], 'pos_std': fit_data['pos_std'],
                              'neg_mean': fit_data['neg_mean'], 'neg_std': fit_data['neg_std'],
                              'auroc': auroc, 'f1': f1, 'tpr05': tpr05}
        return report


def get_level_label_fns(datasets):
    datasets_for_text_detection = ['raid.dev', 'raid.test', 'nonnative.test']
    if all([dataset in datasets_for_text_detection for dataset in datasets]):
        label_fns = {
            'level2': (lambda item: item['task_level2']),
        }
    else:
        label_fns = {
            'level3': (lambda item: item['task_level3']),
            'level2': (lambda item: item['task_level2']),
            'level1': (lambda item: item['task_level1']),
        }
    return label_fns

def print_report(groups, detectors, categories, reports, type='eval'):
    type_templates = {
        'eval': '{auroc:.3f}/{f1:.2f}/{tpr05:.2f}',
        'distrib': '{pos_mean:.2f}({pos_std:.2f})'
    }

    results = []
    for group in groups:
        for idx, detector_name in enumerate(detectors):
            group_str = 'ALL' if group is None else group
            cols0 = [f'{group_str}:']
            cols1 = [f'{detector_name}:']
            for category in categories:
                report = reports[f'{detector_name}-{category}'][group]
                cols0.append('{group}:{category}({npos}/{nsamples})'.format(**report))
                cols1.append(type_templates[type].format(**report))
            if idx == 0:
                results.append('\t'.join(cols0))
            results.append('\t'.join(cols1))

    print('\n'.join(results))

# group name: attack, decoding, repetition_penalty
def main(args):
    label_fns = get_level_label_fns(args.datasets)

    def _group_fn_domain(item):
        return item['domain']

    group_fn = _group_fn_domain  # group by domain
    print(f'Processing {args.datasets} ...')
    categories = list(label_fns.keys())
    groups = set()
    reports = {}
    for detector_name in args.detectors:
        random.seed(args.seed)
        np.random.seed(args.seed)
        detector = DelegateDetector(args, detector_name)
        detector.prepare(args.datasets)
        for category in categories:
            label_fn = label_fns[category]
            devs = [dataset for dataset in args.datasets if dataset.endswith('.dev')]
            tests = [dataset for dataset in args.datasets if dataset.endswith('.test')]
            detector.fit(devs, category, label_fn)
            report = detector.eval(tests, category, label_fn, group_fn)
            reports[f'{detector_name}-{category}'] = report
            groups.update(report.keys())
    # print
    # print_report(groups, args.detectors, categories, reports, 'distrib')
    print_report(groups, args.detectors, categories, reports, 'eval')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default="./exp_main/hart")
    parser.add_argument('--data_path', type=str, default="./benchmark/hart")
    parser.add_argument('--model', type=str, default="SVR", choices=['LogisticRegression', 'LinearSVR', 'SVR'])
    parser.add_argument('--detectors', type=str, default="fast_detect,C(fast_detect),CT(fast_detect)")
    parser.add_argument('--datasets', type=str, default="essay.dev,essay.test")
    parser.add_argument('--ndev', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    args.datasets = args.datasets.split(',')
    args.detectors = args.detectors.split(',')

    random.seed(args.seed)
    np.random.seed(args.seed)
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
