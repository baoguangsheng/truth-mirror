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
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse, Patch
import matplotlib.transforms as transforms
from delegate_detector import load_results
from utils import split_by_level, parse_record

dataset_names = {
    'essay.dev': 'Student Essay',
    'arxiv.dev': 'ArXiv Intro',
    'writing.dev': 'Creative Writing',
    'news.dev': 'CC News',
    'essay.test': 'Student Essay',
    'arxiv.test': 'ArXiv Intro',
    'writing.test': 'Creative Writing',
    'news.test': 'CC News'
}

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def draw_human_2d_distrib(args):
    # load results
    datasets = 'humanize.test,humanize.test'.split(',')
    detector = 'fast_detect'
    results = {}
    for dataset in datasets:
        result_path = path.join(args.result_path, 'manual')
        results[dataset] = split_by_level(load_results(result_path, [dataset], detector))

    # figure
    labels = ['Type 0', 'Type 1', 'Type 2', 'Type 3']
    colors = ['tab:green', 'tab:olive', 'tab:pink', 'tab:red']

    # plot
    nrows = 1
    ncols = len(datasets)
    plt.clf()
    fig = plt.figure(figsize=(2 * ncols, 2 * nrows))
    grids = fig.add_gridspec(nrows, ncols)
    axs = grids.subplots(sharex=True, sharey=True)

    random.seed(0)
    for j in range(len(datasets)):
        dataset = datasets[j]
        axs[j].set_axisbelow(True)
        axs[j].grid(axis='x', color='lightgrey', lw=0.2, linestyle='-')
        axs[j].grid(axis='y', color='lightgrey', lw=0.2, linestyle='-')
        axs[j].set_title(dataset_names[dataset], fontsize=10)
        axs[j].set_xlabel('Content ($C_2$)')
        levels = results[dataset]
        for l in range(len(levels)):
            level = levels[l]
            level = [item for item in level if 'change_ratio' not in item or item['change_ratio'] > 0.]
            if len(level) == 0:
                continue
            xs = np.array([item['content_crit'] for item in level])
            ys = np.array([item['generation_crit'] for item in level])
            confidence_ellipse(xs, ys, axs[j], 1.0, color=colors[l], fill=False, lw=0.5)
            confidence_ellipse(xs, ys, axs[j], 0.01, color=colors[l], fill=False, lw=1.0)

    axs[0].set_ylabel('Expression ($E_2$)')

    # axs[0].legend(loc="lower right", fontsize=5, ncol=1)
    legend = [Patch(color=colors[l], label=labels[l], fill=False) for l in range(len(labels))]
    plt.figlegend(handles=legend, loc='lower center', fontsize=6.5, ncol=4, handlelength=1.5)

    plt.xlim(-2.0, 7.0)
    plt.ylim(-2.0, 7.0)
    # plt.yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    # plt.xticks(xs)
    plt.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.subplots_adjust(bottom=0.35)
    plt.savefig(path.join(args.output_path, 'human_2d_distrib.pdf'))

def draw_domain_2d_distrib(args):
    # load results
    datasets = 'essay.dev,arxiv.dev,writing.dev,news.dev'.split(',')
    detector = 'fast_detect'
    results = {}
    for dataset in datasets:
        result_path = path.join(args.result_path, 'final500')
        results[dataset] = split_by_level(load_results(result_path, [dataset], detector))

    # figure
    labels = ['Type 0', 'Type 1', 'Type 2', 'Type 3']
    colors = ['tab:green', 'tab:olive', 'tab:pink', 'tab:red']

    # plot
    nrows = 1
    ncols = len(datasets)
    plt.clf()
    fig = plt.figure(figsize=(1.5 * ncols, 2 * nrows))
    grids = fig.add_gridspec(nrows, ncols)
    axs = grids.subplots(sharex=True, sharey=True)

    random.seed(0)
    for j in range(len(datasets)):
        dataset = datasets[j]
        axs[j].set_axisbelow(True)
        axs[j].grid(axis='x', color='lightgrey', lw=0.2, linestyle='-')
        axs[j].grid(axis='y', color='lightgrey', lw=0.2, linestyle='-')
        axs[j].set_title(dataset_names[dataset], fontsize=10)
        axs[j].set_xlabel('Content ($C_2$)')
        levels = results[dataset]
        for l in range(len(levels)):
            level = levels[l]
            level = [item for item in level]
            xs = np.array([item['content_crit'] for item in level])
            ys = np.array([item['language_crit'] for item in level])
            confidence_ellipse(xs, ys, axs[j], 1.0, color=colors[l], fill=False, lw=0.5)
            confidence_ellipse(xs, ys, axs[j], 0.01, color=colors[l], fill=False, lw=1.0)

    axs[0].set_ylabel('Expression ($E_2$)')

    # axs[0].legend(loc="lower right", fontsize=5, ncol=1)
    legend = [Patch(color=colors[l], label=labels[l], fill=False) for l in range(len(labels))]
    plt.figlegend(handles=legend, loc='lower center', fontsize=8, ncol=4, handlelength=1.5)

    plt.xlim(-2.0, 7.0)
    plt.ylim(-2.0, 7.0)
    # plt.yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    # plt.xticks(xs)
    plt.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.subplots_adjust(bottom=0.35)
    plt.savefig(path.join(args.output_path, 'domain_2d_distrib.pdf'))

def draw_domain_2d_distrib_refinement(args):
    # load results
    datasets = 'essay.dev,arxiv.dev,writing.dev,news.dev'.split(',')
    detector = 'fast_detect'
    results = {}
    for dataset in datasets:
        result_path = path.join(args.result_path, 'final500')
        level0, level1, level2, level3 = split_by_level(load_results(result_path, [dataset], detector))
        level10 = [item for item in level1 if item['process_records'][-1].find('action(polish)') > 0]
        level11 = [item for item in level1 if item['process_records'][-1].find('action(restructure)') > 0]
        results[dataset] = [level0, level10, level11]
    # figure
    labels = ['Type 0', 'Type 1 (polish)', 'Type 1 (restructure)']
    colors = ['tab:green', 'tab:olive', 'tab:olive']
    lines = ['solid', 'dotted', 'dashed']

    # plot
    nrows = 1
    ncols = len(datasets)
    plt.clf()
    fig = plt.figure(figsize=(1.5 * ncols, 2 * nrows))
    grids = fig.add_gridspec(nrows, ncols)
    axs = grids.subplots(sharex=True, sharey=True)

    random.seed(0)
    n_samples = 200
    for j in range(len(datasets)):
        dataset = datasets[j]
        axs[j].set_axisbelow(True)
        axs[j].grid(axis='x', color='lightgrey', lw=0.2, linestyle='-')
        axs[j].grid(axis='y', color='lightgrey', lw=0.2, linestyle='-')
        axs[j].set_title(dataset_names[dataset], fontsize=10)
        axs[j].set_xlabel('Content ($C_2$)')
        levels = results[dataset]
        centers = []
        for l in range(len(levels)):
            level = levels[l]
            level = [item for item in level]
            random.shuffle(level)
            level = level[:n_samples]
            xs = np.array([item['content_crit'] for item in level])
            ys = np.array([item['language_crit'] for item in level])
            confidence_ellipse(xs, ys, axs[j], 1.0, color=colors[l], fill=False, lw=0.5, linestyle=lines[l])
            confidence_ellipse(xs, ys, axs[j], 0.02, color=colors[l], fill=False, lw=1)
            centers.append((np.mean(xs), np.mean(ys)))
        # draw arrow
        x, y = centers[0]
        dx = centers[1][0] - x
        dy = centers[1][1] - y
        axs[j].arrow(x, y, dx, dy, lw=0.1, color='tab:red', length_includes_head=True, head_width=0.05)
        x, y = centers[0]
        dx = centers[2][0] - x
        dy = centers[2][1] - y
        axs[j].arrow(x, y, dx, dy, lw=0.1, color='tab:red', length_includes_head=True, head_width=0.05)

    axs[0].set_ylabel('Expression ($E_2$)')

    legend = [Patch(color=colors[l], label=labels[l], fill=False, linestyle=lines[l]) for l in range(len(labels))]
    plt.figlegend(handles=legend, loc='lower center', fontsize=8, ncol=4, handlelength=1.5)

    plt.xlim(-1.0, 4.0)
    plt.ylim(-1.0, 4.0)
    plt.yticks([0, 3.0])
    plt.xticks([0, 3.0])
    plt.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.subplots_adjust(bottom=0.35)
    plt.savefig(path.join(args.output_path, 'domain_2d_distrib_refinement.pdf'))

def draw_domain_2d_distrib_humanize(args):
    # load results
    datasets = 'essay.test,arxiv.test,writing.test,news.test'.split(',')
    detector = 'fast_detect'
    results = []
    for dataset in datasets:
        result_path = path.join(args.result_path, 'final500')
        level0, level1, level2, level3 = split_by_level(load_results(result_path, [dataset], detector))
        level20 = [item for item in level2 if item['process_records'][-1].find('action(diversify)') > 0]
        level21 = [item for item in level2 if item['process_records'][-1].find('action(mimic)') > 0]
        level22 = [item for item in level2 if item['process_records'][-1].find('action(https') > 0]
        level23 = [item for item in level2 if item['process_records'][-1].find('action(human-editing)') > 0]
        results[dataset] = [level3, level20, level21, level22, level23]
    # figure
    labels = ['Type 3', 'Type 2 (diversify)', 'Type 2 (mimic)', 'Type 2 (tools)', 'Type 2 (human)']
    colors = ['tab:red', 'tab:pink', 'tab:pink', 'tab:pink', 'tab:pink']
    lines = ['solid', 'dotted', 'dashed', 'dashdot', 'solid']

    # plot
    nrows = 1
    ncols = len(datasets)
    plt.clf()
    fig = plt.figure(figsize=(1.5 * ncols, 2 * nrows))
    grids = fig.add_gridspec(nrows, ncols)
    axs = grids.subplots(sharex=True, sharey=True)

    random.seed(0)
    n_samples = 200
    for j in range(len(datasets)):
        dataset = datasets[j]
        axs[j].set_axisbelow(True)
        axs[j].grid(axis='x', color='lightgrey', lw=0.2, linestyle='-')
        axs[j].grid(axis='y', color='lightgrey', lw=0.2, linestyle='-')
        axs[j].set_title(dataset_names[dataset], fontsize=10)
        axs[j].set_xlabel('Content ($C_2$)')
        levels = results[dataset]
        centers = []
        for l in range(len(levels)):
            level = levels[l]
            level = [item for item in level]
            random.shuffle(level)
            level = level[:n_samples]
            xs = np.array([item['content_crit'] for item in level])
            ys = np.array([item['language_crit'] for item in level])
            confidence_ellipse(xs, ys, axs[j], 1.0, color=colors[l], fill=False, lw=0.5, linestyle=lines[l])
            confidence_ellipse(xs, ys, axs[j], 0.02, color=colors[l], fill=False, lw=1)
            centers.append((np.mean(xs), np.mean(ys)))
        # draw arrow
        for i in range(1, len(centers)):
            x, y = centers[0]
            dx = centers[i][0] - x
            dy = centers[i][1] - y
            axs[j].arrow(x, y, dx, dy, lw=0.1, length_includes_head=True, head_width=0.05)

    axs[0].set_ylabel('Expression ($E_2$)')

    legend = [Patch(color=colors[l], label=labels[l], fill=False, linestyle=lines[l]) for l in range(len(labels))]
    plt.figlegend(handles=legend, loc='lower center', fontsize=8, ncol=4, handlelength=1.5)

    plt.xlim(-2.0, 7.0)
    plt.ylim(-2.0, 7.0)
    plt.yticks([0, 5.0])
    plt.xticks([0, 5.0])
    plt.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.subplots_adjust(bottom=0.35)
    plt.savefig(path.join(args.output_path, 'domain_2d_distrib_humanize.pdf'))

def draw_dataset_2d_distrib(args):
    # load results
    datasets = 'essay.test,arxiv.test,writing.test,news.test'.split(',')
    detector = 'fast_detect'
    results = {'type': [[], [], [], []],
               'refine': [[], [], []],
               'humanize': [[], [], [], [], []]}
    for dataset in datasets:
        result_path = path.join(args.result_path, 'final500')
        level0, level1, level2, level3 = split_by_level(load_results(result_path, [dataset], detector))
        level10 = [item for item in level1 if item['process_records'][-1].find('action(polish)') > 0]
        level11 = [item for item in level1 if item['process_records'][-1].find('action(restructure)') > 0]
        level20 = [item for item in level2 if item['process_records'][-1].find('action(diversify)') > 0]
        level21 = [item for item in level2 if item['process_records'][-1].find('action(mimic)') > 0]
        level22 = [item for item in level2 if item['process_records'][-1].find('action(https') > 0]
        level23 = [item for item in level2 if item['process_records'][-1].find('action(human-editing)') > 0]
        for group, items in zip(results['type'], [level0, level1, level2, level3]):
            group.extend(items)
        for group, items in zip(results['refine'], [level0, level10, level11]):
            group.extend(items)
        for group, items in zip(results['humanize'], [level3, level20, level21, level22, level23]):
            group.extend(items)
    # figure
    labels = {'type': ['type 0', 'type 1', 'type 2', 'type 3'],
              'refine': ['type 0', 'polish', 'restructure'],
              'humanize': ['type 3', 'diversify', 'mimic', 'AI tool', 'human']}
    colors = {'type': ['tab:green', 'tab:olive', 'tab:pink', 'tab:red'],
              'refine': ['tab:green', 'tab:olive', 'tab:olive'],
              'humanize': ['tab:red', 'tab:pink', 'tab:pink', 'tab:pink', 'tab:pink']}
    lines = {'type': ['solid', 'solid', 'solid', 'solid'],
             'refine': ['solid', 'dotted', 'dashed'],
             'humanize': ['solid', 'dotted', 'dashed', 'dashdot', 'solid']}
    titles = {'type': 'Dataset',
              'refine': 'Refinement',
              'humanize': 'Humanizing'}

    # plot
    nrows = 1
    ncols = len(titles)
    plt.clf()
    fig = plt.figure(figsize=(2 * ncols, 2 * nrows))
    grids = fig.add_gridspec(nrows, ncols)
    axs = grids.subplots(sharex=True, sharey=True)

    for j, group in enumerate(titles):
        axs[j].set_axisbelow(True)
        axs[j].grid(axis='x', color='lightgrey', lw=0.2, linestyle='-')
        axs[j].grid(axis='y', color='lightgrey', lw=0.2, linestyle='-')
        axs[j].set_title(titles[group], fontsize=10)
        axs[j].set_xlabel('Content ($C_2$)')
        levels = results[group]
        centers = []
        for l in range(len(levels)):
            level = levels[l]
            xs = np.array([item['content_crit'] for item in level])
            ys = np.array([item['language_crit'] for item in level])
            confidence_ellipse(xs, ys, axs[j], 1.0, color=colors[group][l], fill=False, lw=0.5, linestyle=lines[group][l])
            confidence_ellipse(xs, ys, axs[j], 0.02, color=colors[group][l], fill=False, lw=1)
            centers.append((np.mean(xs), np.mean(ys)))
        # draw arrow
        if j > 0:
            for i in range(1, len(centers)):
                x, y = centers[0]
                dx = centers[i][0] - x
                dy = centers[i][1] - y
                axs[j].arrow(x, y, dx, dy, lw=0.1, length_includes_head=True, head_width=0.05)

        legend = [Patch(color=colors[group][l], label=labels[group][l], fill=False, linestyle=lines[group][l]) for l in range(len(labels[group]))]
        axs[j].legend(handles=legend, loc='lower right', fontsize=6.5, ncol=1, handlelength=1.5)

    axs[0].set_ylabel('Expression ($E_2$)')

    plt.xlim(-1.0, 9.0)
    plt.ylim(-2.0, 6.0)
    plt.yticks([0, 4.0])
    plt.xticks([0, 4.0, 8])
    plt.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.subplots_adjust(left=0.07, bottom=0.25, right=0.99)
    plt.savefig(path.join(args.output_path, 'dataset_2d_distrib.pdf'))

def draw_group_2d_distrib(args):
    group_names = ['model', 'temperature', 'top_p', 'freq_penalty', 'presence_penalty']
    group_titles = {
        'model': 'Source Model',
        'temperature': 'Temperature',
        'top_p': 'Top-$p$',
        'freq_penalty': 'Frequency Penalty',
        'presence_penalty': 'Presence Penalty',
    }
    width_ratios = [1.7, 1, 1, 1, 1]
    # load results
    datasets = 'essay.dev,arxiv.dev,writing.dev,news.dev'.split(',')
    detector = 'fast_detect'
    result_path = path.join(args.result_path, 'final500')
    levels = split_by_level(load_results(result_path, datasets, detector))

    # plot
    nrows = 1
    ncols = len(group_names)
    plt.clf()
    fig = plt.figure(figsize=(2 * ncols, 2 * nrows))
    grids = fig.add_gridspec(nrows, ncols, width_ratios=width_ratios)
    axs = grids.subplots(sharex=False, sharey=True)

    for j, group_name in enumerate(group_names):
        results = defaultdict(list)
        for item in levels[3]:  # machine generated only
            id = item['id']
            assert id.startswith('gen/'), f'Wrong item: {id}'
            record = item['process_records'][-1]
            record = parse_record(record, 'Generate')
            group_value = record[group_name]
            results[group_value].append(item)

        # figure
        groups = list(results.keys())
        groups.sort()
        if j == 0:
            axs[j].set_xlim(-2.0, 12.0)
            colors = ['tab:blue', 'tab:purple', 'tab:green', 'tab:orange', 'tab:olive', 'tab:pink']
        else:
            axs[j].set_xlim(-2.0, 7.0)
            colors = ['tab:blue', 'tab:orange', 'tab:olive']

        assert len(colors) >= len(groups)
        colors = colors[:len(groups)]

        for group, color in zip(groups, colors):
            items = results[group]
            print(f'Group: {group_name} {group}, {len(items)} samples')
            xs = np.array([item['content_crit'] for item in items])
            ys = np.array([item['language_crit'] for item in items])
            confidence_ellipse(xs, ys, axs[j], 1.0, color=color, fill=False, lw=0.5)
            confidence_ellipse(xs, ys, axs[j], 0.01, color=color, fill=False, lw=1.0)

        axs[j].set_title(group_titles[group_name], fontsize=10)
        axs[j].set_axisbelow(True)
        axs[j].grid(axis='x', color='lightgrey', lw=0.2, linestyle='-')
        axs[j].grid(axis='y', color='lightgrey', lw=0.2, linestyle='-')
        axs[j].set_xlabel('Content ($C_2$)')

        legend = [Patch(color=color, label=group, fill=False) for group, color in zip(groups, colors)]
        axs[j].legend(handles=legend, loc='lower right', fontsize=5, ncol=1, handlelength=1.5)
        axs[j].set_xticks([0.0, 5.0])

    axs[0].set_ylabel('Expression ($E_2$)')
    plt.ylim(-2.0, 7.0)
    plt.yticks([0.0, 5.0])
    plt.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.subplots_adjust(bottom=0.35)
    plt.savefig(path.join(args.output_path, f'group_2d_distrib.pdf'))


def draw_content_language_barchart(args):
    # plot
    colors = ['tab:pink', 'tab:olive']
    hatchs = ['', '///']
    width = 0.33

    # plot
    nrows = 1
    ncols = 2
    plt.clf()
    plt.rcParams['hatch.color'] = 'tab:gray'
    fig = plt.figure(figsize=(4 + 1, 2 * nrows))
    grids = fig.add_gridspec(nrows, ncols, width_ratios=[3, 1.5])
    axs = grids.subplots(sharex=False, sharey=True)

    # chart 1
    # xticks = ['outline', 'simplify', 'list', 'replace', 'original']
    xticks = ['$C_1$', '$C_2$', '$E_1$', '$E_2$', '$T$']
    xs = [i for i in range(len(xticks))]
    yss = {
        'AI Content Detection Task': [0.7493, 0.7909, 0.6259, 0.6705, 0.6940],
        'AI Expression Detection Task': [0.5397, 0.5701, 0.6177, 0.7163, 0.7783]
    }
    tasks = list(yss.keys())

    for j, task in enumerate(yss):
        ys = np.array(yss[task])
        xs = np.array(xs)
        axs[0].bar(xs + j * width, ys, width=0.25, color=colors[j], hatch=hatchs[j], label=task)

    axs[0].set_axisbelow(True)
    axs[0].grid(axis='x', color='lightgrey', lw=0.2, linestyle='-')
    axs[0].grid(axis='y', color='lightgrey', lw=0.2, linestyle='-')
    axs[0].set_ylabel('AUROC')
    axs[0].set_xlabel('(a) Textual Features')
    axs[0].set_xticks(xs + width / 2, xticks)

    # chart 2
    xticks = ['4o', 'greedy', '3.5']
    xs = [i for i in range(len(xticks))]
    ys = [0.7909, 0.7797, 0.7193]

    axs[1].set_axisbelow(True)
    axs[1].grid(axis='x', color='lightgrey', lw=0.2, linestyle='-')
    axs[1].grid(axis='y', color='lightgrey', lw=0.2, linestyle='-')
    ys = np.array(ys)
    xs = np.array(xs) * 2 * width
    axs[1].bar(xs, ys, width=0.25, color=colors[0], label=tasks[0])
    axs[1].set_xlabel('(b) Settings for $C_2$')
    axs[1].set_xlim(-width, 5 * width)
    axs[1].set_xticks(xs, xticks)

    plt.ylim(0.3, 0.9)
    plt.yticks([0.3, 0.45, 0.6, 0.75, 0.9])
    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.subplots_adjust(left=0.13, bottom=0.28, right=0.99, top=0.83)
    fig.legend(labels=tasks, loc='upper center', fontsize=8, ncol=2, handlelength=1.5)
    plt.savefig(path.join(args.output_path, 'content_language_barchart.pdf'))


def draw_raid_dev_samples_linechart(args):
    # dev samples
    xs = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180]
    # binoculars
    ys0 = [0.8073, 0.8073, 0.8073, 0.8073, 0.8073, 0.8073, 0.8073, 0.8073, 0.8073, 0.8073]
    # CG (binoculars)
    ys1 = [0.8428, 0.8580, 0.8693, 0.8600, 0.8750, 0.8697, 0.8715, 0.8776, 0.8862, 0.8810]
    # CG (binoculars) with full dev set
    ys2 = [0.8861, 0.8861, 0.8861, 0.8861, 0.8861, 0.8861, 0.8861, 0.8861, 0.8861, 0.8861]
    # plot
    colors = ['tab:orange', 'tab:blue']
    labels = ['Binoculars', '$C_2$-$T$ (Binoculars)', 'Full dev set']

    # plot
    nrows = 1
    ncols = 1
    plt.clf()
    fig = plt.figure(figsize=(3 * ncols, 2 * nrows))
    grids = fig.add_gridspec(nrows, ncols)
    axs = grids.subplots(sharex=True, sharey=True)

    axs.set_axisbelow(True)
    axs.grid(axis='x', color='lightgrey', lw=0.2, linestyle='-')
    axs.grid(axis='y', color='lightgrey', lw=0.2, linestyle='-')
    axs.plot(xs, ys0, lw=0.75, color=colors[1], label=labels[0])
    axs.plot(xs, ys1, lw=0.75, color=colors[0], label=labels[1])
    axs.plot(xs, ys2, lw=0.5, color=colors[0], label=labels[2], linestyle='--')
    axs.set_ylabel('AUROC')
    axs.set_xlabel('Dev Samples')
    axs.legend(labels=labels, loc='upper right', fontsize=6.5, ncol=1, handlelength=1.5)

    plt.ylim(0.80, 0.95)
    plt.xticks([20, 60, 100, 140, 180])
    fig.subplots_adjust(left=0.20, bottom=0.22, right=0.98, top=0.95)
    plt.savefig(path.join(args.output_path, 'raid_dev_samples_linechart.pdf'))

def draw_raid_analysis_barchart(args):
    # attacks
    xticks = ['zero space', 'homoglyph', 'whitespace', 'insert paras', 'misspelling',
                  'alter spelling', 'article del', 'number', 'upper lower', 'paraphrase', 'synonym']
    attack = {
        'xticks': xticks,
        'xs': [i for i in range(len(xticks))],
        # binoculars
        'ys0': [0.6430, 0.6701, 0.8103, 0.8253, 0.8262, 0.8279, 0.8298, 0.8305, 0.9225, 0.7791, 0.8116],
        # C(binoculars)
        'ys1': [0.7920, 0.7835, 0.7850, 0.7632, 0.7833, 0.7856, 0.7833, 0.7801, 0.8506, 0.7593, 0.7446],
        # CG(binoculars)
        'ys2': [0.8053, 0.8235, 0.9283, 0.9215, 0.9433, 0.9243, 0.9424, 0.9006, 0.9654, 0.7935, 0.7795],
    }
    # decoding
    xticks = ['sampling', 'greedy']
    decoding = {
        'xticks': xticks,
        'xs': [i for i in range(len(xticks))],
        # binoculars
        'ys0': [0.7186, 0.8900],
        # C(binoculars)
        'ys1': [0.6936, 0.8460],
        # CG(binoculars)
        'ys2': [0.8438, 0.9292],
    }
    # repetition_penalty
    xticks = ['yes', 'no']
    penalty = {
        'xticks': xticks,
        'xs': [i for i in range(len(xticks))],
        # binoculars
        'ys0': [0.6119, 0.9319],
        # C(binoculars)
        'ys1': [0.6318, 0.8661],
        # CG(binoculars)
        'ys2': [0.8194, 0.9288],
    }
    # all in one
    results = [
        ('Adversarial Attacks', attack),
        ('Decoding', decoding),
        ('Rep-Penalty', penalty),
    ]
    width_ratio = [11, 2, 2]

    # plot
    colors = ['tab:blue', 'burlywood', 'tab:orange']
    labels = ['Binoculars', '$C_2$ (Binoculars)', '$C_2$-$T$ (Binoculars)']

    # plot
    nrows = 1
    ncols = 3
    plt.clf()
    fig = plt.figure(figsize=(2 * ncols, 2.5 * nrows))
    grids = fig.add_gridspec(nrows, ncols, width_ratios=width_ratio)
    axs = grids.subplots(sharex=False, sharey=True)

    width = 0.33
    for j, (title, result) in enumerate(results):
        xticks = result['xticks']
        xs = np.array(result['xs'])
        ys0 = result['ys0']
        ys1 = result['ys1']
        ys2 = result['ys2']
        axs[j].set_axisbelow(True)
        axs[j].grid(axis='x', color='lightgrey', lw=0.2, linestyle='-')
        axs[j].grid(axis='y', color='lightgrey', lw=0.2, linestyle='-')
        bar0 = axs[j].bar(xs, ys0, width=width, color=colors[0], label=labels[0])
        bar2 = axs[j].bar(xs + width, ys2, width=width, color=colors[2], label=labels[2])
        bar1 = axs[j].bar(xs + width, ys1, width=width, color=colors[1], label=labels[1])
        axs[j].set_xlim(-1.5 * width, len(xs) - 0.5 * width)
        axs[j].set_xticks(xs + width / 2, xticks)
        axs[j].set_xlabel(title)
        axs[j].tick_params(axis='x', labelrotation=60, labelsize=8, pad=2)

    axs[0].set_ylabel('AUROC')
    plt.ylim(0.5, 1.0)
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    fig.legend(handles=[bar0, bar1, bar2], loc='upper center', fontsize=8, ncol=3, handlelength=2)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.subplots_adjust(left=0.10, bottom=0.40, right=0.98, top=0.88)
    plt.savefig(path.join(args.output_path, 'raid_analysis_barchart.pdf'))

def draw_level2_analysis_barchart(args):
    # level2
    xticks = ['fully AI', 'diversify', 'mimic', 'humbot.ai', 'bypassgpt.ai', 'undetectable.ai', 'human editing']
    level2 = {
        'xticks': xticks,
        'xs': [i for i in range(len(xticks))],
        # fast_detect
        'ys0': [0.883, 0.587, 0.452, 0.446, 0.428, 0.445, 0.813],
        # C(fast_detect)
        'ys1': [0.818, 0.824, 0.675, 0.880, 0.803, 0.780, 0.862],
        # CG(fast_detect)
        'ys2': [0.901, 0.854, 0.734, 0.857, 0.815, 0.809, 0.834],
    }

    # plot
    colors = ['tab:blue', 'tab:olive', 'tab:orange']
    labels = ['Fast-Detect', '$C_2$ (Fast-Detect)', '$C_2$-$T$ (Fast-Detect)']

    # plot
    nrows = 1
    ncols = 1
    plt.clf()
    fig = plt.figure(figsize=(5 * ncols, 2 * nrows))
    grids = fig.add_gridspec(nrows, ncols)
    axs = grids.subplots(sharex=True, sharey=True)

    width = 0.25
    xs = np.array(level2['xs'])
    ys0 = level2['ys0']
    ys1 = level2['ys1']
    ys2 = level2['ys2']
    axs.set_axisbelow(True)
    axs.grid(axis='x', color='lightgrey', lw=0.2, linestyle='-')
    axs.grid(axis='y', color='lightgrey', lw=0.2, linestyle='-')
    bar0 = axs.bar(xs, ys0, width=width, color=colors[0], label=labels[0])
    bar1 = axs.bar(xs + width, ys1, width=width, color=colors[1], label=labels[1])
    bar2 = axs.bar(xs + width * 2, ys2, width=width, color=colors[2], label=labels[2])
    axs.set_xlim(-1.5 * width, len(xs) - 0.5 * width)
    axs.set_xticks(xs + width, xticks)
    axs.tick_params(axis='x', labelrotation=15, labelsize=8, pad=2)

    axs.set_ylabel('AUROC')
    plt.ylim(0.2, 1.0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    fig.legend(handles=[bar0, bar1, bar2], loc='upper center', fontsize=8, ncol=3, handlelength=2)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.subplots_adjust(left=0.11, bottom=0.20, right=0.99, top=0.84)
    plt.savefig(path.join(args.output_path, 'level2_analysis_barchart.pdf'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default="./exp_test/")
    parser.add_argument('--output_path', type=str, default="./exp_analysis")
    args = parser.parse_args()

    # draw_group_2d_distrib(args)
    # draw_domain_2d_distrib(args)
    # draw_domain_2d_distrib_refinement(args)
    # draw_domain_2d_distrib_humanize(args)
    draw_dataset_2d_distrib(args)
    # draw_domain_2d_distrib3(args)
    # draw_content_language_barchart(args)
    # draw_raid_dev_samples_linechart(args)
    # draw_raid_analysis_barchart(args)
    # draw_level2_analysis_barchart(args)
