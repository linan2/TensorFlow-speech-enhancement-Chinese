#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import errno
import logging
import os
import sys
import warnings

sys.path.append(os.path.dirname(sys.path[0]))
import common as common


try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    g_plot = True
except ImportError:
    warnings.warn(
        """This script requires matplotlib and numpy.
        Please install them to generate plots.
        Proceeding with generation of tables.
        If you are on a cluster where you do not have admin rights you could
        try using virtualenv.""")
    g_plot = False


logger = logging.getLogger('utils')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Generating plots')


def get_args():
    parser = argparse.ArgumentParser(
        description="""Parses the training logs and generates a variety of
        plots.
        e.g. utils/generate_plots.py train_dnn.log exp/train_dnn.
        Look for the report.pdf in the output (report) directory.""")
    parser.add_argument("--adversarial",
                        default=False,
                        action="store_true",
                        help="Flag indicating parse adversarial model or not."
    )
    parser.add_argument("log_file",
                        # required=True,
                        help="name of log file."
    )
    parser.add_argument("output_dir",
                        # required=True,
                        help="report directory."
    )
    args = parser.parse_args()
    return args


g_plot_colors = ['red', 'blue', 'green', 'black', 'magenta', 'yellow', 'cyan']


def generate_loss_plots(adversarial, log_file, output_dir, plot):
    train_key = "TRAIN"
    valid_key = "CROSS"
    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise e
    logger.info("Generating loss plots")
    if adversarial:
        tr_losses = parse_loss_log_adversarial(log_file, train_key)
        cv_losses = parse_loss_log_adversarial(log_file, valid_key)
    else:
        tr_losses = parse_loss_log(log_file, train_key)
        cv_losses = parse_loss_log(log_file, valid_key)

    if plot:
        fig = plt.figure()
        plots = []

    for key_word in sorted(tr_losses.keys()):
        name = key_word
        tr_data = tr_losses[key_word]
        tr_data = np.array(tr_data)
        tr_iters = np.arange(1, tr_data.size+1)
        color_val = g_plot_colors[0]
        plot_handle, = plt.plot(tr_iters[:], tr_data[:], color=color_val,
                               linestyle="--", label="train")
        plots.append(plot_handle)
        color_val = g_plot_colors[1]
        cv_data = cv_losses[key_word]
        cv_data = np.array(cv_data)
        cv_iters = np.linspace(0, tr_data.size, num=cv_data.size, dtype=int)
        plot_handle, = plt.plot(cv_iters[:], cv_data[:], color=color_val,
                                label="valid")
        plots.append(plot_handle)
        if plot:
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            lgd = plt.legend(handles=plots, loc="upper right",
                             ncol=1, borderaxespad=0.)
            plt.grid(True)
            plt.title(key_word)
            figfile_name = "{0}/{1}.pdf".format(output_dir, key_word)
            plt.savefig(figfile_name, bbox_extra_artists=(lgd,),
                        bbox_inches="tight")
            fig = plt.figure()
            plots = []


def parse_loss_log_adversarial(log_file, key):
    """Parse adversarial model loss log file.
    train_loss_string format:
      1/821 (TRAIN AVG.LOSS): d_rl_loss = 0.32810, d_fk_loss = 0.32194, d_loss = 0.65004, g_adv_loss = 0.50822, g_mse_loss = 7.11048, g_l2_loss = 0.00000, g_loss = 36.06060
    valid_loss_string format:
      1/821 (CROSS AVG.LOSS): d_rl_loss = 0.34894, d_fk_loss = 0.17205, d_loss = 0.52099, g_adv_loss = 0.39619, g_mse_loss = 8.70989, g_l2_loss = 0.00000, g_loss = 43.94563
    """
    d_rl_losses = []
    d_fk_losses = []
    d_losses = []
    g_adv_losses = []
    g_mse_losses = []
    g_l2_losses = []
    g_losses = []
    key_word = ["d_rl_loss", "d_fk_loss", "d_loss",
                "g_adv_loss", "g_mse_loss", "g_l2_loss", "g_loss"]
    losses = {key_word[0]: d_rl_losses,
              key_word[1]: d_fk_losses,
              key_word[2]: d_losses,
              key_word[3]: g_adv_losses,
              key_word[4]: g_mse_losses,
              key_word[5]: g_l2_losses,
              key_word[6]: g_losses}

    train_loss_strings = common.get_command_stdout(
        "grep -e {} {}".format(key, log_file))
    for line in train_loss_strings.strip().split("\n"):
        line = line.split(",")
        assert len(line) == 7
        for i in range(7):
            sub_line = line[i].split()
            assert key_word[i] in sub_line
            losses[key_word[i]].append(float(sub_line[-1]))

    return losses


def parse_loss_log(log_file, key):
    """Parse loss log file.
    train_loss_string format:
      1/178 (TRAIN AVG.LOSS): g_mse_loss = 12.76571, g_l2_loss = 0.00000, g_loss = 12.76571, learning_rate= 1.200e-03
    valid_loss_string format:
      1/178 (CROSS AVG.LOSS): g_mse_loss = 9.99273, g_l2_loss = 0.00000, g_loss = 9.99273, time = 3.52 min
    """
    g_mse_losses = []
    g_l2_losses = []
    g_losses = []
    key_word = ["g_mse_loss", "g_l2_loss", "g_loss"]
    losses = {key_word[0]: g_mse_losses,
              key_word[1]: g_l2_losses,
              key_word[2]: g_losses}

    train_loss_strings = common.get_command_stdout(
        "grep -e {} {}".format(key, log_file))
    for line in train_loss_strings.strip().split("\n"):
        line = line.split(",")
        assert len(line) == 4
        for i in range(3):
            sub_line = line[i].split()
            assert key_word[i] in sub_line
            losses[key_word[i]].append(float(sub_line[-1]))

    return losses


def main():
    args = get_args()
    generate_loss_plots(args.adversarial, args.log_file, args.output_dir, g_plot)
    logger.info("Generating loss plots sucessfully.")


if __name__ == "__main__":
    main()
