#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017    Ke Wang

""" This module contains several utility functions and classes that are
commonly used in every scripts.
https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/libs/common.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import subprocess

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def execute_command(command):
    """ Runs a job in the foreground and waits for it to complete; raises an
        exception if its return status is nonzero.  The command is executed in
        'shell' mode so 'command' can involve things like pipes.
        See also: get_command_stdout
    """
    p = subprocess.Popen(command, shell=True)
    p.communicate()
    if p.returncode is not 0:
        raise Exception("Command exited with status {0}: {1}".format(
                p.returncode, command))


def get_command_stdout(command, require_zero_status = True):
    """ Executes a command and returns its stdout output as a string. The
        command is executed with shell=True, so it may contain pipes and
        other shell constructs.
        If require_zero_stats is True, this function will raise an exception if
        the command has nonzero exit status. If False, it just prints a warning
        if the exit status is nonzero.
        See also: execute_command
    """
    p = subprocess.Popen(command, shell=True,
                         stdout=subprocess.PIPE)

    stdout = p.communicate()[0]
    if p.returncode is not 0:
        output = "Command exited with status {0}: {1}".format(
            p.returncode, command)
        if require_zero_status:
            raise Exception(output)
        else:
            logger.warning(output)
    return stdout if type(stdout) is str else stdout.decode()
