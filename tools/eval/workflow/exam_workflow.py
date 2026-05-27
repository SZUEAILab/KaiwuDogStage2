#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from tools.eval.workflow.eval_workflow import eval_workflow


def workflow(*kargs, **kwargs):
    eval_workflow.workflow(*kargs, **kwargs)
