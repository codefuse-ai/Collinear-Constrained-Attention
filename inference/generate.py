# coding=utf-8
# Copyright (c) 2023 Ant Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import time
import json
import torch
import random
import argparse
import jsonlines
import numpy as np
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(args):
    st = time.time()
    checkpoint = args.model_dir
    print('LOAD CKPT: {}'.format(checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
    tokenizer.add_special_tokens({'eos_token': "<|endoftext|>"})
    tokenizer.add_special_tokens({'pad_token': "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.float16)
    print('Model load spend: {:.4f}s'.format(time.time() - st))
    return tokenizer, model