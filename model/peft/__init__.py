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

"""peft models interface."""

from . import utils, tuner
from peft.mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
from peft.utils import TaskType
from .modeling_peft import AntPeftForCausalLM, AntPeftForEmbedding


SUPPORTED_PEFT_TYPES = ["prefix", "lora", "adalora", "bitfit", "roem", "unipelt", "prompt", "ptuning"]

# Register the Ant Causal Language Model
MODEL_TYPE_TO_PEFT_MODEL_MAPPING["ANT_CAUSAL_LM"] = AntPeftForCausalLM
TaskType.ANT_CAUSAL_LM = "ANT_CAUSAL_LM"

MODEL_TYPE_TO_PEFT_MODEL_MAPPING["ANT_EMBEDDING"] = AntPeftForEmbedding
TaskType.ANT_EMBEDDING = "ANT_EMBEDDING"
