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
import sys
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from peft import PeftModelForCausalLM
# from transformers import BitsAndBytesConfig
# from peft import prepare_model_for_kbit_training

model_path='/output/checkpoint/mpt-fsdp-tp-bm-512-tp-1-dp-8-gpu-8-bin/checkpoint-320000'
lora_adapter='/output/checkpoint/mpt-fsdp-tp-bm-512-tp-1-dp-8-gpu-8-bin-lora/checkpoint-20000'
save_path='/output/checkpoint/mpt-fsdp-tp-bm-512-tp-1-dp-8-gpu-8-bin-lora/checkpoint-20000-merge'

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,  
    trust_remote_code=True,
    torch_dtype=torch.float16, 
    return_dict=True,
    device_map="auto"
)
print(base_model)
model_to_merge = PeftModelForCausalLM.from_pretrained(base_model, lora_adapter)
merged_model = model_to_merge.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(model_path)

merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Merge finised: {save_path} saved")