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

"""peft utils interface."""

from .config import PeftConfig, PetuningConfig

from .mapping import TRANSFORMERS_MODELS_ROME_LAYER_MODULES_MAPPING
from .mapping import TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
from .mapping import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from .mapping import TRANSFORMERS_MODELS_TO_LORA_LAGE_TARGET_MODULES_MAPPING
from .mapping import TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING
from .mapping import TRANSFORMERS_MODELS_TO_ROUTELORA_TARGET_MODULES_MAPPING
from .mapping import WEIGHTS_NAME, CONFIG_NAME
from .mapping import bloom_model_postprocess_past_key_value

from .others import get_peft_model_state_dict, set_peft_model_state_dict, _freeze_model, prepare_model_for_kbit_training