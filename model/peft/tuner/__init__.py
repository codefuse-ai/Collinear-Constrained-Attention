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

"""peft tuner methods interface."""

from peft.utils import PeftType
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING

from .adalora import AdaLoraConfig, AdaLoraModel
from .routelora import RouteLoraConfig, RouteLoraModel
from .unipelt import UniPELTConfig, UniPELTModel, PEUniPELTModel
from .pe_base_model import PEBaseModel
from .bitfit import PeftBitfitConfig, PEBitfitModel, PeftBitfitModel
from .roem import PeftROEMConfig, PEROEMModel, PeftROEMModel

# Register new ant peft methods
PeftType.ROUTELORA = "ROUTELORA"
PEFT_TYPE_TO_MODEL_MAPPING[PeftType.ROUTELORA] = RouteLoraModel
PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.ROUTELORA] = RouteLoraConfig

PeftType.UNIPELT = "UNIPELT"
PEFT_TYPE_TO_MODEL_MAPPING[PeftType.UNIPELT] = UniPELTModel
PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.UNIPELT] = UniPELTConfig

PeftType.ROEM = "ROEM"
PEFT_TYPE_TO_MODEL_MAPPING[PeftType.ROEM] = PeftROEMModel
PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.ROEM] = PeftROEMConfig

PeftType.BITFIT = "BITFIT"
PEFT_TYPE_TO_MODEL_MAPPING[PeftType.BITFIT] = PeftBitfitModel
PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.BITFIT] = PeftBitfitConfig