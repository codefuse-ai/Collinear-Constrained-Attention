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
import torch
import sys
sys.path.append("..")
from utils.common_utils import get_model_params_num
from transformers import (  # noqa: E402
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast
)
from .gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from .gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
from .gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast
from .llama.configuration_llama import LlamaConfig
from .llama.modeling_llama import LlamaForCausalLM
from .llama.tokenization_llama import LlamaTokenizer
from .llama.tokenization_llama_fast import LlamaTokenizerFast
# from .glm.modeling_glm import GLMForConditionalGeneration
# from .glm.configuration_glm import GLMConfig
# from .glm.tokenization_glm import GLMTokenizer

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from utils.common_utils import print_rank_0, is_old_version
from tokenizer import build_tokenizer
from tokenizer.tokenizer import HFTokenizer

import peft
from peft.tuners.lora import LoraLayer
from model.peft.utils import prepare_model_for_kbit_training
from peft import (  # noqa
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptEncoderReparameterizationType,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model
)
import model.peft.modeling_peft # noqa
from model.peft.tuner import AdaLoraConfig

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None
try:
    import bitsandbytes as bnb # noqa
except ImportError:
    bnb = None
from packaging import version


def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def setup_model(args, logger, use_cache=False):
    # Load pretrained model and tokenizer

    if args.pretrained_model_path:  # TODO: 实现from pretrained读tokenizer
        if args.model_type == 'gpt_neox':
            # if args.tokenizer_type:
            #     tokenizer = build_tokenizer(args)
            #     tokenizer.eod_token = "<|endoftext|>"
            #     tokenizer.pad_token = "<|pad|>"
            #     # tokenizer.sop_token = "<|endoftext|>"  # 适配multi task dataset
            #     # tokenizer.eop_token = "<|endoftext|>"
            #     tokenizer.eod_id = tokenizer.tokenize(tokenizer.eod_token)[0]
            #     tokenizer.pad_id = tokenizer.tokenize(tokenizer.pad_token)[0]
            # else:
            tokenizer = GPTNeoXTokenizerFast.from_pretrained(args.pretrained_model_path)
            # tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.vocab_file)
            tokenizer.eod_token = "<|endoftext|>"
            tokenizer.pad_token = "<|pad|>"
            tokenizer.sop_token = "<|endoftext|>"  # 适配multi task dataset
            tokenizer.eop_token = "<|endoftext|>"
            tokenizer.eod_id = tokenizer.convert_tokens_to_ids(tokenizer.eod_token)
            tokenizer.pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            
            print_rank_0(f'tokenizer {tokenizer.eod_token} id: {tokenizer.eod_id}')
            print_rank_0(f'tokenizer {tokenizer.pad_token} id: {tokenizer.pad_id}')

        elif args.model_type == 'llama':
            tokenizer = LlamaTokenizerFast.from_pretrained(args.pretrained_model_path)
            # tokenizer = AutoTokenizer.from_pretrained(
                            # args.pretrained_model_path,
                            # trust_remote_code=True,
                        # )
            tokenizer.eod_token = "</s>"
            tokenizer.eos_token = "</s>"
            tokenizer.bos_token = "<s>"
            tokenizer.pad_token = "[PAD]"
            tokenizer.unk_token = "<unk>"
            tokenizer.sop_token = "</s>"  # 适配multi task dataset
            tokenizer.eop_token = "</s>"
            tokenizer.eod_id = tokenizer.convert_tokens_to_ids(tokenizer.eod_token)
            tokenizer.eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
            tokenizer.bos_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
            tokenizer.pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            tokenizer.unk_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
            print_rank_0(f'tokenizer {tokenizer.eod_token} id: {tokenizer.eod_id}')
            print_rank_0(f'tokenizer {tokenizer.eos_token} id: {tokenizer.eos_id}')
            print_rank_0(f'tokenizer {tokenizer.bos_token} id: {tokenizer.bos_id}')
            print_rank_0(f'tokenizer {tokenizer.pad_token} id: {tokenizer.pad_id}')
            print_rank_0(f'tokenizer {tokenizer.unk_token} id: {tokenizer.unk_id}')
        elif args.model_type == 'glm':
            if is_old_version(args.pretrained_model_path):
                from .glm.tokenization_glm_deprecated import GLMChineseTokenizer
                tokenizer = GLMChineseTokenizer.from_pretrained(args.pretrained_model_path)
            else:
                tokenizer = GLMTokenizer.from_pretrained(args.pretrained_model_path)
    elif args.train_mode == 'sst':
        # tokenizer = build_tokenizer(args)
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.vocab_file)
        tokenizer.eod_token = "<|endoftext|>"
        tokenizer.pad_token = "<|pad|>"
        tokenizer.sop_token = "<|endoftext|>"  # 适配multi task dataset
        tokenizer.eop_token = "<|endoftext|>"
        tokenizer.eod_id = tokenizer.convert_tokens_to_ids(tokenizer.eod_token)
        tokenizer.pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        print_rank_0(f'tokenizer {tokenizer.eod_token} id: {tokenizer.eod_id}')
        print_rank_0(f'tokenizer {tokenizer.pad_token} id: {tokenizer.pad_id}')
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_path."
        )
    
    if args.model_type == 'gpt_neox':
        auto_config = GPTNeoXConfig
        auto_model_class = GPTNeoXForCausalLM
    elif args.model_type == 'llama':
        auto_config = LlamaConfig
        auto_model_class = LlamaForCausalLM
    elif args.model_type == 'glm':
        auto_config = GLMConfig
        auto_model_class = GLMForConditionalGeneration
    # else:
    #     auto_config = AutoConfig
    #     auto_model_class = AutoModelForCausalLM

    # with init_empty_weights_with_disk_offload(ignore_tie_weights=False):
    if args.pretrained_model_path:
        logger.info("Training model from checkpoint")
        config = auto_config.from_pretrained(args.pretrained_model_path)
        if args.peft_type != "qlora":
            # config = auto_config.from_pretrained(args.pretrained_model_path)
            # model = auto_model_class.from_pretrained(args.pretrained_model_path, trust_remote_code=True, device_map='auto').cuda()
            model = auto_model_class.from_pretrained(args.pretrained_model_path, trust_remote_code=True).cuda()
        else:
            if BitsAndBytesConfig is None:
                raise ImportError(
                    "To use qlora, please upgrade transformers to 4.30.1 by `pip install -U transformers==4.30.1`"
                )
            if bnb is None:
                raise ImportError("To use qlora, please install bitsandbytes by `pip install -U bitsandbytes==0.39.0`")
            try:
                import accelerate  # noqa
            except ImportError:
                raise ImportError("To use qlora, please install accelerate by `pip install -U accelerate==0.20.3`")
            peft_version = version.parse(peft.__version__)
            if peft_version < version.parse("0.4.0"):
                raise RuntimeError(f"Qlora needs peft>=0.4.0 but current peft version is {peft_version}")
            if args.bits not in [4, 8]:
                raise ValueError(f"Qlora only support 4 bits or 8 bits but got {args.bits} bits.")
            if args.bf16:
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
            if args.fp16:
                compute_dtype = torch.float16
            elif args.bf16:
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = torch.float32
            model = auto_model_class.from_pretrained(  # noqa
                args.pretrained_model_path,
                trust_remote_code=True,
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                torch_dtype=torch_dtype,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=args.bits == 4,
                    load_in_8bit=args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            )
    else:
        logger.info("Training model from scratch")
        if args.model_type == 'gpt_neox':
            config = GPTNeoXConfig.from_json_file(args.config_path + '/config.json')
            # model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)
            model = GPTNeoXForCausalLM._from_config(config)
        elif args.model_type == 'llama':
            config = LlamaConfig.from_json_file(args.config_path + '/config.json')
            # llama use xformers
            if args.use_xformers:
                config.use_xformers = True
            model = LlamaForCausalLM._from_config(config)
        elif args.model_type == 'glm':
            config = GLMConfig.from_json_file(args.config_path + '/config.json')
            model = GLMForConditionalGeneration._from_config(config)
        else:
            config = AutoConfig.from_json_file(args.config_path + '/config.json')
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    if args.model_type not in ['glm']:
        embedding_size = model.get_input_embeddings().weight.shape[0]
        print_rank_0('embedding size: ' + str(embedding_size))
        print_rank_0('vocab size: ' + str(tokenizer.vocab_size))
        if tokenizer.vocab_size > embedding_size:
            model.resize_token_embeddings(tokenizer.vocab_size)
        print_rank_0('resize embedding size: ' + str(model.get_input_embeddings().weight.shape[0]))
    
    print_rank_0(config)
    num_params = get_model_params_num(model)
    print_rank_0("num_params of this model:", num_params)
    args.total_model_param = num_params
    args.hidden_size = config.hidden_size
    args.num_hidden_layers = config.num_hidden_layers
    args.vocab_size = tokenizer.vocab_size
    print_rank_0(f'hidden size: {args.hidden_size}')
    print_rank_0(f'num hidden layers: {args.num_hidden_layers}')
    print_rank_0(f'vocab size: {args.vocab_size}')

    if args.peft_type:
        if args.peft_type in ['lora', 'qlora']:
            target_modules = None
            if args.peft_type == "qlora":
                model = prepare_model_for_kbit_training(model, False)
                target_modules = find_all_linear_names(args, model)
            else:
                target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            print_rank_0(f'target modules: {target_modules}')
            peft_config = LoraConfig(
                task_type=TaskType.ANT_CAUSAL_LM,
                inference_mode=False,
                r=96,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=target_modules,
            )
        logger.info(
            f"Load Peft {args.peft_type} model ......")
        if args.checkpoint_activations and args.peft_type in ["lora", "qlora"]:
            # Make Lora and gradient checkpointing compatible
            # https://github.com/huggingface/peft/issues/137
            model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        if args.peft_type == "qlora":
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    if args.bf16:
                        module = module.to(torch.bfloat16)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        if args.bf16 and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)
        logger.info(
            f"Reduce trainalbe params:\n")
        model.print_trainable_parameters()

    return model, config, tokenizer
