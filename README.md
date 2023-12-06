<div align="center">
<h1>
  Collinear Constrained Attention
</h1>
</div>

<p align="center">
🤖 <a href="https://modelscope.cn/models/codefuse-ai/Collinear-Constrained-Attention/summary" target="_blank">ModelScope</a> 
  • 
📄 <a href="https://arxiv.org/abs/2309.08646" target="_blank">Paper</a>
</p>

<div align="center">

[![GitHub issues](https://img.shields.io/github/issues/codefuse-ai/Collinear-Constrained-Attention)](https://github.com/codefuse-ai/Collinear-Constrained-Attention/issues)
[![GitHub Repo stars](https://img.shields.io/github/stars/codefuse-ai/Collinear-Constrained-Attention?style=social)](https://github.com/codefuse-ai/Collinear-Constrained-Attention)

</div>

[comment]: <> ([<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Weights & Biases monitoring" height=20>]&#40;https://wandb.ai/eleutherai/neox&#41;)

This repository provides an implementation of [CoCA](https://arxiv.org/abs/2309.08646). This implementation is based on 2 transformer models in [Hugging Face]().

- [GPT-NeoX](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gpt_neox) which is an [EleutherAI](https://www.eleuther.ai)'s library for training large-scale language models on GPUs.
- [LLaMA](https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama) from Meta AI team.

We just point out those modifications which made to implement CoCA here. For more information about model training and inference, we recommend [transformers](https://github.com/huggingface/transformers).

For practicality, we enhanced CoCA's computational and spatial efficiency with [opt_einsum](https://github.com/dgasmith/opt_einsum), view this repository for more information.

![Model Structure](https://github.com/codefuse-ai/Collinear-Constrained-Attention/blob/master/assets/model.png "Model Structure")

![PPL Performance](https://github.com/codefuse-ai/Collinear-Constrained-Attention/blob/master/assets/PPL.png "PPL Performance") ![Passkey Performance](https://github.com/codefuse-ai/Collinear-Constrained-Attention/blob/master/assets/passkey.png "Passkey Performance")

[comment]: <> (<img src="https://github.com/codefuse-ai/Collinear-Constrained-Attention/blob/master/assets/PPL.png" width="210px">)

## 🚀 Quick Start

### 💻 Environment
Atorch is an optimized torch version by Ant Group, it's not available for opensource community yet. It will be opensource in near future. Before that, you may use origin torch version instead.

### 📂 Datasets
You can use raw data or tokenized data for training.

When using raw data, please ensure the data format as:
```json
{"content" : "It is a sentence for training."}
```
using `.jsonl` for saving data.

You can also use tokenized data saving in `.bin` via [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) tokenizer.
```bash
python ./data/tokenization/generate_dataset.py
```
notice to modify `input_dict`, `conver_type_list`, `output_name`, `seq_length` for your own dataset.

### 🏋️‍♂️ Training
You can train a model from scratch as follows:
```bash
bash ./train/run_coca.sh 32 1 8 2
```

- first parameter means `per gpu batch size`
- second parameter means `tensor parallel`(larger than 1 is not supported yet)
- third parameter means `data parallel`, equals to the number of GPUs
- last parameter means `train epochs`

If you want to load a pre-trained model, set `--pretrained_model_path $PRETRAINED_MODEL_PATH \`.

### 🧠 Inference
CoCA can be loaded using the `transformers` functionality:

```python
from model.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM, GPTNeoXConfig
from transformers import AutoTokenizer
from transformers import GenerationConfig

config = GPTNeoXConfig.from_pretrained(checkpoint)
config.is_decoder = True

# If you want to inference out of training length, 
# CoCA is compatible with NTK-aware scaled RoPE and performs much more better than original attention structure
rope_scaling= {"type": "dynamic", "factor": 4.0}
config.rope_scaling = rope_scaling

model = GPTNeoXForCausalLM.from_pretrained(checkpoint, 
                                           config=config, 
                                           device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
tokenizer.add_special_tokens({'eos_token': "<|endoftext|>"})
tokenizer.add_special_tokens({'pad_token': "<|pad|>"})
```

## 📝 Administrative Notes

### 📚 Citing CoCA

If you have found the CoCA library helpful in your work, you can cite this repository as

```bibtex
@misc{zhu2023cure,
      title={Cure the headache of Transformers via Collinear Constrained Attention}, 
      author={Shiyi Zhu and Jing Ye and Wei Jiang and Qi Zhang and Yifan Wu and Jianguo Li},
      year={2023},
      eprint={2309.08646},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### 📜 Licensing

This repository hosts code of CoCA project. Copyright (c) 2023, Ant Group. Licensed under the Apache License:

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

This repository is based off code written by EleutherAI that is licensed under the Apache License, Version 2.0. In accordance with the Apache License, all files that are modifications of code originally written by EleutherAI maintain a EleutherAI copyright header. When the EleutherAI code has been modified from its original version, that fact is noted in the copyright header. All derivative works of this repository must preserve these headers under the terms of the Apache License.

This repository is based off code written by Meta AI that is licensed under the Apache License, Version 2.0. In accordance with the Apache License, all files that are modifications of code originally written by Meta AI maintain a Meta AI copyright header. When the Meta AI code has been modified from its original version, that fact is noted in the copyright header. All derivative works of this repository must preserve these headers under the terms of the Apache License.

This repository is based off code written by NVIDIA that is licensed under the Apache License, Version 2.0. In accordance with the Apache License, all files that are modifications of code originally written by NVIDIA maintain a NVIDIA copyright header. All files that do not contain such a header are the exclusive copyright of EleutherAI. When the NVIDIA code has been modified from its original version, that fact is noted in the copyright header. All derivative works of this repository must preserve these headers under the terms of the Apache License.

This repository also contains code written by a number of other authors. Such contributions are marked and the relevant licensing is included where appropriate.

For full terms, see the `LICENSE` file. If you have any questions, comments, or concerns about licensing please email me at zhushiyi.zsy@antgroup.com.
