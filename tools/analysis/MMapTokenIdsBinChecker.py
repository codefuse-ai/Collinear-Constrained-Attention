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

import sys
sys.path.append("..")
sys.path.append("../..")

import os
import struct
from transformers import PreTrainedTokenizerFast
import random
import numpy as np
from model.gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast

tokenizer_vocab_file = '/mnt/user/bingchang/multisft/code/13b/code/v1-old/gpt-neox-2.0-sft-6b/tokenizer-ant-v5.json'

table = {ord(f): ord(t) for f, t in zip(
         u'，。！？：【】（）％＃＠＆１２３４５６７８９０',
         u',.!?:[]()%#@&1234567890')}


def punctuation_format(text):
    # Replace non-breaking space with space
    text = text.strip() + '\n'
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # change chinese punctuation to english ones
    text = text.translate(table)
    return text


def save_to_file(file_path, text):
    """
    写给定的<text>追加写入到<file_path>文件中
    """
    with open(file_path, 'a') as f:
        f.write(f'{text}')


def detokenize(input_ids, tokenizer, padding_token=None):
    """
    使用给定的<tokenizer>对给定的token id列表<input_ids>进行解码，如果给定了padding_token，则将padding部分移除
    """
    result = tokenizer.decode(input_ids)
    if padding_token and padding_token in result:
        result = result[:result.index(padding_token)]
    return result


def convert_bytes_to_elements(byte_data, dtype):
    """
    将字节数组转为对应数据类型数组
    """
    result = np.frombuffer(byte_data, dtype=dtype)
    return [x for x in result]


class MMapTokenIdsBinChecker:
    """
    检查GPT Neox MMAP方式生成的input_ids.bin文件
    """
    
    # 用于检查的随机采样数量
    _SAMPLING_NUM = 100

    _SEED = 202306192219

    _PADING_TOKEN = "<|pad|>"

    def __init__ (self, input_ids_bin_path:str, loss_mask_bin_path:str, tokenizer_path:str, detokenize_output_path:str, seq_len:int, element_size:int, dtype:np.dtype, sample_total:int, ramdom_sampling_num:int):
        assert os.path.exists(input_ids_bin_path), (
            "给定的input_ids.bin文件路径不存在"
            "请确保给定的路径是存在的"
        )
        assert os.path.isfile(input_ids_bin_path), (
            "给定的input_ids.bin文件不是一个文件"
            "请确保给定的是一个GPT Neox MMAP方式生成的input_ids.bin文件"
        )
        assert os.path.exists(loss_mask_bin_path) and os.path.isfile(loss_mask_bin_path), (
            "给定的loss_mask.bin文件路径不存在或者非文件"
            "请确保给定有效的loss_mask.bin文件路径"
        )
        assert os.path.exists(tokenizer_path) and os.path.isfile(tokenizer_path), (
            "给定的词表文件不存在或者不是一个文件"
            "请确保给定有效的词表文件路径"
        )

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

        self._SAMPLING_NUM = ramdom_sampling_num
        
        sampled_input_ids = []
        sampled_loss_masks = []
        sampled_indexes = []
        with open(input_ids_bin_path, 'rb') as fb, open(loss_mask_bin_path, 'rb') as f_lm:
            # 随机选取若干个样本，以进行detokenization验证和loss mask验证
            random.seed(self._SEED)
            random_indexes = random.sample(range(0, sample_total), min(self._SAMPLING_NUM, sample_total))
            print('随机采样样本索引为：', random_indexes)
            # 依次处理每个取样的样本
            for i in random_indexes:
                # 通过设定文件offset位置，读取取样的一个样本
                reset_pos = max(0, i-1)*seq_len*element_size
                fb.seek(reset_pos)
                data = fb.read(element_size*seq_len)
                # 将样本从byte序列转为int序列
                token_ids = convert_bytes_to_elements(data, dtype)
                sampled_input_ids.append(token_ids)
                text = detokenize(token_ids, self.tokenizer, self._PADING_TOKEN)
                # 保存到文件中供人工校验
                save_to_file(detokenize_output_path, '\n' + '[' + str(i) + ']' + '=*='*30 + '\n')
                save_to_file(detokenize_output_path, f"{text}\n")
                # 读取样本对应的loss_mask，用于检查是否只有<bot>部分的loss mask为1
                f_lm.seek(reset_pos)
                loss_mask_data = convert_bytes_to_elements(f_lm.read(seq_len*element_size), dtype)
                sampled_loss_masks.append(loss_mask_data)

                # my_text = punctuation_format(text)
                # my_tokenizer = GPTNeoXTokenizerFast.from_pretrained("/mnt/user/fuhang/checkpoints/neox-2.0-125m-sst-0614/hf_ckpt")
                # my_tokenizer.eod_token = "<|endoftext|>"
                # my_tokenizer.pad_token = "<|extratoken_1|>"
                # my_tokenizer.sop_token = "<|endoftext|>"  # 适配multi task dataset
                # my_tokenizer.eop_token = "<|endoftext|>"
                # my_tokenizer.eod_id = my_tokenizer.convert_tokens_to_ids(my_tokenizer.eod_token)
                # my_tokenizer.pad_id = my_tokenizer.convert_tokens_to_ids(my_tokenizer.pad_token)
                # my_token_ids = my_tokenizer(my_text)['input_ids']

                # sampled_indexes.append(i)
                # if i == 1926485:
                #     print('\n\n', '=*='*50, '\n', token_ids)
                #     print(i, text)

                # print('\n\n', '=*='*50, '\n', token_ids)
                print('\n\n', '=*='*50, '\n')
                print('token ids: ')
                print(token_ids)
                print('loss mask:')
                print(loss_mask_data)
                # print('\n\n', '=*='*50, '\n')
                # print('my token ids: ')
                # print(my_token_ids)
                # print(i)
                # print(text)

                
        self._sampled_input_ids = sampled_input_ids
        self._sampled_loss_masks = sampled_loss_masks
        self._sampled_indexes = sampled_indexes

    
    def check_loss_mask(self):
        """
        检查是否只有bot角色的内容对应的loss mask为1
        """
        for i in range(len(self._sampled_input_ids)):
            sampled_input_ids = self._sampled_input_ids[i]
            sampled_loss_mask = self._sampled_loss_masks[i]

            #print(i, 'input_ids', sampled_input_ids)
            #print('\n')
            #print(i, 'loss mask', len(sampled_loss_mask), sampled_loss_mask)
            #print('\n\n', '=*='*30)

            # 找出loss mask为1的片段
            pieces = []
            if 1 not in sampled_loss_mask:
                print(f'\033[1;31;47m【异常】样本{self._sampled_indexes[i]} loss mask全为0\033[0m')
                print('detokenizee', detokenize(sampled_input_ids, self.tokenizer, self._PADING_TOKEN))
                print('input_ids', sampled_input_ids)
                return False
            if 0 not in sampled_loss_mask:
                print(f'\033[1;31;47m【异常】样本{self._sampled_indexes[i]} loss mask全为1\033[0m')
                return False
    
            start_index = sampled_loss_mask.index(1)
            accul_index = 0
            while start_index > -1:
                #print('start_index', start_index)
                
                if 0 in sampled_loss_mask[start_index:]:
                    end_index = sampled_loss_mask[start_index:].index(0)
                    end_index = len(sampled_loss_mask)
                else:
                    print(self._sampled_loss_masks[i])
                    end_index = start_index + end_index
                #print('end_index', end_index)

                pieces.append((accul_index + start_index, accul_index + end_index))

                sampled_input_ids = sampled_input_ids[end_index:]
                sampled_loss_mask = sampled_loss_mask[end_index:]
                accul_index += end_index

                if 1 not in sampled_loss_mask:
                    break
                start_index = sampled_loss_mask.index(1)
            

            # 检查每段loss mask为1的数据对应的token ids之前的三个词是否是<|role_start|>bot<|role_end|>，最后一个词是否是<|end|>
            for piece in pieces:
                token_ids_piece = self._sampled_input_ids[i][max(0, piece[0]-3):piece[1]]
                text_piece = detokenize(token_ids_piece, self.tokenizer, self._PADING_TOKEN)
                if not text_piece.startswith("<|role_start|>bot<|role_end|>") or not text_piece.endswith('<|end|>'):
                    print(f'\033[1;31;47m【异常】样本{self._sampled_indexes[i]}存在loss mask为1但对应的不是bot片段:{text_piece}\033[0m')
                    return False
                
        return True
    

    def check(self):
        return self.check_loss_mask()
