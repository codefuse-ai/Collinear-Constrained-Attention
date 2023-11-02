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

import MMapIndexDatasetParser
import MMapTokenIdsBinChecker
import argparse
import os


def process_dataset(index_dataset_path, input_ids_bin_path, loss_mask_bin_path, tokenizer_path, detokenization_output_path, seq_len, random_sampling_num):
    # 检查index dataset IDX文件
    mmap_index_dataset_checker = MMapIndexDatasetParser.MMapIndexDatasetChecker(index_dataset_path)
    if not os.path.exists(input_ids_bin_path) or not os.path.isfile(input_ids_bin_path):
        print(f"给定的input_ids.bin路径不存在或不是文件:{input_ids_bin_path}")
        return False

    check_result = mmap_index_dataset_checker.check(bin_bytes_size=os.path.getsize(input_ids_bin_path), seq_len=seq_len)
    if not check_result:
        print('!!!'*40)
        print(f"!\033[1;31;47mIDX检查未通过 {index_dataset_path}\033[0m")
        print('!!!'*40)
        return False

    # 检查input ids BIN文件
    mmap_token_ids_bin_checker = MMapTokenIdsBinChecker.MMapTokenIdsBinChecker(input_ids_bin_path=input_ids_bin_path, 
                                                                               loss_mask_bin_path=loss_mask_bin_path, 
                                                                               tokenizer_path=tokenizer_path,
                                                                               detokenize_output_path=detokenization_output_path,
                                                                               seq_len=seq_len,
                                                                               element_size=mmap_index_dataset_checker.mmap_index_dataset._dtype_size,
                                                                               dtype=mmap_index_dataset_checker.mmap_index_dataset._dtype,
                                                                               sample_total=mmap_index_dataset_checker.mmap_index_dataset._len,
                                                                               ramdom_sampling_num=random_sampling_num)
    
    check_result = mmap_token_ids_bin_checker.check()
    if not check_result:
        print('!!!'*40)
        print(f'!\033[1;31;47m【ERROR】数据集{loss_mask_bin_path}抽检未通过\033[0m')
        print('!!!'*40)
        return False
    else:
        print('###'*40)
        print(f'#\033[1;32;47m【OK】数据集{loss_mask_bin_path}抽检通过\033[0m')
        print('###'*40)
        return True



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('datasets_dir', help="GPT Neox MMAP方式生成的数据集目录路径，要求该路径下每个子目录对应一个数据集")
    parser.add_argument('tokenizer_path', help="词表文件路径")
    parser.add_argument('detokenization_output_dir', help='存储detokenization结果文件的目录路径')
    parser.add_argument('seq_len', help="每个样本token数量")
    parser.add_argument('--random_sampling_num', '-rsn', type=int,default=5, help='要随机抽样检查的数量，默认是100')

    args = parser.parse_args()

    for dir in os.listdir(args.datasets_dir):
        #if dir != 'codecompletion':
        #    continue
        for file in os.listdir(os.path.join(args.datasets_dir, dir)):
            if file.endswith('_input_ids.idx'):
                mmap_index_dataset_path = os.path.join(args.datasets_dir, dir, file)
            elif file.endswith('_input_ids.bin'):
                mmap_input_ids_bin_path = os.path.join(args.datasets_dir, dir, file)
            elif file.endswith('loss_mask.bin'):
                mmap_loss_mask_bin_path = os.path.join(args.datasets_dir, dir, file)

        detokenization_output_path = os.path.join(args.detokenization_output_dir, f"{dir}.txt")
        print(f'\n\n开始检查数据集{dir}....')
        process_dataset(mmap_index_dataset_path, mmap_input_ids_bin_path, mmap_loss_mask_bin_path, args.tokenizer_path, detokenization_output_path, int(args.seq_len), args.random_sampling_num)