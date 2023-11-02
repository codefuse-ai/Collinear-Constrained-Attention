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

import struct
import os
import numpy as np


dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}


class MMapIndexDataset:
    """
    描述GPT-Neox mmap实现方式获得的*.idx文件对应的数据集，即Tokenization索引数据集
    """

    # magic code
    _HDR_MAGIC = b"MMIDIDX\x00\x00"

    _VERSION = 1

    def __init__(self, index_dataset_file_path):
        """
        对于给定的GPT-Neox mmap实现方式的索引文件<index_dataset_file_path>生成对应的数据集描述
        """
        assert os.path.exists, (
            "给定的路径不存在"
            "请确保给定的.idx文件路径是存在的"
        )
        assert os.path.isfile, (
            "给定的路径不是一个文件"
            "请确保给定的是一个.idx文件的路径"
        )
        
        self.path = index_dataset_file_path
        
        with open(self.path, 'rb') as fb:
            magic = fb.read(9)
            assert magic == self._HDR_MAGIC, (
                "Magic Code与期望格式不匹配"
                "请确保提供的是GPT Neox MMAP方式生成的.idx文件"
            )

            version = struct.unpack('<Q', fb.read(8))
            assert version[0] == self._VERSION, (
                "提供的文件版本与期望不一致"
            )

            # 每个token用什么数据类型表达，例如，4字节INT32类型，2字节UINT16类型等
            dtype_code = struct.unpack('<B', fb.read(1))[0]
            self._dtype = dtypes[dtype_code]
            self._dtype_size = self._dtype().itemsize

            # 数据集样本数量
            self._len = struct.unpack('<Q', fb.read(8))[0]
            # doc index数量
            self._doc_index_count = struct.unpack('<Q', fb.read(8))[0]

            # 每个样本token数量，这里假设是SFT Padding模式
            self._seq_len = struct.unpack('<I', fb.read(4))[0]



class MMapIndexDatasetChecker:
    """
    对GPT Neox MMAP方式生成的Index数据集文件进行检查
    """

    def __init__(self, mmap_index_dataset_path):
        self.mmap_index_dataset_path = mmap_index_dataset_path
        self.mmap_index_dataset = MMapIndexDataset(self.mmap_index_dataset_path)


    def check(self, bin_bytes_size=None, seq_len=None):
        try:
            if seq_len and self.mmap_index_dataset._seq_len != seq_len:
                print(f"\033[1;31;47mIDX文件中单个样本长度({self.mmap_index_dataset._seq_len})与给定长度({seq_len})不一致\033[0m")
                return False
            print('bin_bytes_size', bin_bytes_size)
            if seq_len and bin_bytes_size and self.mmap_index_dataset._len * self.mmap_index_dataset._dtype_size * self.mmap_index_dataset._seq_len != bin_bytes_size:
                print("\033[1;31;47m依据IDX文件中的样本数量和单个样本大小计算出来的bin文件字节数与给定的不相同\033[0m")
                return False

            return True
        except:
            return False
