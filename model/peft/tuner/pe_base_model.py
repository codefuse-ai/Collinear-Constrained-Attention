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

class PEBaseModel:
    """PEtuning的基类模型，定义了PEtuning模型都该有的方法"""

    def __init__():
        return

    def get_model(self):
        """对模型进行修改，冻结参数或者插入可训模块"""
        pass

    @classmethod
    def restore(self, model=None, path=None):
        """从path恢复PE模型

        Args:
            model (_type_, optional): 原始模型. Defaults to None.
            path (_type_, optional): 增量路径. Defaults to None.
        """
        pass
