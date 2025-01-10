# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/haotian-liu/LLaVA/


from unittest import mock
from llava.train.train import train
from llava.train.transformer_normalize_monkey_patch import patched_normalize

# 定义BatchSamplerShard的长度计算方法
def __len__(self):
    """返回BatchSamplerShard实例的元素数量。
    
    该方法通过调用batch_sampler的__len__方法来确定元素数量。
    """
    return len(self.batch_sampler)

# 定义BatchSamplerShard的迭代器方法
def __iter__(self):
    """返回BatchSamplerShard实例的迭代器。
    
    该方法通过调用batch_sampler的__iter__方法来获取迭代器。
    """
    return self.batch_sampler.__iter__()

# 程序入口
if __name__ == "__main__":
    # 使用mock.patch装饰器替换特定方法
    # 此处解释了为什么使用mock.patch：为了在训练过程中替换特定的函数实现，以测试或优化为目的
    with (
        mock.patch('transformers.image_processing_utils.normalize', new=patched_normalize),
        mock.patch('accelerate.data_loader.BatchSamplerShard.__len__', new=__len__),
        mock.patch('accelerate.data_loader.BatchSamplerShard.__iter__', new=__iter__)
        ):
            # 调用train函数开始训练过程
            train()