# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

#############################################
# copied and modified (from project SECOND) #
#############################################

from torchplus.train import learning_schedules_fastai as lsf


def build(lrs_config, total_step, optimizer):
    """Create lr scheduler based on config. note that
    lr_scheduler must accept a optimizer that has been restored.
  
    Args:
      lrs_config: A learning rate scheduler configuration
      total_step: Total training steps
      optimizer: An associated optimizer

    Returns:
      A learning rate scheduler.
  
    Raises:
      ValueError: when using an unsupported input data type.
    """
    lr_scheduler = _create_learning_rate_scheduler(lrs_config, total_step, optimizer)
    return lr_scheduler


def _create_learning_rate_scheduler(lrs_config, total_step, optimizer):
    """Create optimizer learning rate scheduler based on config.
  
    Args:
      lrs_config: A learning rate scheduler configuration
      total_step: Total training steps
      optimizer: An associated optimizer
  
    Returns:
      A learning rate scheduler.
  
    Raises:
      ValueError: when using an unsupported input data type.
    """
    lrs_type = lrs_config['scheme']
    lr_scheduler = None

    if lrs_type == 'one_cycle':
        lr_max = lrs_config['lr_max']
        moms = lrs_config['moms']
        div_factor = lrs_config['div_factor']
        pct_start = lrs_config['pct_start']
        lr_scheduler = lsf.OneCycle(
            optimizer, total_step, lr_max, moms, div_factor, pct_start)

    if lr_scheduler is None:
        raise ValueError('Learning_rate %s not supported.' % lrs_type)

    return lr_scheduler
