# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import logging
from abc import ABC
from typing import Dict

import numpy as np
import torch


class Metric(ABC):
    def __init__(self, logger=None):
        self.metric = {}
        if logger is None:
            self.logger = logging.getLogger(__name__)

    def update(self, key="", value=None, cumulate=False):
        """
        只做参数更新
        key: str
        value: dict|list|tensor. when key is None, maybe value is a dict
        """
        self.logger.info(f'========= in metrics update start ==========')
        if cumulate:
            self.logger.info('cumulate1')
            if isinstance(value, Dict):
                self.logger.info(f'========= in metrics, value is instance dict ==========')
                if cumulate:
                    self.logger.info('cumulate2')
                    for key in value:
                        if key in self.metric:
                            if isinstance(self.metric[key], list):
                                self.metric[key].extend(value[key])
                            else:
                                self.metric[key] = value[key]
                        else:
                            self.metric[key] = [*value[key]]
                else:
                    self.logger.info('cumulate3')
                    self.metric.update(value)
            else:
                self.logger.info(f'========= value is not instance dict  ==========')
                if key in self.metric:
                    self.logger.info(f'========= key in metrics ==========')
                    if isinstance(self.metric[key], list):
                        self.metric[key].extend(value)
                    else:
                        self.metric[key] = value
                else:
                    self.logger.info(f'========= key not in metrics ==========')
                    self.metric[key] = [*value]
        else:
            self.logger.info('cumulate false')
            if isinstance(value, Dict):
                self.logger.info('cumulate4')
                self.metric.update(value)
            else:
                self.logger.info('cumulate5')
                self.metric[key] = value

        self.logger.info(f'========= in metrics update finish ==========')

    def compute_mean(self, key, value, axis=0):
        """
        计算并返回当前的指标的均值。
        """
        value_mean = None
        if isinstance(value, torch.Tensor):
            value_mean = torch.mean(value).detach().item()
        elif isinstance(value, np.ndarray):
            value_mean = np.mean(value, axis=axis)
        elif isinstance(value, list):
            # 过滤非数值元素
            filtered_data = [x for x in value if isinstance(x, (int, float))]
            value_mean = sum(filtered_data) / len(filtered_data)
        elif isinstance(value, tuple):
            value_mean = sum(value) / len(value)
        elif isinstance(value, dict):
            value_mean = sum(value.values()) / len(value)

        return value_mean

    def compute_max(self, key, value, axis=0):
        """
        计算并返回当前的指标的最大值。
        """
        value_max = None
        if isinstance(value, torch.Tensor):
            value_max = torch.max(value).detach().item()
        elif isinstance(value, np.ndarray):
            value_max = np.max(value, axis=axis)
        elif isinstance(value, list):
            # 过滤非数值元素
            filtered_data = [x for x in value if isinstance(x, (int, float))]
            value_max = max(filtered_data)
        elif isinstance(value, tuple):
            value_max = max(value)
        elif isinstance(value, dict):
            value_max = max(value.values())

        return value_max

    def compute_min(self, key, value, axis=0):
        """
        计算并返回当前的指标的最小值。
        """
        value_min = None
        if isinstance(value, torch.Tensor):
            value_min = torch.min(value).detach().item()
        elif isinstance(value, np.ndarray):
            value_min = np.min(value, axis=axis)
        elif isinstance(value, list):
            # 过滤非数值元素
            filtered_data = [x for x in value if isinstance(x, (int, float))]
            value_min = min(filtered_data)
        elif isinstance(value, tuple):
            value_min = min(value)
        elif isinstance(value, dict):
            value_min = min(value.values())

        return value_min

    def reset(self):
        """
        重置指标状态。
        """
        self.metric = {}
