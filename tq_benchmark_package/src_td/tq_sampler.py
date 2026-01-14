# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import random
from seqlen_balancing import get_seqlen_balanced_partitions

class BaseSampler(ABC):
    @abstractmethod
    def sample(self, indices: List[int], count: Optional[int] = None, **kwargs) -> List[int]:
        """
        采样逻辑接口
        :param indices: 输入的可选索引池
        :param count: 
            - None: 返回所有 indices (按策略排序后的列表)
            - 1: 返回单个选中的索引 (int)
            - n: 返回包含 n 个元素的列表 (List[int])
        """
        pass


class RandomSampler(BaseSampler):
    def __init__(self, seed: Optional[int] = None, replace: bool = False):
        """
        :param seed: 随机种子
        :param replace: 是否允许重复采样
        """
        # 使用私有的随机实例，避免干扰全局随机状态
        self.seed = seed
        self.replace = replace
        self._random = random.Random(seed)

    def sample(self, indices: List[int], count: Optional[int] = None, **kwargs) -> List[int]:
        if not indices:
            return []
        if count is None:
            return list(indices)
        if self.replace:
            return self._random.choices(indices, k=count)        
        else:
            actual_count = min(count, len(indices))
            return self._random.sample(indices, k=actual_count)


class VersionSampler(BaseSampler):
    def __init__(self, 
                 n_samples: int = 1, 
                 by_group: bool = False,
                 mode: str = "newest"):
        """
        :param n_samples: 每组样本的数量
        :param by_group: 是否按组逻辑处理
        :param mode: "newest" (降序, 取version最大) 或 "oldest" (升序, 取version最小)
        """
        self.n_samples = max(1, n_samples)
        self.by_group = by_group
        
        # 根据 mode 预设逻辑开关
        if mode == "newest":
            self.descending = True
            self.agg_func = max
        elif mode == "oldest":
            self.descending = False
            self.agg_func = min
        else:
            raise ValueError("mode must be 'newest' or 'oldest'")

    def sample(self, indices: List[int], count: Optional[int] = None, **kwargs) -> List[int]:
        # 从 kwargs 获取延迟绑定的 versions： 与传入 sample 的 indices 一一对应的版本号列表
        versions = kwargs.get("versions")
        if versions is None:
            raise ValueError("VersionSampler requires 'versions' passed via kwargs at runtime.")

        # --- 模式 A: 普通采样 ---
        if not self.by_group:
            paired = sorted(zip(indices, versions), 
                            key=lambda x: x[1], 
                            reverse=self.descending)
            sorted_indices = [p[0] for p in paired]
            return sorted_indices if count is None else sorted_indices[:count]

        # --- 模式 B: 分组模式 ---
        if len(indices) % self.n_samples != 0:
            raise ValueError(f"Total indices ({len(indices)}) must be a multiple of n_samples ({self.n_samples})")
        if count is not None and count % self.n_samples != 0:
            raise ValueError(f"count ({count}) must be a multiple of n_samples ({self.n_samples})")

        groups = []
        for i in range(0, len(indices), self.n_samples):
            group_indices = indices[i : i + self.n_samples]
            group_versions = versions[i : i + self.n_samples]
            
            # 动态选择 max 或 min 作为组特征
            groups.append({
                "data": group_indices,
                "v_feature": self.agg_func(group_versions)
            })

        # 根据 descending 动态决定组间排序顺序
        groups.sort(key=lambda x: x["v_feature"], reverse=self.descending)
        
        num_groups = len(groups) if count is None else min(count // self.n_samples, len(groups))
        result = [idx for g in groups[:num_groups] for idx in g["data"]]
        
        return result


# TODO
class SeqLenBalSampler(BaseSampler):
    pass


