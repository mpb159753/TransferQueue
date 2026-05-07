# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import random
from abc import ABC, abstractmethod
from typing import Optional


class BaseSampler(ABC):
    @abstractmethod
    def sample(self, indexes: list[int], count: int, **kwargs) -> list[int]:
        """
        Sampling logic interface
        :param indexes: Input pool of available indexes
        :param count: Number of indexes to return
        """
        pass


class RandomSampler(BaseSampler):
    def __init__(self, seed: int = 42, replace: bool = False):
        """
        :param seed: Random seed
        :param replace: Whether to allow sampling with replacement
        """
        # Use private random instance to avoid interfering with global random state
        self.seed = seed
        self.replace = replace
        self._random = random.Random(seed)

    def sample(self, indexes: list[int], count: int, **kwargs) -> list[int]:
        if not indexes or not count:
            raise ValueError("Sampler: Must provide indexes and count")
        if self.replace:
            return self._random.choices(indexes, k=count)
        else:
            actual_count = min(count, len(indexes))
            return self._random.sample(sorted(indexes), k=actual_count)


class VersionSampler(BaseSampler):
    def __init__(
        self, n_samples: int = 1, by_group: bool = False, selection_mode: str = "newest", group_ver: str = None
    ):
        """
        :param n_samples: Number of samples per group
        :param by_group: Whether to process by group logic
        :param selection_mode: "newest" (descending, select max version) or "oldest" (ascending, select min version)
        :param group_ver: "max" (group_version takes max within group), "min" (group_version takes min within group), or None (auto-infer from selection_mode)
        """
        self.n_samples = max(1, n_samples)
        self.by_group = by_group

        # Sorting direction
        if selection_mode == "newest":
            self.descending = True
        elif selection_mode == "oldest":
            self.descending = False
        else:
            raise ValueError("selection_mode must be 'newest' or 'oldest'")

        # Group version feature extraction method
        if group_ver is None:
            # Backward compatibility: auto-infer from selection_mode when not specified
            self.group_ver = max if selection_mode == "newest" else min
        elif group_ver == "max":
            self.group_ver = max
        elif group_ver == "min":
            self.group_ver = min
        else:
            raise ValueError("group_ver must be 'max', 'min', or None")

    def sample(self, indexes: list[int], count: int, versions: list[int], **kwargs) -> list[int]:
        if not indexes or not count:
            raise ValueError("Sampler: Must provide indexes and count")
        if versions is None:
            raise ValueError("VersionSampler requires 'versions' passed via kwargs at runtime.")

        # --- Mode A: Normal sampling ---
        if not self.by_group:
            paired = sorted(zip(indexes, versions, strict=False), key=lambda x: x[1], reverse=self.descending)
            sorted_indexes = [p[0] for p in paired]
            return sorted_indexes if count is None else sorted_indexes[:count]

        # --- Mode B: Grouping mode ---
        if len(indexes) % self.n_samples != 0:
            raise ValueError(f"Total indexes ({len(indexes)}) must be a multiple of n_samples ({self.n_samples})")
        if count is not None and count % self.n_samples != 0:
            raise ValueError(f"count ({count}) must be a multiple of n_samples ({self.n_samples})")

        groups = []
        for i in range(0, len(indexes), self.n_samples):
            group_indexes = indexes[i : i + self.n_samples]
            group_versions = versions[i : i + self.n_samples]

            # Dynamically select max or min as group feature
            groups.append({"data": group_indexes, "v_feature": self.group_ver(group_versions)})

        # Dynamically determine group sorting order based on descending
        groups.sort(key=lambda x: x["v_feature"], reverse=self.descending)

        num_groups = len(groups) if count is None else min(count // self.n_samples, len(groups))
        result = [idx for g in groups[:num_groups] for idx in g["data"]]

        return result


# TODO
class SeqLenBalSampler(BaseSampler):
    pass


class SameVersionSampler(BaseSampler):
    def __init__(self, target_version: int = None, selection_mode: str = "newest"):
        """
        Sampler for retrieving all indexes from the same version.

        Two scenarios:
        1. If target_version is specified: return indexes of that specific version
        2. If selection_mode is "newest" or "oldest": automatically select newest/oldest version and return its indexes

        Args:
            target_version: Specific version to sample. If None, use selection_mode to auto-select.
            selection_mode: "newest" (max version) or "oldest" (min version). Only used when target_version is None.

        Raises:
            ValueError: If selection_mode is invalid.
        """
        self.target_version = target_version
        self.selection_mode = selection_mode

        if target_version is None and selection_mode not in ("newest", "oldest"):
            raise ValueError(f"selection_mode must be 'newest' or 'oldest', but got '{selection_mode}'")

    def sample(self, indexes: list[int], count: int, versions: list[int], **kwargs) -> Optional[list[int]]:
        """
        Sample indexes from the same version.

        Args:
            indexes: Available indexes for sampling
            count: Maximum number of indexes to return
            versions: Version number for each index (aligned with indexes)
            **kwargs: Additional parameters (for compatibility)

        Returns:
            List[int]: Selected indexes from the same version
            None: If version doesn't exist or insufficient indexes
        """
        if not indexes or not count:
            raise ValueError("SameVersionSampler: Must provide indexes and count")
        if versions is None:
            raise ValueError("SameVersionSampler requires 'versions' passed via kwargs at runtime")
        if len(indexes) != len(versions):
            raise ValueError(
                f"SameVersionSampler: indexes length ({len(indexes)}) != versions length ({len(versions)})"
            )

        # Determine which version to select
        if self.target_version is not None:
            selected_version = self.target_version
        else:
            if self.selection_mode == "newest":
                selected_version = max(versions)
            elif self.selection_mode == "oldest":
                selected_version = min(versions)
            else:
                raise ValueError(f"selection_mode must be 'newest' or 'oldest', but got '{self.selection_mode}'")

        # Find all indexes of the selected version
        version_indexes = [idx for idx, ver in zip(indexes, versions, strict=False) if ver == selected_version]

        # Handle edge cases
        if not version_indexes:
            return None  # Version not found

        if len(version_indexes) < count:
            return None  # Insufficient indexes

        return version_indexes[:count]
