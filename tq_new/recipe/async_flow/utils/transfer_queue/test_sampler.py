@pytest.mark.xdist_group(name="new_test")
def test_same_version_sampler_scenario_a(topic_basic):
    """测试 SameVersionSampler 指定 target_version (场景A)"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    # 准备测试数据
    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    # 添加数据并设置版本
    allocated = tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    # 设置版本: version 0 -> [0,1], version 1 -> [2,3], version 2 -> [4,5]
    tq_mgr = ray.get_actor("TransferQueueManager")
    ray.get(tq_mgr.record_versions.remote(topic, 0, allocated[0:n_samples]))
    ray.get(tq_mgr.record_versions.remote(topic, 1, allocated[n_samples : 2 * n_samples]))
    ray.get(tq_mgr.record_versions.remote(topic, 2, allocated[2 * n_samples :]))

    # 场景A: 请求版本 1
    sampler = SameVersionSampler(target_version=1)
    data, idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt"],
        experience_count=2,  # 版本 1 恰好有 2 个索引 [2, 3]
        indexes=None,
        get_n_samples=False,
        topic=topic,
        sampler_func=sampler,
    )

    assert data is not None
    assert len(idxs) == 2
    assert set(idxs) == {2, 3}

    # 请求超过可用数量
    data, idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt"],
        experience_count=5,  # 版本 1 只有 2 个索引
        indexes=None,
        get_n_samples=False,
        topic=topic,
        sampler_func=SameVersionSampler(target_version=1),
    )
    # 应返回 None (索引不足)
    assert data is None or len(idxs) == 0


@pytest.mark.xdist_group(name="new_test")
def test_same_version_sampler_scenario_b_newest(topic_basic):
    """测试 SameVersionSampler 自动选择最新版本 (场景B)"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    # 准备测试数据
    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    allocated = tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    # 设置版本: version 0 -> [0,1], version 1 -> [2,3], version 2 -> [4,5]
    tq_mgr = ray.get_actor("TransferQueueManager")
    ray.get(tq_mgr.record_versions.remote(topic, 0, allocated[0:n_samples]))
    ray.get(tq_mgr.record_versions.remote(topic, 1, allocated[n_samples : 2 * n_samples]))
    ray.get(tq_mgr.record_versions.remote(topic, 2, allocated[2 * n_samples :]))

    # 场景B: 自动选择最新版本
    sampler = SameVersionSampler(selection_mode="newest")
    data, idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt"],
        experience_count=2,  # 最新版本 2 有 2 个索引 [4, 5]
        indexes=None,
        get_n_samples=False,
        topic=topic,
        sampler_func=sampler,
    )

    assert data is not None
    assert len(idxs) == 2
    assert set(idxs) == {4, 5}


@pytest.mark.xdist_group(name="new_test")
def test_same_version_sampler_scenario_b_oldest(topic_basic):
    """测试 SameVersionSampler 自动选择最旧版本 (场景B)"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    # 准备测试数据
    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    allocated = tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    # 设置版本: version 0 -> [0,1], version 1 -> [2,3], version 2 -> [4,5]
    tq_mgr = ray.get_actor("TransferQueueManager")
    ray.get(tq_mgr.record_versions.remote(topic, 0, allocated[0:n_samples]))
    ray.get(tq_mgr.record_versions.remote(topic, 1, allocated[n_samples : 2 * n_samples]))
    ray.get(tq_mgr.record_versions.remote(topic, 2, allocated[2 * n_samples :]))

    # 场景B: 自动选择最旧版本
    sampler = SameVersionSampler(selection_mode="oldest")
    data, idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt"],
        experience_count=2,  # 最旧版本 0 有 2 个索引 [0, 1]
        indexes=None,
        get_n_samples=False,
        topic=topic,
        sampler_func=sampler,
    )

    assert data is not None
    assert len(idxs) == 2
    assert set(idxs) == {0, 1}


@pytest.mark.xdist_group(name="new_test")
def test_same_version_sampler_version_not_found(topic_basic):
    """测试 SameVersionSampler 请求不存在的版本"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    # 准备测试数据
    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    # 设置版本 0, 1, 2
    tq_mgr = ray.get_actor("TransferQueueManager")
    allocated = list(range(6))
    ray.get(tq_mgr.record_versions.remote(topic, 0, allocated[0:n_samples]))
    ray.get(tq_mgr.record_versions.remote(topic, 1, allocated[n_samples : 2 * n_samples]))
    ray.get(tq_mgr.record_versions.remote(topic, 2, allocated[2 * n_samples :]))

    # 请求不存在的版本 99
    sampler = SameVersionSampler(target_version=99)
    data, idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt"],
        experience_count=2,
        indexes=None,
        get_n_samples=False,
        topic=topic,
        sampler_func=sampler,
    )

    # 应返回 None (版本未找到)
    assert data is None or len(idxs) == 0


@pytest.mark.xdist_group(name="new_test")
def test_same_version_sampler_basic_direct():
    """直接测试 SameVersionSampler 的基本功能 (不依赖 Ray)"""
    indexes = [0, 1, 2, 3, 4, 5]
    versions = [0, 0, 1, 1, 2, 2]

    # 场景A: 指定版本 1
    sampler = SameVersionSampler(target_version=1)
    result = sampler.sample(indexes, count=2, versions=versions)
    assert result == [2, 3]

    # 场景A: 索引不足
    result = sampler.sample(indexes, count=5, versions=versions)
    assert result is None

    # 场景A: 版本不存在
    sampler = SameVersionSampler(target_version=99)
    result = sampler.sample(indexes, count=2, versions=versions)
    assert result is None

    # 场景B: 自动选择最新版本
    sampler = SameVersionSampler(selection_mode="newest")
    result = sampler.sample(indexes, count=2, versions=versions)
    assert result == [4, 5]

    # 场景B: 自动选择最旧版本
    sampler = SameVersionSampler(selection_mode="oldest")
    result = sampler.sample(indexes, count=2, versions=versions)
    assert result == [0, 1]
