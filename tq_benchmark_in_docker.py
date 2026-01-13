import os
import sys
import json
import time
import argparse
import subprocess
import threading
import datetime
import re
import tempfile
import shutil
import uuid

# ================= 配置区域 =================

# 镜像名称
DOCKER_IMAGE = "run_test"

# 所有待测 Config
ALL_CONFIGS = ["debug", "tiny", "small", "medium", "large", "xlarge", "huge"]

# 待测的分片数 (Shards)
SHARD_VARIANTS = [8]

# 分支测试配置
BRANCH_CONFIGS = [
    {
        "name": "optimized-v0.15",
        "branch": "feature/optimize-serialization-v0.15",
        "env_vars": {},
        "description": "优化后的序列化分支"
    },
    {
        "name": "main-no-zerocopy",
        "branch": "main",
        "env_vars": {"TQ_ZERO_COPY_SERIALIZATION": "false"},
        "description": "main 分支，未开启零拷贝"
    },
    {
        "name": "main-zerocopy",
        "branch": "main",
        "env_vars": {"TQ_ZERO_COPY_SERIALIZATION": "true"},
        "description": "main 分支，开启零拷贝"
    },
]

# 定义测试场景
SCENARIOS = [
    {
        "name": "TransferQueue",
        "cmd_args": [],
        "env_vars": {"PYTHONPATH": "."},
        "workdir": "transferqueue"
    }
]


# ================= Git 工具 =================

def prepare_branch_source(branch_name, build_base_dir):
    """
    使用 git archive 将指定分支的代码导出到 build_base_dir 下的独立目录。
    返回导出后的目录绝对路径。
    """
    # 简单的目录名清理，防止特殊字符
    safe_branch_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', branch_name)
    target_dir_name = f"{safe_branch_name}_{str(uuid.uuid4())[:8]}"
    target_path = os.path.join(build_base_dir, target_dir_name)
    
    os.makedirs(target_path, exist_ok=True)

    print(f"[Git] 正在导出分支 '{branch_name}' 到 {target_path} ...")
    
    # 使用 git archive 导出
    # git archive --format=tar {branch} | tar -x -C {dest}
    try:
        p1 = subprocess.Popen(["git", "archive", "--format=tar", branch_name], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["tar", "-x", "-C", target_path], stdin=p1.stdout)
        p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits
        p2.communicate()
        
        if p2.returncode != 0:
            raise subprocess.CalledProcessError(p2.returncode, "tar")
            
        print(f"[Git] ✓ 分支 '{branch_name}' 导出完成")
        return target_path
    
    except Exception as e:
        print(f"[Error] 导出分支 '{branch_name}' 失败: {e}")
        return None


# ================= 工具类 =================

class DockerMonitor:
    def __init__(self, container_name, interval=0.5):
        self.container_name = container_name
        self.interval = interval
        self.running = False
        self.stats = {
            "cpu_samples": [],
            "mem_samples_mb": [],
            "timestamps": []
        }
        self.thread = None

    def _parse_size(self, size_str):
        """将 docker stats 的内存字符串 (1.5GiB, 500MiB) 转为 MB"""
        units = {"B": 1 / (1024 ** 2), "KiB": 1 / 1024, "MiB": 1, "GiB": 1024, "TiB": 1024 ** 2}
        match = re.search(r"([0-9\.]+)([a-zA-Z]+)", size_str)
        if not match: return 0.0
        val, unit = float(match.group(1)), match.group(2)
        return val * units.get(unit, 1)

    def _parse_cpu(self, cpu_str):
        """将 150.5% 转为 float"""
        try:
            return float(cpu_str.replace("%", ""))
        except:
            return 0.0

    def _monitor_loop(self):
        cmd = [
            "docker", "stats", self.container_name,
            "--format", "{{.CPUPerc}}|{{.MemUsage}}",
            "--no-stream"
        ]

        while self.running:
            try:
                start_t = time.time()
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode == 0 and ret.stdout.strip():
                    line = ret.stdout.strip().split('\n')[0]
                    parts = line.split('|')
                    if len(parts) == 2:
                        cpu_raw, mem_raw = parts
                        mem_used_raw = mem_raw.split('/')[0].strip()

                        cpu_val = self._parse_cpu(cpu_raw)
                        mem_val = self._parse_size(mem_used_raw)

                        self.stats["cpu_samples"].append(cpu_val)
                        self.stats["mem_samples_mb"].append(mem_val)
                        self.stats["timestamps"].append(start_t)

                elapsed = time.time() - start_t
                sleep_time = max(0, self.interval - elapsed)
                time.sleep(sleep_time)
            except Exception:
                pass

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def get_summary(self):
        if not self.stats["cpu_samples"]:
            return {"max_cpu_percent": 0, "avg_cpu_percent": 0, "max_mem_mb": 0, "cpu_seconds": 0,
                    "duration_seconds": 0}

        timestamps = self.stats["timestamps"]
        cpus = self.stats["cpu_samples"]
        mems = self.stats["mem_samples_mb"]

        # 计算积分 (CPU Seconds)
        total_cpu_seconds = 0
        for i in range(1, len(cpus)):
            dt = timestamps[i] - timestamps[i - 1]
            # 前值积分
            total_cpu_seconds += (cpus[i] / 100.0) * dt

        return {
            "max_cpu_percent": max(cpus),
            "avg_cpu_percent": sum(cpus) / len(cpus),
            "max_mem_mb": max(mems),
            "avg_mem_mb": sum(mems) / len(mems),
            "cpu_seconds": total_cpu_seconds,
            "duration_seconds": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        }


def wait_until(target_time_str):
    """
    阻塞直到指定时间。如果时间已过，则等待到明天的该时间。
    格式: HH:MM
    """
    if not target_time_str:
        return

    now = datetime.datetime.now()
    try:
        target_time = datetime.datetime.strptime(target_time_str, "%H:%M").time()
    except ValueError:
        print(f"[Error] 时间格式错误: {target_time_str}，应为 HH:MM")
        sys.exit(1)

    target_dt = datetime.datetime.combine(now.date(), target_time)

    # 如果时间已过，推迟到明天
    if target_dt < now:
        target_dt += datetime.timedelta(days=1)

    print(f"[{now.strftime('%H:%M:%S')}] 计划于 {target_dt} 开始测试...")

    # 睡眠等待
    while datetime.datetime.now() < target_dt:
        delta = (target_dt - datetime.datetime.now()).total_seconds()
        if delta > 1:
            time.sleep(min(delta, 60))
        else:
            time.sleep(0.1)

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 时间到，开始执行！")


def get_cpuset_range(count, offset=2):
    """
    计算绑核范围字符串。
    策略：跳过前 offset 个核（避开 OS 干扰），向后取 count 个核。
    例如：count=8, offset=2 -> 返回 "2-9"
    """
    count = int(count)
    if count <= 0:
        return "0" # 异常兜底

    # 获取宿主机最大核数
    try:
        total_cpus = os.cpu_count() or 1
    except:
        total_cpus = 1
    
    # 如果请求数 >= 总核数，则使用全部核
    if count >= total_cpus:
        print(f"[Warn] 请求核数 {count} >= 物理核数 {total_cpus}，将使用所有核 0-{total_cpus-1}")
        return f"0-{total_cpus - 1}"

    start = offset
    end = start + count - 1

    # 如果范围超出了物理核边界，回退到从 0 开始
    if end >= total_cpus:
        print(f"[Warn] 绑核范围 {start}-{end} 超出物理上限 {total_cpus-1}，回退至从0开始")
        start = 0
        end = count - 1

    return f"{start}-{end}"


# ================= 主流程 =================
def run_single_benchmark(scenario, config_name, run_id, rounds, shards, cpu_limit, branch_config, source_path):
    """
    使用 cpuset-cpus 进行物理绑核，并同步配置 Ray 环境变量
    branch_config: 当前测试的分支配置字典
    source_path: 代码所在的主机路径 (build_dir/branch_xxx)
    """
    container_name = f"bench_{run_id}_{int(time.time())}"
    result_file_name = f"result_{container_name}.json"
    
    # 计算绑核范围字符串 (如 "2-9")
    cpuset_str = get_cpuset_range(cpu_limit)
    
    # 确保传入 Ray 的核数是整数
    ray_cpu_count = int(cpu_limit)

    # 构造 Docker 命令
    cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "--ipc=host",
        "--network=host",
        f"--cpuset-cpus={cpuset_str}",  # 物理绑核
        
        # --- Ray 专用配置 ---
        "-e", "RAY_DISABLE_DOCKER_CPU_WARNING=1", # 屏蔽警告
        "-e", f"RAY_NUM_CPUS={ray_cpu_count}",     # 强制 Ray 调度器只使用分配的核数
        # -------------------
        
        # 挂载 isolated source code
        "-v", f"{source_path}:/app",
        "-w", f"/app/{scenario['workdir']}"
    ]

    # 1. 应用场景级 Env
    for k, v in scenario.get("env_vars", {}).items():
        cmd.extend(["-e", f"{k}={v}"])

    # 2. 应用 Branch 级 Env (覆盖场景级)
    for k, v in branch_config.get("env_vars", {}).items():
        cmd.extend(["-e", f"{k}={v}"])

    cmd.append(DOCKER_IMAGE)

    # 容器内命令
    inner_cmd = [
                    "python", "put_benchmark.py",
                    "--config", config_name,
                    "--rounds", str(rounds),
                    "--shards", str(shards),
                    "--output", result_file_name
                ] + scenario["cmd_args"]

    cmd.extend(inner_cmd)

    print(f"\n[执行] {scenario['name']} | Branch: {branch_config['name']} | Config: {config_name} | CPUs: {cpu_limit} (Bind: {cpuset_str}) | Shards: {shards}")
    # print(f"      SRC: {source_path}") 
    # print(f"      CMD: {' '.join(cmd)}")

    monitor = DockerMonitor(container_name)

    try:
        process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        monitor.start()
        process.wait()
    finally:
        monitor.stop()

    # 读取结果 - 注意结果文件生成在 source_path 下
    monitor_stats = monitor.get_summary()
    perf_data = []
    
    # 结果文件路径在 source_path/workdir/result_xxx
    result_file_path = os.path.join(source_path, scenario['workdir'], result_file_name)
    
    if os.path.exists(result_file_path):
        try:
            with open(result_file_path, 'r') as f:
                perf_data = json.load(f)
            os.remove(result_file_path)
        except Exception as e:
            print(f"[Error] 读取结果 JSON 失败: {e}")
    else:
        print(f"[Error] 结果文件未生成: {result_file_path}")

    aggregated_record = {
        "scenario": scenario["name"],
        "branch_name": branch_config["name"],  # 记录测试配置名
        "git_branch": branch_config["branch"], # 记录实际 git 分支
        "config": config_name,
        "rounds": rounds,
        "shards": shards,
        "cpu_limit": cpu_limit,
        "cpuset_used": cpuset_str, # 记录实际绑定的物理核索引
        "timestamp": datetime.datetime.now().isoformat(),
        "resource_usage": monitor_stats,
        "performance_data": perf_data
    }

    print(f"      Stats -> CPU-Sec: {monitor_stats['cpu_seconds']:.2f}s | Max-Mem: {monitor_stats['max_mem_mb']:.0f}MB")
    return aggregated_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter_config", type=str, help="只跑特定 config，例如 'large'")
    parser.add_argument("--start_time", type=str, default=None, help="开始时间 HH:MM，未指定则立即开始")
    parser.add_argument("--rounds", type=int, default=20, help="每个 Config 的测试轮次 (默认 20)")
    # 新增参数: cpus
    parser.add_argument("--cpus", type=str, default="8", help="指定 CPU 核数列表，逗号分隔 (默认: '8')，例如: '8,24,48'")
    # 新增参数：debug
    parser.add_argument("--debug_dir", type=str, default=None, help="如果不为空，则不使用临时目录，而是保留代码到该目录方便调试")

    args = parser.parse_args()

    # 解析 CPU 列表
    try:
        cpu_variants = [float(x.strip()) for x in args.cpus.split(',')]
    except ValueError:
        print(f"[Error] CPUs 参数格式错误: {args.cpus}，应为数字列表，如 '8,16'")
        sys.exit(1)

    # 1. 调度等待
    wait_until(args.start_time)

    # 2. 准备代码构建环境
    if args.debug_dir:
        build_ctx = args.debug_dir
        os.makedirs(build_ctx, exist_ok=True)
        print(f"[Info] 使用固定构建目录: {build_ctx}")
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="tq_bench_ctx_")
        build_ctx = temp_dir_obj.name
        print(f"[Info] 创建临时构建目录: {build_ctx}")

    # 3. 准备任务
    final_results = []
    configs_to_run = [args.filter_config] if args.filter_config else ALL_CONFIGS

    # 计算总任务数
    total_tasks = len(SCENARIOS) * len(BRANCH_CONFIGS) * len(SHARD_VARIANTS) * len(cpu_variants) * len(configs_to_run)
    curr_task = 0

    print(f"开始测试，计划执行 {total_tasks} 个任务。CPU 组: {cpu_variants}")
    print(f"待测分支配置: {[b['name'] for b in BRANCH_CONFIGS]}")

    try:
        # cache 为 branch_name -> source_path
        branch_source_map = {}

        # 4. 执行循环
        for branch_cfg in BRANCH_CONFIGS:
            branch_name = branch_cfg["branch"]
            
            # --- 准备代码 ---
            if branch_name not in branch_source_map:
                src_path = prepare_branch_source(branch_name, build_ctx)
                if not src_path:
                    print(f"[Skip] 无法准备分支 {branch_name} 的代码，跳过")
                    continue
                branch_source_map[branch_name] = src_path
            
            current_source_path = branch_source_map[branch_name]

            for scenario in SCENARIOS:
                for shards in SHARD_VARIANTS:  # 分片数
                    for cpu_val in cpu_variants: # CPU 限制
                        for cfg in configs_to_run:
                            curr_task += 1
                            print(f"=== 进度 {curr_task}/{total_tasks} | Branch: {branch_cfg['name']} ===")

                            record = run_single_benchmark(
                                scenario, cfg, curr_task, 
                                args.rounds, shards, cpu_val, 
                                branch_cfg, current_source_path
                            )
                            final_results.append(record)

                            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n[Warn] 用户中断测试")
    finally:
        # 清理临时目录
        if not args.debug_dir and 'temp_dir_obj' in locals():
            print(f"[Info] 清理临时构建目录...")
            temp_dir_obj.cleanup()

    output_filename = f"final_benchmark_summary_{datetime.datetime.now().strftime('%Y%m%d')}.json"
    with open(output_filename, "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"\n✅ 所有测试完成。汇总结果 ({len(final_results)}项) 已保存至: {output_filename}")


if __name__ == "__main__":
    main()
