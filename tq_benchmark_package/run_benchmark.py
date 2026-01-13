import os
import sys
import json
import time
import argparse
import subprocess
import threading
import datetime
import re

DOCKER_IMAGE = "run_test"
ALL_CONFIGS = ["debug", "tiny", "small", "medium", "large", "xlarge", "huge"]
SHARD_VARIANTS = [8]

BRANCH_CONFIGS = [
    {"name": "optimized-v0.15", "path": "src_optimized", "env_vars": {}, "description": "Optimized"},
    {"name": "main-no-zerocopy", "path": "src_main", "env_vars": {"TQ_ZERO_COPY_SERIALIZATION": "false"}, "description": "No ZeroCopy"},
    {"name": "main-zerocopy", "path": "src_main", "env_vars": {"TQ_ZERO_COPY_SERIALIZATION": "true"}, "description": "ZeroCopy"},
]
SCENARIOS = [{"name": "TransferQueue", "cmd_args": [], "env_vars": {"PYTHONPATH": "."}, "workdir": "."}]

class DockerMonitor:
    def __init__(self, container_name, interval=0.5):
        self.container_name = container_name
        self.interval = interval
        self.running = False
        self.stats = {"cpu_samples": [], "mem_samples_mb": [], "timestamps": []}
        self.thread = None
        
    def _parse_size(self, size_str):
        units = {"B": 1 / (1024 ** 2), "KiB": 1 / 1024, "MiB": 1, "GiB": 1024, "TiB": 1024 ** 2}
        match = re.search(r"([0-9\.]+)([a-zA-Z]+)", size_str)
        if not match: return 0.0
        val, unit = float(match.group(1)), match.group(2)
        return val * units.get(unit, 1)

    def _parse_cpu(self, cpu_str):
        try: return float(cpu_str.replace("%", ""))
        except: return 0.0

    def _monitor_loop(self):
        cmd = ["docker", "stats", self.container_name, "--format", "{{.CPUPerc}}|{{.MemUsage}}", "--no-stream"]
        while self.running:
            try:
                start_t = time.time()
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode == 0 and ret.stdout.strip():
                    line = ret.stdout.strip().split('\n')[0]
                    parts = line.split('|')
                    if len(parts) == 2:
                        self.stats["cpu_samples"].append(self._parse_cpu(parts[0]))
                        self.stats["mem_samples_mb"].append(self._parse_size(parts[1].split('/')[0].strip()))
                        self.stats["timestamps"].append(start_t)
                time.sleep(max(0, self.interval - (time.time() - start_t)))
            except: pass
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()
    def stop(self):
        self.running = False
        if self.thread: self.thread.join()
    def get_summary(self):
        if not self.stats["cpu_samples"]: return {"cpu_seconds": 0, "max_mem_mb": 0}
        cpus, mems, timestamps = self.stats["cpu_samples"], self.stats["mem_samples_mb"], self.stats["timestamps"]
        total_cpu_seconds = sum((cpus[i] / 100.0) * (timestamps[i] - timestamps[i - 1]) for i in range(1, len(cpus)))
        return {"max_cpu_percent": max(cpus), "avg_cpu_percent": sum(cpus)/len(cpus), "max_mem_mb": max(mems), "cpu_seconds": total_cpu_seconds}

def wait_until(target_time_str):
    if not target_time_str: return
    now = datetime.datetime.now()
    try: target_time = datetime.datetime.strptime(target_time_str, "%H:%M").time()
    except: return
    target_dt = datetime.datetime.combine(now.date(), target_time)
    if target_dt < now: target_dt += datetime.timedelta(days=1)
    print(f"Scheduled for {target_dt}")
    while datetime.datetime.now() < target_dt: time.sleep(1)

def get_cpuset_range(count, offset=2):
    count = int(count)
    if count <= 0: return "0"
    try: total_cpus = os.cpu_count() or 1
    except: total_cpus = 1
    if count >= total_cpus: return f"0-{total_cpus - 1}"
    start = offset
    end = start + count - 1
    if end >= total_cpus: start = 0; end = count - 1
    return f"{start}-{end}"

def run_single_benchmark(scenario, config_name, run_id, rounds, shards, cpu_limit, branch_config):
    container_name = f"bench_{run_id}_{int(time.time())}"
    source_path = os.path.abspath(branch_config["path"])
    cpuset_str = get_cpuset_range(cpu_limit)
    cmd = [
        "docker", "run", "--rm", "--name", container_name, "--ipc=host", "--network=host",
        f"--cpuset-cpus={cpuset_str}",
        "-e", "RAY_DISABLE_DOCKER_CPU_WARNING=1",
        "-e", f"RAY_NUM_CPUS={int(cpu_limit)}",
        "-v", f"{source_path}:/app", "-w", "/app"
    ]
    for k, v in branch_config.get("env_vars", {}).items(): cmd.extend(["-e", f"{k}={v}"])
    cmd.append(DOCKER_IMAGE)
    cmd.extend(["python", "put_benchmark.py", "--config", config_name, "--rounds", str(rounds), "--shards", str(shards), "--output", "res.json"])
    
    print(f"Running {branch_config['name']}...")
    monitor = DockerMonitor(container_name)
    try:
        p = subprocess.Popen(cmd)
        monitor.start()
        p.wait()
    finally:
        monitor.stop()
    print(monitor.get_summary())
    return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter_config", type=str)
    parser.add_argument("--start_time", type=str)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--cpus", type=int, default=30, help="Number of CPUs to allocate for Docker container")
    parser.add_argument("--shards", type=int, default=8, help="Number of storage shards")
    args = parser.parse_args()
    wait_until(args.start_time)
    
    for branch_cfg in BRANCH_CONFIGS:
        if os.path.exists(branch_cfg["path"]):
            for config in [args.filter_config] if args.filter_config else ALL_CONFIGS:
                run_single_benchmark(SCENARIOS[0], config, 0, args.rounds, args.shards, args.cpus, branch_cfg)

if __name__ == "__main__":
    main()
