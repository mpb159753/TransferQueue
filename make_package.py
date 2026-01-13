import os
import subprocess
import shutil
import zipfile

# Configuration
PACKAGE_NAME = "tq_benchmark_package"
BRANCH_MAP = {
    "src_main": "main",
    "src_optimized": "feature/optimize-serialization-v0.15"
}

# ================= Container Internal Orchestrator =================
# Behaves like run_profile.sh but in Python
CONTAINER_PROFILER_SCRIPT = r'''import os
import sys
import time
import subprocess
import signal
import glob

# Try to import psutil, but fallback to subprocess if missing (likely missing in minimal docker)
# But standard docker image might not have psutil. We will use subprocess ps -ef.

SAMPLING_RATE = 20
OUTPUT_DIR = f"profile_results_{int(time.time())}"

def wait_for_flag(flag_name, main_pid):
    print(f"[Orchestrator] Waiting for {flag_name}...")
    while not os.path.exists(flag_name):
        # Check if main process is still alive using os.kill
        try:
            os.kill(main_pid, 0)
        except OSError:
            print("[Error] Python script exited unexpectedly.")
            sys.exit(1)
        time.sleep(0.1)
    if os.path.exists(flag_name):
        try: os.remove(flag_name)
        except: pass

def send_signal(flag_name):
    print(f"[Orchestrator] Sending signal {flag_name}...")
    with open(flag_name, 'w') as f:
        f.write('1')

def start_profilers(pid_list, suffix):
    procs = []
    print(f"[Orchestrator] Starting py-spy for {suffix} on PIDs: {pid_list}")
    for i, pid in enumerate(pid_list):
        out_file = f"{OUTPUT_DIR}/profile_{suffix}_{pid}.svg"
        # py-spy record --idle -r 20 -o file --pid pid
        # We assume py-spy is in path or installed.
        cmd = ["py-spy", "record", "--idle", "-r", str(SAMPLING_RATE), "-o", out_file, "--pid", str(pid)]
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            procs.append(p)
        except Exception as e:
            print(f"[Error] Failed to start py-spy: {e}")
    
    # Wait a bit to ensure they attach
    time.sleep(1.5)
    return procs

def stop_profilers(procs):
    print("[Orchestrator] Stopping profilers...")
    for p in procs:
        p.send_signal(signal.SIGINT)
    
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()

def main():
    if len(sys.argv) < 2:
        print("Usage: python container_profiler.py <cmd_to_run...>")
        sys.exit(1)
    
    cmd = sys.argv[1:]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Clean flags
    for f in glob.glob("*.flag"): 
        try: os.remove(f)
        except: pass

    print(f"[Orchestrator] Starting benchmark: {' '.join(cmd)}")
    bench_proc = subprocess.Popen(cmd)
    main_pid = bench_proc.pid
    
    try:
        # Wait for init
        wait_for_flag("init_ready.flag", main_pid)
        
        # Detect Ray Processes
        def find_pids(pattern):
            try:
                # ps -ef 
                out = subprocess.check_output(["ps", "-ef"]).decode()
                pids = []
                for line in out.splitlines():
                    if pattern in line and "grep" not in line:
                        parts = line.split()
                        if parts: pids.append(int(parts[1]))
                return pids
            except: return []

        storage_pids = find_pids("SimpleStorageUnit")
        controller_pids = find_pids("TransferQueueController")
        
        target_pids = [main_pid] + storage_pids[:2] + controller_pids
        print(f"[Orchestrator] Target PIDs: {target_pids}")

        # --- PUT Phase ---
        spies = start_profilers(target_pids, "PUT")
        send_signal("put_start.flag")
        
        wait_for_flag("put_done.flag", main_pid)
        stop_profilers(spies)
        
        # --- GET Phase ---
        # Note: put_benchmark might do META then GET. 
        # Our injection sends put_done, then waits for get_prepare (from orch), then sends get_ready.
        
        send_signal("get_prepare.flag")
        wait_for_flag("get_ready.flag", main_pid)
        
        spies = start_profilers(target_pids, "GET")
        send_signal("get_start.flag")
        
        bench_proc.wait()
        stop_profilers(spies)
        
    except Exception as e:
        print(f"[Orchestrator] Error: {e}")
        try: bench_proc.kill()
        except: pass
    
    print(f"[Orchestrator] Done. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
'''

# ================= Host Profiling Runner =================
HOST_PROFILING_SCRIPT = r'''import os
import sys
import argparse
import subprocess
import time
import shutil

# This script runs the docker container with py-spy capability
DOCKER_IMAGE = "run_test"

BRANCH_CONFIGS = [
    {"name": "optimized-v0.15", "path": "src_optimized", "env": {}},
    {"name": "main-no-zerocopy", "path": "src_main", "env": {"TQ_ZERO_COPY_SERIALIZATION": "false"}},
    {"name": "main-zerocopy", "path": "src_main", "env": {"TQ_ZERO_COPY_SERIALIZATION": "true"}},
]

def run_profiling(config_name, round_count, branch_cfg, shards=8, cpu_limit=8):
    source_path = os.path.abspath(branch_cfg["path"])
    timestamp = int(time.time())
    container_name = f"profile_{branch_cfg['name']}_{timestamp}"
    
    # We must start 'container_profiler.py' inside via docker
    # Benchmark cmd: python put_benchmark.py ...
    bench_cmd = [
        "python", "put_benchmark.py", 
        "--config", config_name, 
        "--rounds", str(round_count), 
        "--shards", str(shards),
        "--output", f"res_{timestamp}.json"
    ]
    
    # Wrapper cmd
    wrapper_cmd = ["python", "container_profiler.py"] + bench_cmd
    
    cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "--cap-add=SYS_PTRACE",
        "--security-opt", "seccomp=unconfined", # Needed for some py-spy versions
        "--ipc=host", "--network=host",
        "-e", f"RAY_NUM_CPUS={int(cpu_limit)}",
        "-v", f"{source_path}:/app",
        "-w", "/app",
    ]
    
    for k, v in branch_cfg["env"].items(): cmd.extend(["-e", f"{k}={v}"])
    
    cmd.append(DOCKER_IMAGE)
    # Ensure py-spy is installed. If not, we might need to install it.
    # We prepend a pip install command if we are not sure. 
    # But for cleaner invocation, let's assumes it's there or user installs it.
    # Let's try to pip install it first just in case.
    shell_cmd = f"pip install py-spy && {' '.join(wrapper_cmd)}"
    
    cmd.extend(["/bin/bash", "-c", shell_cmd])
    
    print(f"\n[PROFILE] Running branch: {branch_cfg['name']}")
    print(f"CMD: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="medium", help="Workload config size")
    parser.add_argument("--rounds", type=int, default=1, help="Rounds to run (keep small for profiling)")
    args = parser.parse_args()
    
    for b in BRANCH_CONFIGS:
        if os.path.exists(b["path"]):
            run_profiling(args.config, args.rounds, b)
        else:
            print(f"Skipping missing path {b['path']}")

if __name__ == "__main__":
    main()
'''

RUN_BENCHMARK_CONTENT = r'''import os
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
    args = parser.parse_args()
    wait_until(args.start_time)
    
    for branch_cfg in BRANCH_CONFIGS:
        if os.path.exists(branch_cfg["path"]):
            for config in [args.filter_config] if args.filter_config else ALL_CONFIGS:
                run_single_benchmark(SCENARIOS[0], config, 0, args.rounds, 8, 8, branch_cfg)

if __name__ == "__main__":
    main()
'''


def inject_profiling_hooks(file_path):
    print(f"Injecting profiling hooks into {file_path}...")
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Define sync helper
    sync_func = [
        "\n",
        "# --- Profiling Hook Helper ---\n",
        "import os\n",
        "import time\n",
        "def sync_stage(flag_to_create, flag_to_wait):\n",
        "    with open(flag_to_create, 'w') as f: f.write('1')\n",
        "    while not os.path.exists(flag_to_wait): time.sleep(0.05)\n",
        "    try: os.remove(flag_to_wait)\n",
        "    except: pass\n",
        "# -----------------------------\n",
        "\n"
    ]
    
    # Insert helper before class definition
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("class TQBandwidthTester"):
            insert_idx = i
            break
    
    # If we didn't find the class (unlikely), fallback to top
    if insert_idx == 0: insert_idx = 20 
    
    lines[insert_idx:insert_idx] = sync_func
    
    final_lines = []
    
    # Zero copy check injection block
    check_code = [
        "    try:\n",
        "        from transfer_queue.utils import serial_utils\n",
        "        print(f'[Benchmark Startup Check] TQ_ZERO_COPY_SERIALIZATION = {serial_utils.TQ_ZERO_COPY_SERIALIZATION}')\n",
        "    except ImportError:\n",
        "        print('[Benchmark Startup Check] Could not import serial_utils to check flag.')\n"
    ]

    for line in lines:
        if "if __name__ == \"__main__\":" in line:
             final_lines.append(line)
             final_lines.extend(check_code)
             continue

        if "asyncio.run(self.data_system_client.async_put" in line:
            indent = line[:line.find(line.lstrip())]
            final_lines.append(f"{indent}if i == 0: sync_stage('init_ready.flag', 'put_start.flag')\n")
            final_lines.append(line)
            continue
            
        if "put_time = " in line and "start_put" in line:
            final_lines.append(line)
            indent = line[:line.find(line.lstrip())]
            final_lines.append(f"{indent}if i == 0: sync_stage('put_done.flag', 'get_prepare.flag')\n")
            continue
            
        if "retrieved_data = asyncio.run(self.data_system_client.async_get_data" in line:
            indent = line[:line.find(line.lstrip())]
            final_lines.append(f"{indent}if i == 0: sync_stage('get_ready.flag', 'get_start.flag')\n")
            final_lines.append(line)
            continue
            
        final_lines.append(line)

    with open(file_path, "w") as f:
        f.writelines(final_lines)

def main():
    if os.path.exists(PACKAGE_NAME):
        shutil.rmtree(PACKAGE_NAME)
    os.makedirs(PACKAGE_NAME)

    # 1. Export Code & Copy local utils
    for dir_name, branch in BRANCH_MAP.items():
        dest = os.path.join(PACKAGE_NAME, dir_name)
        os.makedirs(dest, exist_ok=True)
        print(f"Exporting {branch}...")
        subprocess.run(f"git archive --format=tar {branch} | tar -x -C {dest}", shell=True)
        shutil.copy("put_benchmark.py", os.path.join(dest, "put_benchmark.py"))
        
        # Inject Hooks
        inject_profiling_hooks(os.path.join(dest, "put_benchmark.py"))
        
        # Write Container Orchestrator
        with open(os.path.join(dest, "container_profiler.py"), "w") as f:
            f.write(CONTAINER_PROFILER_SCRIPT)

    # 2. Host Scripts
    with open(os.path.join(PACKAGE_NAME, "run_benchmark.py"), "w") as f:
        f.write(RUN_BENCHMARK_CONTENT)
        
    with open(os.path.join(PACKAGE_NAME, "run_profiling.py"), "w") as f:
        f.write(HOST_PROFILING_SCRIPT)

    # 3. Zip
    print("Zipping...")
    shutil.make_archive(PACKAGE_NAME, 'zip', ".", PACKAGE_NAME)
    print("Done!")

if __name__ == "__main__":
    main()
