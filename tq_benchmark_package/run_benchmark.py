import os
import sys
import json
import time
import argparse
import subprocess
import threading
import datetime
import re
import socket

# from transfer_queue.utils import zmq_utils # Needed for get_ip_address if available, or use socket

DOCKER_IMAGE = "run_test"
ALL_CONFIGS = ["debug", "tiny", "small", "medium", "large", "xlarge", "huge"]
SHARD_VARIANTS = [8]

BRANCH_CONFIGS = [
    # Group 1: Pre-refactor baseline (commit 87d7e13)
    {"name": "pre-refactor", "path": "tq_benchmark_package/src_pre_refactor", "env_vars": {"PYTHONPATH": "/app"}, "description": "Pre-Refactor"},
    # Group 2: Columnar BatchMeta refactor (commit 6ad4d07)
    {"name": "columnar-batch-meta", "path": "tq_benchmark_package/src_columnar_batch_meta", "env_vars": {"PYTHONPATH": "/app"}, "description": "Columnar-BatchMeta"},
    # Group 3: Columnar FieldSchema refactor (refactor/columnar-field-schema branch latest)
    {"name": "columnar-field-schema", "path": "tq_benchmark_package/src_columnar_field_schema", "env_vars": {"PYTHONPATH": "/app"}, "description": "Columnar-FieldSchema"},
]
SCENARIOS = [{"name": "TransferQueue", "cmd_args": [], "env_vars": {"PYTHONPATH": "."}, "workdir": "."}]

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
        s.close()
    except Exception:
        IP = '127.0.0.1'
    return IP

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

def get_free_port():
    """Get a random free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def run_single_benchmark_local(scenario, config_name, run_id, rounds, shards, cpu_limit, branch_config, role="single", head_ip=None, worker_ip=None, head_port=None):
    container_name = f"bench_{run_id}_{int(time.time())}"
    source_path = os.path.abspath(branch_config["path"])
    cpuset_str = get_cpuset_range(cpu_limit)
    result_file = os.path.join(source_path, "res.json")
    
    # Clean up old result file only if we are head or single
    if role in ["single", "head"] and os.path.exists(result_file):
        os.remove(result_file)
    
    cmd = [
        "docker", "run", "--rm", "--name", container_name, "--ipc=host", "--network=host",
        f"--cpuset-cpus={cpuset_str}",
        "-e", "RAY_DISABLE_DOCKER_CPU_WARNING=1",
        "-e", f"RAY_NUM_CPUS={int(cpu_limit)}",
        "-v", f"{source_path}:/app", "-w", "/app"
    ]
    for k, v in branch_config.get("env_vars", {}).items(): cmd.extend(["-e", f"{k}={v}"])
    cmd.append(DOCKER_IMAGE)
    
    # Construct command to start Ray inside container
    ray_start_cmd = ""
    # Only start external Ray for distributed modes
    if role == "head":
        ray_start_cmd = f"ray start --head --node-ip-address={head_ip} --port={head_port} --num-cpus={int(cpu_limit)} --include-dashboard=false --disable-usage-stats --block & sleep 5 && "
    elif role == "worker":
        ray_start_cmd = f"ray start --address={head_ip}:{head_port} --node-ip-address={worker_ip} --num-cpus={int(cpu_limit)} --disable-usage-stats --block & sleep 5 && "
    # role == "single": do NOT start ray, let script handle it

    # Construct python command
    # put_benchmark.py in refactor branch only accepts: --ip, --config, --output, --rounds, --shards, --profile
    py_cmd_str = f"python scripts/put_benchmark.py --config {config_name} --rounds {rounds} --shards {shards}"
    
    if role == "single":
         py_cmd_str += " --output res.json"
         # No --role, No --ip (defaults to local init)
    elif role == "head":
        py_cmd_str += f" --output res.json --ip {head_ip}" 
        # passing ip triggers address="auto", connecting to the ray started above
    elif role == "worker":
        # Worker doesn't run the python script main logic usually, but if it did:
        py_cmd_str += f" --ip {head_ip}"

    # Wrap in bash -c, to run ray stop only if ray started?
    # Or just always run ray stop for safety (it will fail harmlessly if not started)
    final_cmd = ["/bin/bash", "-c", f"{ray_start_cmd}{py_cmd_str}; ray stop || true; sleep 2"]
    cmd.extend(final_cmd)
    
    print(f"Running {branch_config['name']} - {config_name} [Role: {role}]...")
    # print(f"CMD: {' '.join(cmd)}")
    monitor = DockerMonitor(container_name)
    try:
        p = subprocess.Popen(cmd)
        monitor.start()
        p.wait()
    except KeyboardInterrupt:
        print(f"\n[Stop] Interrupted by user. Stopping container {container_name}...")
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
        raise  # Re-raise to trigger cleanup in main
    except Exception as e:
        print(f"Error running benchmark: {e}")
        p.kill()
    finally:
        monitor.stop()
        # Ensure container is stopped if it's still running (double safety)
        subprocess.run(["docker", "stop", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    docker_stats = monitor.get_summary()
    print(f"  Docker Stats: {docker_stats}")
    
    # Only read results if we are head or single
    results = []
    if role in ["single", "head"]:
        if os.path.exists(result_file):
            try:
                with open(result_file, "r") as f:
                    benchmark_results = json.load(f)
                for r in benchmark_results:
                    r["branch"] = branch_config["name"]
                    r["docker_stats"] = docker_stats
                    results.append(r)
            except Exception as e:
                print(f"  Warning: Failed to read results from {result_file}: {e}")
        else:
            print(f"  Warning: Result file {result_file} not found")
    
    return results

def deploy_and_run_worker(worker_ip, ssh_user, ssh_key, deploy_path, branch_config, config_name, rounds, shards, docker_cpu, head_ip, head_port):
    print(f"\n[Deploy] Syncing code to {worker_ip}...")
    
    ssh_opts = f"-o StrictHostKeyChecking=no -i {ssh_key}"
    
    # 1. Create directory
    subprocess.run(f"ssh {ssh_opts} {ssh_user}@{worker_ip} 'mkdir -p {deploy_path}'", shell=True, check=True)
    
    # 2. Rsync current package
    rsync_cmd = f"rsync -avz -e 'ssh {ssh_opts}' --exclude '*.json' --exclude '__pycache__' ./ {ssh_user}@{worker_ip}:{deploy_path}/"
    subprocess.run(rsync_cmd, shell=True, check=True)
    
    # 3. Run worker command in background
    print(f"[Worker] Starting worker on {worker_ip} connecting to {head_ip}:{head_port}...")
    
    remote_cmd = (
        f"cd {deploy_path} && "
        f"python3 run_benchmark.py "
        f"--role worker "
        f"--head-ip {head_ip} "
        f"--head-port {head_port} " # New arg
        f"--worker-ip {worker_ip} "
        f"--filter_config {config_name} " 
        f"--rounds {rounds} "
        f"--shards {shards} "
        f"--cpus {docker_cpu} "
        f"--filter_branch {branch_config['name']}" 
    )
    
    final_ssh_cmd = f"ssh {ssh_opts} {ssh_user}@{worker_ip} '{remote_cmd}'"
    # print(f"CMD: {final_ssh_cmd}")
    
    return subprocess.Popen(final_ssh_cmd, shell=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter_config", type=str)
    parser.add_argument("--start_time", type=str)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--cpus", type=int, default=30, help="Number of CPUs to allocate for Docker container")
    parser.add_argument("--shards", type=int, default=8, help="Number of storage shards")
    parser.add_argument("--output", type=str, default="benchmark_summary.json", help="Output summary JSON file")
    
    # Dual-node args
    parser.add_argument("--role", type=str, default="single", choices=["single", "head", "worker"], help="Node role")
    parser.add_argument("--head-ip", type=str, default="192.168.100.1", help="Head node IP (required for worker)")
    parser.add_argument("--head-port", type=int, help="Head node Port (for worker)")
    parser.add_argument("--worker-ip", type=str, help="Worker node IP (for deployment or identification)")
    parser.add_argument("--ssh-user", type=str, default="root", help="SSH user for remote deployment")
    parser.add_argument("--ssh-key", type=str, default="/root/.ssh/id_ed25519", help="SSH key path")
    parser.add_argument("--deploy-path", type=str, default="/tmp/tq_benchmark", help="Remote deployment path")
    parser.add_argument("--filter_branch", type=str, help="Filter specific branch config to run (internal use)")

    args = parser.parse_args()
    wait_until(args.start_time)
    
    # If we are in 'single' mode but have a worker_ip, we become the 'head' orchestrator
    is_orchestrator = args.role == "single" and args.worker_ip is not None
    
    # Determine Head Port
    head_port = args.head_port
    if is_orchestrator:
         # Generate a random port for head
        head_port = get_free_port()
        print(f"[Mode] Dual-node Orchestrator (Auto-Deploy). Selected Head Port: {head_port}")
        # Hardcode Head IP as 192.168.100.1 per user request (simplification)
        my_ip = "192.168.100.1"
        print(f"Local IP (Head): {my_ip}")
        print(f"Remote IP (Worker): {args.worker_ip}")
        
        args.head_ip = my_ip
        args.role = "head" 
    
    if args.role == "single" and not head_port:
        head_port = get_free_port() # Use dynamic port for single mode to avoid conflicts

    if args.role == "worker" and not (args.head_ip and head_port):
         print("Error: Worker requires --head-ip and --head-port")
         return

    all_results = []
    
    target_branches = BRANCH_CONFIGS
    if args.filter_branch:
        target_branches = [b for b in BRANCH_CONFIGS if b["name"] == args.filter_branch]
    
    for branch_cfg in target_branches:
        if os.path.exists(branch_cfg["path"]):
            for config in [args.filter_config] if args.filter_config else ALL_CONFIGS:
                
                worker_proc = None
                
                if is_orchestrator:
                    print("-" * 50)
                    print(f"Preparing Dual-Node Test for {branch_cfg['name']} - {config}")
                    
                    # Start worker in a separate thread after a delay to ensure Head starts first
                    def delayed_deploy():
                        print("Waiting 10s for Head to initialize before deploying Worker...")
                        time.sleep(10)
                        deploy_and_run_worker(
                            args.worker_ip, args.ssh_user, args.ssh_key, args.deploy_path,
                            branch_cfg, config, args.rounds, args.shards, args.cpus, args.head_ip, head_port
                        )
                    
                    # We can't easily capture the subprocess Popen from a thread to kill it later 
                    # unless we use a shared variable or class.
                    # For simplicity, let's just fire and forget the deployment? 
                    # No, we need to kill it.
                    # Let's use a class wrapper or strict threading.
                    # Simple approach: launch thread, let it run.
                    # Warning: we won't be able to kill it cleaning up.
                    # Better: start logic now.
                    t = threading.Thread(target=delayed_deploy)
                    t.start()
                    
                    # We will rely on manual cleanup or user killing workers if they hang.
                    # Or we can send a kill command via SSH at end of loop.
                
                try:
                    results = run_single_benchmark_local(
                        SCENARIOS[0], config, 0, args.rounds, args.shards, args.cpus, 
                        branch_cfg, role=args.role, head_ip=args.head_ip, worker_ip=args.worker_ip, head_port=head_port
                    )
                    
                    if results:
                        all_results.extend(results)
                except KeyboardInterrupt:
                    print("\n[Main] Benchmark interrupted. Cleaning up...")
                    break # Break inner loop, will fall through to finally
                finally:
                    # Cleanup worker
                    if is_orchestrator:
                        print("Stopping remote worker...")
                        # Just ssh kill python processes in deploy path?
                        subprocess.run(f"ssh -o StrictHostKeyChecking=no -i {args.ssh_key} {args.ssh_user}@{args.worker_ip} 'pkill -f run_benchmark.py'", shell=True)
                        pass
        if args.role == "head" and is_orchestrator and not all_results:
             break # Stop outer loop on interrupt

    # Save results
    if args.role in ["single", "head"] and all_results:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\n💾 Summary saved to {args.output} ({len(all_results)} records)")
    elif args.role in ["single", "head"]:
        print("\n⚠️ No results collected")

if __name__ == "__main__":
    main()

