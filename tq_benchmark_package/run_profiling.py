import os
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
