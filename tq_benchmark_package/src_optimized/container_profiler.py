import os
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
