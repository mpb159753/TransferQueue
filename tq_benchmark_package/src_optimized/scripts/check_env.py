import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
print(f"Python path: {sys.path}")
try:
    import transfer_queue
    print("Import transfer_queue SUCCESS")
    from transfer_queue import (
        AsyncTransferQueueClient,
        SimpleStorageUnit,
        TransferQueueController,
        process_zmq_server_info,
    )
    from transfer_queue.utils.utils import get_placement_group
    print("Import submodules SUCCESS")
    import ray
    print("Import ray SUCCESS")
    ray.init()
    print("Ray init SUCCESS")
    ray.shutdown()
except Exception as e:
    print(f"Import transfer_queue FAILED: {e}")
    import traceback
    traceback.print_exc()
