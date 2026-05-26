import os
import sys
import gc
import ctypes

try:
    libc = ctypes.CDLL("libc.so.6")
except Exception:
    libc = None

def trim_memory():
    """Force garbage collection and return free memory to the OS."""
    gc.collect()
    if libc is not None:
        try:
            libc.malloc_trim(0)
        except Exception:
            pass

def setup_malloc_arena():
    """Enforce MALLOC_ARENA_MAX=1 to avoid fragmentation on multi-core systems.
    
    If the environment variable is not set, sets it and re-executes the script.
    """
    if os.environ.get("MALLOC_ARENA_MAX") != "1":
        os.environ["MALLOC_ARENA_MAX"] = "1"
        print("Re-executing script with MALLOC_ARENA_MAX=1...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
