class MemoryManager:
    """Simple reference-count based memory manager for node outputs."""
    def __init__(self, nodes, outgoing):
        """Initialize with reference counts based on ``outgoing`` links."""
        self.ref_counts = {nid: len(outgoing.get(nid, [])) for nid in nodes}
        self.values = {}

    def store(self, nid, value):
        """Store value if it will be needed by other nodes."""
        if self.ref_counts.get(nid, 0) > 0:
            self.values[nid] = value

    def get(self, nid):
        """Retrieve value and decrease its reference count."""
        val = self.values.get(nid)
        if nid in self.ref_counts:
            self.ref_counts[nid] -= 1
            if self.ref_counts[nid] <= 0 and nid in self.values:
                del self.values[nid]
        return val

    def flush(self):
        """Clear all stored values and release GPU memory if possible."""
        # ensure all references held by this manager are removed
        for key in list(self.values.keys()):
            try:
                del self.values[key]
            except Exception:
                pass
        self.values.clear()
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
