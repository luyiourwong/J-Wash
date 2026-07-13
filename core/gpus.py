import pynvml

_initialized = False


def _ensure_init():
    global _initialized
    if not _initialized:
        pynvml.nvmlInit()
        _initialized = True


def gpu_stats():
    _ensure_init()
    stats = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        stats.append(
            {
                "index": index,
                "name": name,
                "vram_total": memory.total,
                "vram_used": memory.used,
                "util_pct": util.gpu,
            }
        )
    return stats
