from .ComfygQueue import ComfygQueue 
from .ComfygQueue import ComfygQueueTrigger

NODE_CLASS_MAPPINGS = {
    "ComfygQueue": ComfygQueue,
    "ComfygQueueTrigger": ComfygQueueTrigger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    ComfygQueue: "Batch Resolution Generator", 
    ComfygQueueTrigger: "Multi-Queue Trigger"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']