import torch
import threading

from modules import shared, progress

thread_lock = threading.Lock()

class func_thread(threading.Thread):

    def __init__(self, name, func, id_task):
        threading.Thread.__init__(self, name=name)
        self.name = name
        self.func = func
        self.result = None
        self.id_task = id_task
    
    def run(self):
        torch.cuda.set_device(f"cuda:{self.name}")
        shared.state.begin()
        progress.start_task(self.id_task)
        print(f"-----------[2] current device is cuda{torch.cuda.current_device()}-----------")
        self.result = self.func
    
    def join(self):
        super().join()
        return self.result

