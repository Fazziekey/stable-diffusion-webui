import threading
from modules import scripts, scripts_postprocessing

class ExtrasThread(threading.Thread):

    def __init__(self, name, image, args):
        threading.Thread.__init__(self, name=name)
        self.name = name
        self.args=args
        self.image = image
        self.pp = None
    
    def run(self):
        print(f"Starting threading of {self.name}")
        self.pp = scripts_postprocessing.PostprocessedImage(self.image.convert("RGB"))
        scripts.scripts_postproc.run(self.pp, self.args)
    
    def join(self):
        super().join()
        print(f"Returning result of {self.name}")
        return self.pp

