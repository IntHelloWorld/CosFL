import os


class WorkDir():
    def __init__(self, path):
        self.work_dir = path
        self.cwd = os.getcwd()
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    
    def __enter__(self):
        os.chdir(self.work_dir)
        return None
    
    def __exit__(self, exc_type, exc_value, traceback):
        os.chdir(self.cwd)
        return True


