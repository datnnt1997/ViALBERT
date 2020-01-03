import os


class ModelSaver(object):
    def __init__(self):
        pass

    def save(self):
        return NotImplementedError

    @staticmethod
    def _rm_checkpoint(name):
        os.remove(name)
