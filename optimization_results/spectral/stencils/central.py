from .base import ReconstructionStencil


class Central(ReconstructionStencil):
    def __init__(self):
        pass

    @staticmethod
    def apply(value):
        if not isinstance(value, list):
            raise Exception("Inputs must be a list.")
        else:
            return 0.5 * (value[0] + value[1])
