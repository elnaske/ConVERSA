from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info

def get_flops(model, inp):
    pass

def get_macs(model, inp):
    pass

def get_n_operations(model):
    # Load a test file for tracing

    flops = get_flops()
    macs = get_macs()
    return flops, macs