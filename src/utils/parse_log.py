import re
import numpy as np

def parse_log(path: str):
    with open(path, "r") as f:
        log = f.read()

    regex = r"inference speed = (\d+\.\d+)"
    fps = np.array([float(match) for match in re.findall(regex, log)])

    regex = r"size:(\d+)->(\d+)"
    sizes = np.array([(int(insize), int(outsize)) for insize, outsize in re.findall(regex, log)])
    insizes = sizes[:,0]
    outsizes = sizes[:,1]

    latencies = outsizes / fps

    return fps, latencies, insizes