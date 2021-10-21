import numpy as np


def save_output(data, filename: str):
    np.savetxt(filename, data, fmt="%d", delimiter=",", header="ImageId,Label", comments="")
