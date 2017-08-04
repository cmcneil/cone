import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glabtools.io as io
import cortex


def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


if run_from_ipython():
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic('matplotlib inline')
    from IPython.display import Image

cci = io.get_cc_interface()
