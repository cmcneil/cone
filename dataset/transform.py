import numpy as np
import os
import sys


def log_xform(x):
    return np.sign(x) * np.log1p(np.abs(x))


def map_over_folders(xfm, fname, directory, print_freq=20):
    i = 0
    for p, d, f in os.walk(directory):
        if fname in f:
            location = os.path.join(p, fname)
            dat = np.load(location)
            np.save(location, xfm(dat))
            i += 1
            if i % print_freq == 0:
                print('...transformed %d files' % (i,))


if __name__ == "__main__":
    directory = sys.argv[2]
    fname = sys.argv[3]
    if sys.argv[1] == 'logxform':
        map_over_folders(log_xform, fname, directory)
    else:
        print('Error, function not available.')
