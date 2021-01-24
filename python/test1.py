from scipy import signal
import numpy as np
from custom_filtfilt import *


def getDataFromFile(path):
    with open(path) as f:
        array = np.array([float(line)for line in f])
    return array


def getNumaretor():
    return getDataFromFile('data/b.txt')


def getDenumaretor():
    return getDataFromFile('data/a.txt')


def getSignals():
    return getDataFromFile('data/x_c.txt')


def getExpectedRes():
    return getDataFromFile('data/x_f.txt')


b = getNumaretor()
a = getDenumaretor()
sig = getSignals()
expect = getExpectedRes()

# a = []
# b = []
# sig = []

# b.append(4.2)
# b.append(5.3)
# b.append(6.3)

# a.append(1.3)
# a.append(2.3)
# a.append(3.2)

# sig.append(1)
# sig.append(2)
# sig.append(3)
# sig.append(4)
# sig.append(5)
# sig.append(6)
# sig.append(7)
# sig.append(8)
# sig.append(9)
# sig.append(10)

print("A ", a)
print("B ", b)
print("Sig ", sig)

# filtered_scipy = signal.filtfilt(b, a, sig)
filtered_custom = custom_filtfilt(b, a, sig)
print("Result: ",filtered_custom)
