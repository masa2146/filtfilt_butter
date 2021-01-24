import numpy


def custom_filter(b, a, x):
    """ 
    Filter implemented using state-space representation.

    Assume a filter with second order difference equation (assuming a[0]=1):

        y[n] = b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2] + ...
                         - a[1]*y[n-1] - a[2]*y[n-2]

    """
    # State space representation (transposed direct form II)
    A = numpy.array([[-a[1], 1], [-a[2], 0]])
    B = numpy.array([b[1] - b[0] * a[1], b[2] - b[0] * a[2]])
    C = numpy.array([1.0, 0.0])
    D = b[0]

    # print("A: ", A)
    # print("B: ", B)
    # print("C: ", C)
    # print("D: ", D)

    eye = numpy.eye(2)
    # print("eye: ", eye)
    # Determine initial state (solve zi = A*zi + B, see scipy.signal.lfilter_zi)
    zi = numpy.linalg.solve(eye - A, B)

    # print("zi: ", zi)

    # Scale the initial state vector zi by the first input value
    z = zi * x[0]

    # Apply filter
    y = numpy.zeros(numpy.shape(x))
    for n in range(len(x)):
        # Determine n-th output value (note this simplifies to y[n] = z[0] + b[0]*x[n])
        y[n] = numpy.dot(C, z) + D * x[n]
        # Determine next state (i.e. z[n+1])
        z = numpy.dot(A, z) + B * x[n]
    return y


def custom_filtfilt(b, a, x):
    # Apply 'odd' padding to input signal
    # the scipy.signal.filtfilt default
    padding_length = 3 * max(len(a), len(b))
    x_forward = numpy.concatenate((
        [2 * x[0] - xi for xi in x[padding_length:0:-1]],
        x,
        [2 * x[-1] - xi for xi in x[-2:-padding_length-2:-1]]))

    # Filter forward
    y_forward = custom_filter(b, a, x_forward)

    # Filter backward
    x_backward = y_forward[::-1]  # reverse
    y_backward = custom_filter(b, a, x_backward)

    # Remove padding and reverse
    return y_backward[-padding_length-1:padding_length-1:-1]
