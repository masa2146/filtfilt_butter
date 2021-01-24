import numpy
import numpy.testing
import scipy.signal
from matplotlib import pyplot
from custom_filtfilt import *


def sinusoid(sampling_frequency_Hz=50.0, signal_frequency_Hz=1.0, periods=1.0,
             amplitude=1.0, offset=0.0, phase_deg=0.0, noise_std=0.1):
    """
    Create a noisy test signal sampled from a sinusoid (time series)

    """
    signal_frequency_rad_per_s = signal_frequency_Hz * 2 * numpy.pi
    phase_rad = numpy.radians(phase_deg)
    duration_s = periods / signal_frequency_Hz
    number_of_samples = int(duration_s * sampling_frequency_Hz)
    time_s = (numpy.array(range(number_of_samples), float) /
              sampling_frequency_Hz)
    angle_rad = signal_frequency_rad_per_s * time_s
    signal = offset + amplitude * numpy.sin(angle_rad - phase_rad)
    noise = numpy.random.normal(loc=0.0, scale=noise_std, size=signal.shape)
    return signal + noise


if __name__ == '__main__':
    # Design filter
    sampling_freq_hz = 50.0
    cutoff_freq_hz = 2.5
    order = 4
    normalized_frequency = cutoff_freq_hz * 2 / sampling_freq_hz
    b, a = scipy.signal.butter(order, normalized_frequency, btype='lowpass')

    # Create test signal
    signal = sinusoid(sampling_frequency_Hz=sampling_freq_hz,
                      signal_frequency_Hz=1.5, periods=3, amplitude=2.0,
                      offset=2.0, phase_deg=25)

    # Apply zero-phase filters
    print("Type: ", type(a), " A: ", a)
    print("B: ", b)
    print("Signal: ", signal)
    filtered_custom = custom_filtfilt(b, a, signal)
    filtered_scipy = scipy.signal.filtfilt(b, a, signal)

    # Verify near-equality
    # numpy.testing.assert_array_almost_equal(filtered_custom, filtered_scipy,
    #                                         decimal=12)

    # Plot result
    pyplot.subplot(1, 2, 1)
    pyplot.plot(signal)
    pyplot.plot(filtered_scipy)
    pyplot.plot(filtered_custom, '.')
    pyplot.title('raw vs filtered signals')
    pyplot.legend(['raw', 'scipy filtfilt', 'custom filtfilt'])
    pyplot.subplot(1, 2, 2)
    pyplot.plot(filtered_scipy-filtered_custom)
    pyplot.title('difference (scipy vs custom)')
    pyplot.show()
