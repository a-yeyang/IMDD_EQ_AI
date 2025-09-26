import numpy as np


class OpticalChannel:
    def __init__(self, snr_db):
        self.snr_db = snr_db

    def propagate(self, signal):
        # Add AWGN
        signal_power = np.mean(np.abs(signal) ** 2)
        sigma2 = signal_power * 10 ** (-self.snr_db / 10)
        noise = np.sqrt(sigma2 / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))

        # For a real-valued signal
        noise = np.real(noise)

        # Introduce some simple ISI (can be made more complex)
        channel_response = np.array([0.8, 0.1, -0.05])
        signal_with_isi = np.convolve(signal, channel_response, mode='same')

        return signal_with_isi + noise