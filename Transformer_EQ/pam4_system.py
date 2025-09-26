import numpy as np
import torch
from scipy.signal import upfirdn


class PAM4System:
    def __init__(self, sps=8, rolloff=0.1, filter_len=1024):
        self.sps = sps  # Samples per symbol
        self.rolloff = rolloff
        self.filter_len = filter_len
        self.pam4_levels = [-3, -1, 1, 3]
        self.pam4_mapping = {0: -3, 1: -1, 2: 1, 3: 3}
        self.rrc_filter = self._design_rrc_filter()

    def _design_rrc_filter(self):
        t = np.arange(-self.filter_len / 2, self.filter_len / 2) / self.sps
        with np.errstate(divide='ignore', invalid='ignore'):
            beta = self.rolloff
            numerator = np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))
            denominator = np.pi * t * (1 - (4 * beta * t) ** 2)
            h_rrc = (1 / np.sqrt(self.sps)) * (numerator / denominator)
            h_rrc[t == 0] = (1 / np.sqrt(self.sps)) * (1 - beta + 4 * beta / np.pi)
            h_rrc[np.abs(t) == 1 / (4 * beta)] = (beta / (np.sqrt(2 * self.sps))) * \
                                                 ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (
                                                             1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        return h_rrc

    def transmit(self, symbols_indices):
        # Map indices to PAM4 levels
        pam4_symbols = np.array([self.pam4_mapping[i] for i in symbols_indices])

        # Upsample
        upsampled_symbols = np.zeros(len(pam4_symbols) * self.sps)
        upsampled_symbols[::self.sps] = pam4_symbols

        # Pulse shaping
        tx_signal = upfirdn(self.rrc_filter, upsampled_symbols, up=1, down=1)
        return tx_signal, pam4_symbols

    # MODIFIED FUNCTION
    def receive(self, rx_signal, num_symbols):
        # Matched filtering
        matched_filtered_signal = upfirdn(self.rrc_filter, rx_signal, up=1, down=1)

        # Downsample, accounting for the combined filter delay
        start_index = self.filter_len - 1
        downsampled_symbols = matched_filtered_signal[start_index::self.sps]

        # Truncate the output to the expected number of symbols
        return downsampled_symbols[:num_symbols]