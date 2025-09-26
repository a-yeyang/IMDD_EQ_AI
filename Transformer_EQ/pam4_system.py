import numpy as np
import torch
from scipy.signal import upfirdn
from commpy.filters import rrcosfilter  # 导入commpy的RRC滤波器函数


class PAM4System:
    def __init__(self, sps=2, rolloff=0.1, num_taps=8):
        """
        初始化PAM4系统.
        :param sps: 每个符号的采样点数 (Samples per symbol).
        :param rolloff: RRC滤波器的滚降系数.
        :param num_taps: RRC滤波器的抽头数 (滤波器长度). 建议为奇数.
        """
        if num_taps % 2 == 0:
            print("Warning: num_taps should preferably be an odd number. Adding 1.")
            num_taps += 1

        self.sps = sps
        self.rolloff = rolloff
        self.num_taps = num_taps
        self.pam4_levels = [-3, -1, 1, 3]
        self.pam4_mapping = {0: -3, 1: -1, 2: 1, 3: 3}

        # 使用 commpy 设计RRC滤波器
        self.rrc_filter = self._design_rrc_filter()

    def _design_rrc_filter(self):
        """
        使用commpy库的rrcosfilter函数设计根升余弦滤波器.
        Ts (符号周期) 和 Fs (采样率) 的比值决定了每个符号的采样点数.
        这里我们设置 Ts=1, Fs=sps, 意味着 Fs/Ts = sps.
        """
        # commpy.rrcosfilter 返回 (时间序列, 抽头系数) 的元组
        _, h_rrc = rrcosfilter(self.num_taps, self.rolloff, Ts=1, Fs=self.sps)
        return h_rrc

    def transmit(self, symbols_indices):
        """发射信号"""
        # Map indices to PAM4 levels
        pam4_symbols = np.array([self.pam4_mapping[i] for i in symbols_indices])

        # Upsample
        upsampled_symbols = np.zeros(len(pam4_symbols) * self.sps)
        upsampled_symbols[::self.sps] = pam4_symbols

        # Pulse shaping with the RRC filter
        tx_signal = upfirdn(self.rrc_filter, upsampled_symbols, up=1, down=1)
        return tx_signal, pam4_symbols

    def receive(self, rx_signal, num_symbols):
        """接收信号"""
        # Matched filtering
        matched_filtered_signal = upfirdn(self.rrc_filter, rx_signal, up=1, down=1)

        # Downsample, accounting for the combined filter delay.
        # The total delay of two matched filters is num_taps - 1 samples.
        # The optimal sampling point is at the center, so the delay in samples is (num_taps - 1) / 2.
        # However, to align with the previous code structure and typical simulation setups,
        # we start sampling after the main lobe of the convolved filter has passed its peak.
        # Using `num_taps - 1` as the start index is a common practice that accounts for the full filter length delay.
        start_index = self.num_taps - 1
        downsampled_symbols = matched_filtered_signal[start_index::self.sps]

        # Truncate the output to the expected number of symbols
        return downsampled_symbols[:num_symbols]