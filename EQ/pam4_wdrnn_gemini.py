import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.signal import lfilter, butter
from commpy.filters import rrcosfilter


# ------------------------------
# --- 1. 环境与设备自适应 ---
# ------------------------------
def get_device():
    """自动选择可用的计算设备 (NVIDIA GPU, Apple Silicon, or CPU)"""
    if torch.cuda.is_available():
        print("使用 NVIDIA CUDA 设备")
        return torch.device("cuda")
    # 注意：较新版本的PyTorch才能很好地支持MPS
    elif torch.backends.mps.is_available():
        print("使用 Apple MPS 设备")
        return torch.device("mps")
    else:
        print("使用 CPU")
        return torch.device("cpu")


DEVICE = get_device()


# ------------------------------
# --- 2. 仿真参数 ---
# ------------------------------
class SimParams:
    # 信号参数
    num_symbols = 60000  # 用于测试的总符号数 [cite: 121]
    train_symbols_ratio = 5 / 11  # 训练符号占总符号的比例, 对应论文的50000训练/60000测试 [cite: 121]
    baud_rate = 50e9  # 波特率 (论文中为100Gb/s PAM-4, 即50Gbaud)
    sps = 4  # 每个符号的采样点数 (Samples Per Symbol), 仿真中适当提高

    # RRC滤波器参数
    rolloff = 0.01  # 滚降系数, 对应论文 [cite: 125]
    filter_span = 20  # 滤波器长度（符号数）

    # 信道参数
    # 使用一个巴特沃斯低通滤波器模拟DML的带宽限制效应
    # 论文中平均10dB带宽为20.2GHz [cite: 7, 129]
    # 我们用一个3dB带宽近似的低通滤波器来简化模拟
    channel_bw_3db = 20e9  # 假设信道的3dB带宽
    channel_order = 4  # 滤波器阶数
    snr_db = 18  # 信噪比 (dB)，用于在接收端加入高斯白噪声

    # WD-RNN 均衡器参数
    # 论文中优化后的网络结构为(61, 20, 1) [cite: 191]
    # 为简化起见，我们缩小规模，但保留其结构
    input_taps = 31  # 输入层神经元数量 n[0]
    hidden_nodes = 10  # 隐藏层神经元数量 n[1]
    # 论文中优化后k=6 [cite: 194]
    feedback_taps = 6  # 反馈延迟数量 k

    # 训练参数
    learning_rate = 0.001
    epochs = 30  # 训练轮次 [cite: 286]
    batch_size = 256

    # WD (Weighted Decision) 参数
    # 论文中优化后 alpha=5, beta=0.14 [cite: 279]
    wd_alpha = 5.0
    wd_beta = 0.14


# 实例化参数
PARAMS = SimParams()


# ------------------------------
# --- 3. 发射机 (Transmitter) ---
# ------------------------------
class Transmitter:
    def __init__(self, params):
        self.params = params
        self.pam4_levels = [-3, -1, 1, 3]
        # 生成RRC滤波器抽头
        # 论文中提到Tx和Rx都有RRC滤波器(匹配滤波) [cite: 125, 135]
        self.fs = params.baud_rate * params.sps
        t, h = rrcosfilter(params.filter_span * params.sps + 1, params.rolloff, 1 / params.baud_rate, self.fs)
        self.rrc_filter_coeffs = h

    def generate_pam4_symbols(self, n_symbols):
        """生成随机PAM-4符号"""
        bits = np.random.randint(0, 4, n_symbols)
        symbols = np.array(self.pam4_levels)[bits]
        return symbols

    def upsample(self, symbols):
        """上采样"""
        upsampled = np.zeros(len(symbols) * self.params.sps)
        upsampled[::self.params.sps] = symbols
        return upsampled

    def filter(self, signal):
        """通过RRC滤波器进行脉冲整形"""
        return lfilter(self.rrc_filter_coeffs, 1, signal)

    def process(self, n_symbols):
        """完整的发射机处理流程"""
        # 对应论文 PAM-4 Generation -> Upsampling -> RRC filter [cite: 125]
        symbols = self.generate_pam4_symbols(n_symbols)
        upsampled_signal = self.upsample(symbols)
        tx_signal = self.filter(upsampled_signal)
        return tx_signal, symbols


# ------------------------------
# --- 4. 信道 (Channel) ---
# ------------------------------
class Channel:
    def __init__(self, params):
        self.params = params
        self.fs = params.baud_rate * params.sps
        # 设计一个低通滤波器来模拟带宽限制
        # 这是对DML带宽约束的一种简化模型
        nyquist = 0.5 * self.fs
        normal_cutoff = params.channel_bw_3db / nyquist
        self.b, self.a = butter(params.channel_order, normal_cutoff, btype='low', analog=False)

    def apply_bandwidth_limit(self, signal):
        """应用带宽限制"""
        return lfilter(self.b, self.a, signal)

    def apply_nonlinearity(self, signal):
        """应用一个简单的非线性函数来模拟DML非线性 [cite: 29]"""
        return np.tanh(signal * 0.5)  # 用tanh模拟压缩效应

    def process(self, signal):
        """完整的信道处理流程"""
        bw_limited_signal = self.apply_bandwidth_limit(signal)
        nonlinear_signal = self.apply_nonlinearity(bw_limited_signal)
        return nonlinear_signal


# ------------------------------
# --- 5. 接收机 (Receiver) ---
# ------------------------------
class Receiver:
    def __init__(self, params, rrc_coeffs):
        self.params = params
        self.rrc_filter_coeffs = rrc_coeffs  # 匹配滤波器

    def filter(self, signal):
        """匹配滤波"""
        return lfilter(self.rrc_filter_coeffs, 1, signal)

    def add_noise(self, signal):
        """添加高斯白噪声"""
        signal_power = np.mean(np.abs(signal) ** 2)
        sigma2 = signal_power * 10 ** (-self.params.snr_db / 10)
        noise = np.sqrt(sigma2 / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
        return signal.real + noise.real

    def downsample(self, signal):
        """下采样到1 sps (波特率采样) [cite: 135]"""
        # 简单的时钟恢复：找到能量最大的采样点
        # 实际系统中时钟恢复更复杂，这里做简化
        energy = np.array([np.sum(signal[i::self.params.sps] ** 2) for i in range(self.params.sps)])
        best_phase = np.argmax(energy)
        return signal[best_phase::self.params.sps]

    def process(self, signal):
        """完整的接收机处理流程"""
        noisy_signal = self.add_noise(signal)
        # 对应论文 Matched RRC filter -> Timing recovery -> Downsampling [cite: 135]
        filtered_signal = self.filter(noisy_signal)
        downsampled_signal = self.downsample(filtered_signal)
        return downsampled_signal


# -----------------------------------
# --- 6. WD-RNN 均衡器模型 ---
# -----------------------------------
class WDRNNEqualizer(nn.Module):
    # 对应论文 Fig. 1 的结构 [cite: 88]
    def __init__(self, params):
        super(WDRNNEqualizer, self).__init__()
        self.p = params

        # 定义网络层
        # 输入层: 当前及过去的接收符号 + 过去的反馈符号
        # w[1] in paper
        self.input_layer = nn.Linear(self.p.input_taps + self.p.feedback_taps, self.p.hidden_nodes)

        # 隐藏层激活函数
        # 论文选择tanh因为它对称的输出范围更匹配信号分布 [cite: 95]
        self.hidden_activation = nn.Tanh()

        # 输出层
        # w[2] in paper
        self.output_layer = nn.Linear(self.p.hidden_nodes, 1)
        # 输出层使用线性激活 [cite: 96]

    def pam4_slicer(self, x):
        """PAM-4 硬判决"""
        # 这是计算ŷ(n)的部分
        levels = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=x.device)
        return levels[(torch.abs(x.unsqueeze(-1) - levels)).argmin(dim=-1)]

    def compressed_sigmoid(self, x):
        """
        压缩S型函数 S(x)
        对应论文公式(4) [cite: 117]
        """
        alpha, beta = self.p.wd_alpha, self.p.wd_beta
        val = alpha * (x / beta - 1)
        return 0.5 * ((1 - torch.exp(-val)) / (1 + torch.exp(-val)) + 1)

    def forward(self, x, labels=None):
        """
        前向传播
        x: (batch_size, sequence_length)
        labels: (batch_size, sequence_length) - 仅在训练时提供
        """
        batch_size, seq_len = x.shape
        outputs = torch.zeros_like(x)

        # 初始化反馈缓冲区
        feedback_buffer = torch.zeros(batch_size, self.p.feedback_taps, device=x.device)

        for i in range(seq_len):
            # 准备输入向量
            # [x_{n}, x_{n-1}, ..., d_{n-1}, d_{n-2}, ...]
            input_slice = x[:, max(0, i - self.p.input_taps + 1):i + 1]
            if input_slice.shape[1] < self.p.input_taps:
                padding = torch.zeros(batch_size, self.p.input_taps - input_slice.shape[1], device=x.device)
                input_slice = torch.cat([padding, input_slice], dim=1)

            combined_input = torch.cat([input_slice, feedback_buffer], dim=1)

            # 网络前向传播
            # 对应论文公式(1) [cite: 101]
            hidden_out = self.hidden_activation(self.input_layer(combined_input))
            y_n = self.output_layer(hidden_out).squeeze(-1)  # 当前符号的原始输出
            outputs[:, i] = y_n

            # --- 关键部分：根据训练/测试模式选择反馈信号 ---
            if self.training:
                # 训练模式: 使用教师强制 (Teacher Forcing)
                # 反馈真实的标签
                feedback_signal = labels[:, i]
            else:
                # 测试模式: 使用加权判决 (Weighted Decision)
                y_hat_n = self.pam4_slicer(y_n)  # 硬判决 ŷ(n)

                # 计算可靠性 γ_n
                # 对应论文公式(3)
                gamma_n = 1.0 - torch.abs(y_n - y_hat_n)

                # 计算S(γ_n)
                s_gamma_n = self.compressed_sigmoid(gamma_n)

                # 计算加权判决 ỹ(n)
                # 对应论文公式(2)
                y_tilde_n = s_gamma_n * y_hat_n + (1 - s_gamma_n) * y_n
                feedback_signal = y_tilde_n

            # 更新反馈缓冲区
            feedback_buffer = torch.roll(feedback_buffer, shifts=1, dims=1)
            feedback_buffer[:, 0] = feedback_signal

        return outputs


# -----------------------------------
# --- 7. 辅助函数和主流程 ---
# -----------------------------------
def generate_dataset(params):
    """生成完整的仿真数据集"""
    print("--- 正在生成仿真数据 ---")
    tx = Transmitter(params)
    channel = Channel(params)
    rx = Receiver(params, tx.rrc_filter_coeffs)

    tx_signal, original_symbols = tx.process(params.num_symbols)
    channel_signal = channel.process(tx_signal)
    rx_symbols = rx.process(channel_signal)

    # 归一化接收符号能量
    rx_symbols = rx_symbols / np.sqrt(np.mean(rx_symbols ** 2)) * np.sqrt(np.mean(original_symbols ** 2))

    # 忽略滤波器暂态效应
    transient = params.filter_span * params.sps
    original_symbols = original_symbols[transient:]
    rx_symbols = rx_symbols[transient:len(original_symbols)]

    print(f"数据生成完毕，共 {len(rx_symbols)} 个有效符号。")
    return torch.FloatTensor(rx_symbols), torch.FloatTensor(original_symbols)


def calculate_ber(y_true, y_pred):
    """计算BER"""
    pam4_map = {-3: '00', -1: '01', 1: '11', 3: '10'}  # Gray mapping

    # 确保是numpy array
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # 符号到比特的映射
    bits_true = "".join([pam4_map[s] for s in y_true])
    bits_pred = "".join([pam4_map[s] for s in y_pred])

    error_bits = sum(c1 != c2 for c1, c2 in zip(bits_true, bits_pred))
    total_bits = len(bits_true)

    ber = error_bits / total_bits
    return ber


def main():
    """主执行函数"""
    rx_data, labels = generate_dataset(PARAMS)

    # 划分训练集和测试集
    train_size = int(PARAMS.train_symbols_ratio * len(rx_data))

    X_train = rx_data[:train_size].unsqueeze(0).to(DEVICE)
    y_train = labels[:train_size].unsqueeze(0).to(DEVICE)
    X_test = rx_data[train_size:].unsqueeze(0).to(DEVICE)
    y_test = labels[train_size:]  # BER计算在CPU上进行

    # 初始化模型、损失函数和优化器
    model = WDRNNEqualizer(PARAMS).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=PARAMS.learning_rate)

    # --- 训练过程 ---
    print("\n--- 开始训练WD-RNN均衡器 ---")
    model.train()  # 设置为训练模式
    for epoch in range(PARAMS.epochs):
        optimizer.zero_grad()
        # 在训练时传入标签以启用教师强制
        outputs = model(X_train, y_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{PARAMS.epochs}], Loss: {loss.item():.6f}')

    print("--- 训练完成 ---")

    # --- 测试过程 ---
    print("\n--- 开始测试 ---")
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        # 测试时不传入标签，模型内部将使用加权判决
        equalized_output = model(X_test).squeeze(0)

    # 在CPU上进行判决和BER计算
    equalized_output_cpu = equalized_output.cpu()

    # 原始未均衡信号的BER
    raw_sliced = model.pam4_slicer(X_test.squeeze(0).cpu()).long()
    ber_before = calculate_ber(y_test.long().numpy(), raw_sliced.numpy())
    print(f'均衡前 BER: {ber_before:.2e}')

    # 均衡后信号的BER
    equalized_sliced = model.pam4_slicer(equalized_output_cpu).long()
    ber_after = calculate_ber(y_test.long().numpy(), equalized_sliced.numpy())
    print(f'均衡后 BER: {ber_after:.2e}')

    # 论文中以7% HD-FEC作为门限，BER为3.8e-3 [cite: 8]
    fec_threshold = 3.8e-3
    if ber_after < fec_threshold:
        print(f"性能达标：BER ({ber_after:.2e}) 低于 7% HD-FEC 门限 ({fec_threshold:.2e})")
    else:
        print(f"性能未达标：BER ({ber_after:.2e}) 高于 7% HD-FEC 门限 ({fec_threshold:.2e})")


if __name__ == '__main__':
    # 确保可复现性
    torch.manual_seed(42)
    np.random.seed(42)

    # 安装依赖
    print("请确保已安装 PyTorch, NumPy, SciPy, commpy:")
    print("pip install torch numpy scipy scikit-commpy")

    main()