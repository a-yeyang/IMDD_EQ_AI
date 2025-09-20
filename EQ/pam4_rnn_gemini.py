import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib
import platform
import matplotlib.pyplot as plt
# 配置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
# ==============================================================================
# 1. 环境配置 (依据: 无，这是为了代码的可移植性)
# ==============================================================================
# 在此处选择你的硬件: 'cuda' (NVIDIA), 'mps' (Apple Silicon), or 'cpu'
print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
print("CUDA版本:", torch.version.cuda if torch.cuda.is_available() else "N/A")
if torch.cuda.is_available():
    print("GPU数量:", torch.cuda.device_count())
    print("当前GPU:", torch.cuda.get_device_name(0))
   # 更智能的设备选择方式
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"✓ 检测到Apple Silicon, 将使用MPS进行计算")
else:
    device = torch.device("cpu")
    print(f"✓ 检测到CPU, 将使用CPU进行计算")
# ==============================================================================
# 2. 系统与模型参数 (依据: 论文全文)
# ==============================================================================
# --- 通信系统参数 ---
BIT_RATE = 100e9  # 100 Gb/s (依据:)
BAUD_RATE = BIT_RATE / 2  # PAM4 每个符号2个bit
RRC_ROLLOFF = 0.1  # RRC滤波器滚降系数 (依据:)
SSMF_LENGTH = 15e3  # 15 km SSMF (依据:)
DML_3DB_BW = 16e9  # DML 3dB 带宽 16 GHz (依据:)

# --- 训练参数 ---
TRAIN_SYMBOLS = 20000  # 训练符号数 (依据:)
VALID_SYMBOLS = 10000  # 验证符号数 (依据:)
TEST_SYMBOLS = 1200000  # 测试符号数 (依据:)
TOTAL_SYMBOLS = TRAIN_SYMBOLS + VALID_SYMBOLS + TEST_SYMBOLS

# --- Cascade RNN 模型参数 (依据: Fig. 5(a),) ---
# 论文中对比了多种尺寸, 这里我们使用性能最佳的 (23,12)
N0 = 23  # 输入层神经元数量 (n^[0]) (依据:)
N1 = 12  # 隐藏层神经元数量 (n^[1]) (依据:)
K_DELAY = 2  # 延迟数量 k (依据:)


# ==============================================================================
# 3. 简化信道仿真 (依据: 论文对信道损伤的描述)
# ==============================================================================
# 这是一个简化的信道模型，旨在捕捉论文中提到的关键物理效应，
# 包括DML带宽限制、光纤色散和光电探测器的平方率检测。
def simulate_pam4_channel(symbols, length, baud_rate, dml_bw, add_noise=True):
    """
    模拟一个简化的IM/DD PAM4带限信道
    依据: 系统损伤主要来自DML带宽限制, CD与平方率检测的混合
    """
    # PAM4 幅度电平
    amplitude_levels = [-3, -1, 1, 3]
    tx_signal = np.array([amplitude_levels[s] for s in symbols])

    # 1. 带宽限制 (模拟 DML 和其他器件)
    # 使用一个简单的低通滤波器来模拟系统的端到端响应
    from scipy.signal import butter, lfilter
    nyquist = 0.5 * baud_rate
    cutoff = dml_bw / nyquist
    b, a = butter(4, min(0.99, cutoff), btype='low')
    bw_limited_signal = lfilter(b, a, tx_signal)

    # 2. 模拟色散与平方率检测引入的非线性 (简化)
    # 真实效应是 E_out = F_inv(F(E_in) * H_CD)。 E_rx_elec ~ |E_out|^2
    # 这里我们用一个简单的非线性函数来近似这种效应，比如 Volterra 的一部分
    # y[n] = a1*x[n] + a2*x[n-1] + b1*x[n]^2 + b2*x[n]*x[n-1]
    # 这里我们只引入一个简单的记忆非线性效应
    nonlinear_signal = bw_limited_signal.copy()
    for i in range(1, len(nonlinear_signal)):
        # 引入码间干扰和非线性项
        nonlinear_signal[i] += 0.1 * nonlinear_signal[i - 1] - 0.05 * nonlinear_signal[i] * nonlinear_signal[i - 1]

    # 3. 添加高斯白噪声 (AWGN)
    if add_noise:
        signal_power = np.mean(nonlinear_signal ** 2)
        # SNR可以根据ROP进行调整，这里设为一个固定值以供演示
        snr_db = 20
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(nonlinear_signal))
        received_signal = nonlinear_signal + noise
    else:
        received_signal = nonlinear_signal

    # 归一化
    received_signal = received_signal / np.std(received_signal)

    return received_signal


# ==============================================================================
# 4. Cascade RNN 模型定义 (依据: Fig. 1, Eq. (1), (2))
# ==============================================================================
class CascadeRNN(nn.Module):
    def __init__(self, n0, n1, k):
        # 依据: 模型结构在Fig. 1 和 Eq. (1), (2) 中描述
        super(CascadeRNN, self).__init__()
        self.n0 = n0
        self.n1 = n1
        self.k = k

        # 隐藏层 (1st layer)
        # 权重 w^[1] 和 w^d
        self.hidden_weights = nn.Linear(n0 + k, n1, bias=True)
        # 激活函数 f^[1] (tanh) (依据:)
        self.hidden_activation = nn.Tanh()

        # 输出层 (2nd layer)
        # 权重 w^[2] 和 w^c
        # 输入来自: 隐藏层(n1), 原始输入(n0), 延迟输出(k)
        self.output_weights = nn.Linear(n1 + n0 + k, 1, bias=True)
        # 激活函数 f^[2] (linear) (依据:)
        # PyTorch的nn.Linear默认就是线性输出，所以不需要额外激活函数

    def forward(self, x, y_d):
        # x: 当前输入窗口, shape [batch_size, n0]
        # y_d: 延迟的输出, shape [batch_size, k]

        # Eq. (1): H^[1] = f^[1]([w^[1],w^d][X^T,Y_d^T]^T + b^[1]) (依据:)
        hidden_input = torch.cat((x, y_d), dim=1)
        h1 = self.hidden_activation(self.hidden_weights(hidden_input))

        # Eq. (2): y = f^[2]([w^[2]},w^c][H^[1]T,X^T,Y_d^T]^T + b^[2]) (依据:)
        # [H^[1]T, X^T, Y_d^T]^T 对应于将三个张量连接起来
        output_input = torch.cat((h1, x, y_d), dim=1)
        y = self.output_weights(output_input)

        return y.squeeze(-1)


# ==============================================================================
# 5. 数据准备 (依据:)
# ==============================================================================
def prepare_data(total_symbols, n0, k):
    print("正在生成和处理数据...")
    # 生成原始 PAM4 符号 {0, 1, 2, 3}
    original_indices = np.random.randint(0, 4, total_symbols)
    pam4_map = np.array([-3, -1, 1, 3])
    original_symbols = pam4_map[original_indices]

    # 通过仿真信道
    received_signal = simulate_pam4_channel(original_indices, SSMF_LENGTH, BAUD_RATE, DML_3DB_BW)

    # 创建输入窗口 X 和目标 y
    # 依据: 当前符号, (n0-1)/2 过去和 (n0-1)/2 未来符号作为输入 (依据:)
    # 目标 y 是窗口中心的原始符号
    half_window = (n0 - 1) // 2

    X = []
    Y_target = []
    # 为了处理窗口和延迟的边界，我们从 (half_window + k) 开始
    for i in range(half_window + k, len(received_signal) - half_window):
        X.append(received_signal[i - half_window: i + half_window + 1])
        Y_target.append(original_symbols[i])

    X = np.array(X)
    Y_target = np.array(Y_target)

    # 划分数据集
    # 依据: 2万训练, 1万验证, 120万测试 (依据:)
    # 这里我们按比例划分
    num_samples = len(X)
    train_end = int(TRAIN_SYMBOLS / TOTAL_SYMBOLS * num_samples)
    valid_end = train_end + int(VALID_SYMBOLS / TOTAL_SYMBOLS * num_samples)

    X_train, Y_train_target = X[:train_end], Y_target[:train_end]
    X_valid, Y_valid_target = X[train_end:valid_end], Y_target[train_end:valid_end]
    X_test, Y_test_target = X[valid_end:], Y_target[valid_end:]

    print(f"数据准备完成. 训练集: {len(X_train)}, 验证集: {len(X_valid)}, 测试集: {len(X_test)}")
    return (torch.FloatTensor(d).to(device) for d in
            [X_train, Y_train_target, X_valid, Y_valid_target, X_test, Y_test_target])


# ==============================================================================
# 6. 训练与测试流程 (依据: 论文第三页的描述)
# ==============================================================================
def train_model(model, X_train, Y_train_target, X_valid, Y_valid_target, k):
    # 损失函数: MSE (Mean Square Error) (依据:)
    criterion = nn.MSELoss()
    # 优化器: 论文用LMA, 但在PyTorch中不常用。
    # 我们使用Adam，它在实践中对于NN训练非常高效且稳定。
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early stopping: 验证检查失败x次则停止 (依据:)
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    patience = 10  # 对应论文中的x次
    TrainingtIimes=1000
    print("\n开始训练模型...")
    for epoch in range(TrainingtIimes):  # 最多训练100个epoch
        model.train()

        # 在训练中，延迟输入 Y_d 来自于真实的目标符号 (teacher forcing)
        # 这是训练RNN的常用技巧，可以稳定和加速收敛
        Y_d_train = torch.zeros(len(X_train), k, device=device)
        for i in range(k):
            Y_d_train[:, i] = torch.roll(Y_train_target, shifts=i + 1, dims=0)

        # mini-batch training
        train_loader = DataLoader(TensorDataset(X_train, Y_d_train, Y_train_target), batch_size=256, shuffle=True)

        for batch_x, batch_yd, batch_yt in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x, batch_yd)
            loss = criterion(outputs, batch_yt)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            Y_d_valid = torch.zeros(len(X_valid), k, device=device)
            for i in range(k):
                Y_d_valid[:, i] = torch.roll(Y_valid_target, shifts=i + 1, dims=0)

            valid_outputs = model(X_valid, Y_d_valid)
            valid_loss = criterion(valid_outputs, Y_valid_target)

        print(f"Epoch {epoch + 1}, 训练损失: {loss.item():.6f}, 验证损失: {valid_loss.item():.6f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print("验证损失连续 %d 个epoch未改善，提前停止训练。" % patience)
            break

    model.load_state_dict(torch.load("best_model.pth"))
    return model


# (请将此函数替换掉原来代码中的 test_model 函数)

def test_model(model, X_test, Y_test_target, k):
    print("\n开始测试模型 (GPU优化版)...")
    model.eval()  # 将模型设置为评估模式

    # --- 设备和数据类型设置 ---
    # 确保所有计算都在指定的GPU设备上进行
    pam4_levels = torch.tensor([-3, -1, 1, 3], device=device, dtype=torch.float32)

    # --- 使用DataLoader进行高效批处理 ---
    # 依据: 这是在PyTorch中处理大规模数据集的标准做法，可以实现高效的内存管理和数据加载
    batch_size = 2048  # 可以根据你的GPU显存大小调整这个值
    test_dataset = TensorDataset(X_test, Y_test_target)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_errors = 0
    total_symbols = 0

    # 初始化延迟寄存器 (Y_d_test)，用于存储整个序列的前序判决结果
    # 形状为 [1, k]，代表一个数据流的k个延迟
    Y_d_test = torch.zeros(1, k, device=device, dtype=torch.float32)

    # 使用 torch.no_grad() 来关闭梯度计算，这会加速计算并减少内存使用
    # 依据: 在推理（测试）阶段，我们不需要计算梯度，关闭它可以带来显著的性能提升
    with torch.no_grad():
        # 外层循环遍历数据批次
        for batch_x, batch_y_target in tqdm(test_loader, desc="BER 计算 (批处理)"):

            # 内层循环处理批次内的每一个样本，以维持自回归依赖
            # 这个循环在批次大小的维度上进行，而不是整个数据集
            for i in range(batch_x.shape[0]):
                # 从批次中获取当前符号的输入特征
                # .unsqueeze(0) 是为了给它增加一个 batch 维度 (1, n0) 以匹配模型输入
                current_x = batch_x[i].unsqueeze(0)

                # --- 核心计算步骤（在GPU上并行执行） ---
                # 虽然Python循环是串行的，但model()内部的所有矩阵运算
                # 都在GPU上高度并行地执行。
                output = model(current_x, Y_d_test)

                # 判决过程 (也在GPU上执行)
                # 使用广播机制并行计算到4个电平的距离
                distances = torch.abs(output.view(-1, 1) - pam4_levels)
                # 找到距离最小的电平的索引
                decision_index = torch.argmin(distances, dim=1)
                # 根据索引获取判决结果
                decision = pam4_levels[decision_index]

                # 更新延迟寄存器
                # torch.roll 在GPU上高效执行
                Y_d_test = torch.roll(Y_d_test, shifts=1, dims=1)
                Y_d_test[0, 0] = decision

                # 比较并计算误符号数 (在GPU上比较，结果移回CPU累加)
                if decision != batch_y_target[i]:
                    total_errors += 1

            total_symbols += batch_x.shape[0]

    symbol_error_rate = total_errors / total_symbols
    # 对于格雷编码的PAM4，误比特率约等于误符号率的一半
    bit_error_rate = symbol_error_rate / 2

    print(f"测试完成. 总符号数: {total_symbols}, 错误符号数: {total_errors}")
    print(f"误符号率 (SER): {symbol_error_rate:.2e}")
    print(f"误比特率 (BER): {bit_error_rate:.2e}")


# ==============================================================================
# 7. 主函数
# ==============================================================================
if __name__ == '__main__':
    # 准备数据
    X_train, Y_train_target, X_valid, Y_valid_target, X_test, Y_test_target = prepare_data(TOTAL_SYMBOLS, N0, K_DELAY)

    # 初始化模型
    model = CascadeRNN(n0=N0, n1=N1, k=K_DELAY).to(device)

    # 训练模型
    trained_model = train_model(model, X_train, Y_train_target, X_valid, Y_valid_target, K_DELAY)

    # 测试模型
    test_model(trained_model, X_test, Y_test_target, K_DELAY)