import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import math
import platform

# ==============================================================================
# 1. 环境与系统参数
# ==============================================================================
# --- 环境配置 ---
print("PyTorch版本:", torch.__version__)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✓ 检测到GPU: {torch.cuda.get_device_name(0)}, 将使用CUDA进行计算")
elif torch.backends.mps.is_available() and platform.processor() == 'arm':
    device = torch.device("mps")
    print(f"✓ 检测到Apple Silicon, 将使用MPS进行计算")
else:
    device = torch.device("cpu")
    print(f"✓ 未检测到GPU, 将使用CPU进行计算")

# --- 通信系统参数 ---
BIT_RATE = 100e9
BAUD_RATE = BIT_RATE / 2
SSMF_LENGTH = 15e3
DML_3DB_BW = 16e9

# --- 训练参数 ---
TRAIN_SYMBOLS = 200000
VALID_SYMBOLS = 50000
TEST_SYMBOLS = 1200000
TOTAL_SYMBOLS = TRAIN_SYMBOLS + VALID_SYMBOLS + TEST_SYMBOLS

# ==============================================================================
# 2. Transformer模型参数
# ==============================================================================
SEQ_LEN = 256
D_MODEL = 128
N_HEAD = 8
NUM_ENCODER_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1


# ==============================================================================
# 3. 简化信道仿真
# ==============================================================================
def simulate_pam4_channel(symbols, length, baud_rate, dml_bw, add_noise=True):
    amplitude_levels = [-3, -1, 1, 3]
    tx_signal = np.array([amplitude_levels[s] for s in symbols])
    from scipy.signal import butter, lfilter
    nyquist = 0.5 * baud_rate
    cutoff = dml_bw / nyquist
    b, a = butter(4, min(0.99, cutoff), btype='low')
    bw_limited_signal = lfilter(b, a, tx_signal)
    nonlinear_signal = bw_limited_signal.copy()
    for i in range(1, len(nonlinear_signal)):
        nonlinear_signal[i] += 0.1 * nonlinear_signal[i - 1] - 0.05 * nonlinear_signal[i] * nonlinear_signal[i - 1]
    if add_noise:
        signal_power = np.mean(nonlinear_signal ** 2)
        snr_db = 20
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(nonlinear_signal))
        received_signal = nonlinear_signal + noise
    else:
        received_signal = nonlinear_signal
    return received_signal / np.std(received_signal)


# ==============================================================================
# 4. Transformer 模型定义
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEqualizer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerEqualizer, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, src):
        src = src.unsqueeze(-1)
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.output_projection(output)
        return output.squeeze(-1)


# ==============================================================================
# 5. 数据准备
# ==============================================================================
def prepare_data_transformer(total_symbols, seq_len, stride=128):
    print("正在为Transformer生成和处理 [重叠] 数据...")
    original_indices = np.random.randint(0, 4, total_symbols)
    pam4_map = np.array([-3, -1, 1, 3])
    original_symbols = pam4_map[original_indices]
    received_signal = simulate_pam4_channel(original_indices, SSMF_LENGTH, BAUD_RATE, DML_3DB_BW)

    num_sequences = (len(received_signal) - seq_len) // stride + 1
    X = np.zeros((num_sequences, seq_len))
    Y_target = np.zeros((num_sequences, seq_len))

    for i in range(num_sequences):
        start_idx = i * stride
        end_idx = start_idx + seq_len
        X[i, :] = received_signal[start_idx:end_idx]
        Y_target[i, :] = original_symbols[start_idx:end_idx]

    num_samples = len(X)
    train_end = int(TRAIN_SYMBOLS / TOTAL_SYMBOLS * num_samples)
    valid_end = train_end + int(VALID_SYMBOLS / TOTAL_SYMBOLS * num_samples)
    X_train, Y_train_target = X[:train_end], Y_target[:train_end]
    X_valid, Y_valid_target = X[train_end:valid_end], Y_target[train_end:valid_end]
    X_test, Y_test_target = X[valid_end:], Y_target[valid_end:]

    print(
        f"数据准备完成. 序列长度: {seq_len}, 步长: {stride}, 训练集: {len(X_train)}条, 验证集: {len(X_valid)}条, 测试集: {len(X_test)}条")
    return (torch.FloatTensor(d).to(device) for d in
            [X_train, Y_train_target, X_valid, Y_valid_target, X_test, Y_test_target])


# ==============================================================================
# 6. 训练与测试流程
# ==============================================================================
def train_model_transformer(model, X_train, Y_train_target, X_valid, Y_valid_target):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # [修正] 移除了不支持的 'verbose=True' 参数
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2)

    best_valid_loss = float('inf')
    epochs_no_improve = 0
    patience = 5

    train_loader = DataLoader(TensorDataset(X_train, Y_train_target), batch_size=128, shuffle=True)
    valid_loader = DataLoader(TensorDataset(X_valid, Y_valid_target), batch_size=128, shuffle=False)

    print("\n开始训练 [修正后的] Transformer模型...")
    for epoch in range(50):
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1} [训练]"):
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in tqdm(valid_loader, desc=f"Epoch {epoch + 1} [验证]"):
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                total_valid_loss += loss.item()
        avg_valid_loss = total_valid_loss / len(valid_loader)

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1}, 平均训练损失: {avg_train_loss:.6f}, 平均验证损失: {avg_valid_loss:.6f}, 当前学习率: {current_lr:.6f}")

        scheduler.step(avg_valid_loss)

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "transformer_equalizer_v2.pth")
            print(f"  验证损失改善，模型已保存至 transformer_equalizer_v2.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"验证损失连续 {patience} 个epoch未改善，提前停止训练。")
            break

    model.load_state_dict(torch.load("transformer_equalizer_v2.pth"))
    return model


def test_model_transformer(model, X_test, Y_test_target):
    print("\n开始测试 [修正后的] Transformer模型 (并行计算模式)...")
    model.eval()
    test_loader = DataLoader(TensorDataset(X_test, Y_test_target), batch_size=256, shuffle=False)
    pam4_levels = torch.tensor([-3, -1, 1, 3], device=device, dtype=torch.float32)
    total_errors, total_symbols = 0, 0

    with torch.no_grad():
        for batch_x, batch_y_target in tqdm(test_loader, desc="BER 计算"):
            equalized_output = model(batch_x)
            distances = torch.abs(equalized_output.unsqueeze(-1) - pam4_levels)
            decision_indices = torch.argmin(distances, dim=2)
            decisions = pam4_levels[decision_indices]
            errors = torch.sum(decisions != batch_y_target)
            total_errors += errors.item()
            total_symbols += batch_y_target.numel()

    symbol_error_rate = total_errors / total_symbols
    bit_error_rate = symbol_error_rate / 2
    print(f"测试完成. 总符号数: {total_symbols}, 错误符号数: {total_errors}")
    print(f"误符号率 (SER): {symbol_error_rate:.2e}")
    print(f"误比特率 (BER): {bit_error_rate:.2e}")


# ==============================================================================
# 7. 主函数
# ==============================================================================
if __name__ == '__main__':
    X_train, Y_train_target, X_valid, Y_valid_target, X_test, Y_test_target = prepare_data_transformer(TOTAL_SYMBOLS,
                                                                                                       SEQ_LEN)
    model = TransformerEqualizer(
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)

    trained_model = train_model_transformer(model, X_train, Y_train_target, X_valid, Y_valid_target)
    test_model_transformer(trained_model, X_test, Y_test_target)