import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os  # 引入os模块用于文件路径操作

from pam4_system import PAM4System
from channel import OpticalChannel
from model import TransformerEqualizer
from utils import calculate_ber


def generate_data(system, channel, num_symbols, seq_len):
    """生成PAM4信号数据"""
    all_rx_symbols = []
    all_tx_indices = []

    num_sequences = num_symbols // seq_len
    print(f"  Generating {num_sequences} sequences...")
    for _ in range(num_sequences):
        tx_indices = np.random.randint(0, 4, seq_len)
        tx_signal, _ = system.transmit(tx_indices)
        # plt.plot(tx_signal, '.');
        # plt.show()


        rx_signal = channel.propagate(tx_signal)
        rx_symbols = system.receive(rx_signal, seq_len)

        if len(rx_symbols) == seq_len:
            all_rx_symbols.append(rx_symbols)
            all_tx_indices.append(tx_indices)

    return torch.FloatTensor(np.array(all_rx_symbols)), torch.LongTensor(np.array(all_tx_indices))


def evaluate(model, data_loader, device):
    """在给定数据集上评估模型性能"""
    model.eval()  # 将模型设置为评估模式
    all_preds = []
    all_true = []
    with torch.no_grad():  # 在评估时不计算梯度
        for rx_batch, tx_batch in data_loader:
            rx_batch = rx_batch.to(device)
            logits = model(rx_batch)
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_true.append(tx_batch.numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_true = np.concatenate(all_true).flatten()
    ber = calculate_ber(all_preds, all_true)
    return ber


def main():
    # --- 超参数与设置 ---
    SEQ_LEN = 128
    D_MODEL = 64
    N_LAYERS = 4
    N_HEAD = 8
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    SNR_DB = 10
    MODEL_SAVE_PATH = "best_transformer_equalizer.pth"

    # --- 设备自适应选择 ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Backend: NVIDIA CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Backend: Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        print("Backend: CPU")

    # --- 系统与模型初始化 ---
    pam4_system = PAM4System(sps=8)
    channel = OpticalChannel(snr_db=SNR_DB)
    model = TransformerEqualizer(
        d_model=D_MODEL, n_layers=N_LAYERS, n_head=N_HEAD, n_position=SEQ_LEN
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 数据集生成 ---
    print("\n--- Generating Datasets ---")
    print("Generating training data...")
    train_rx, train_tx = generate_data(pam4_system, channel, num_symbols=500_000, seq_len=SEQ_LEN)
    train_dataset = TensorDataset(train_rx, train_tx)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Generating validation data...")
    val_rx, val_tx = generate_data(pam4_system, channel, num_symbols=50_000, seq_len=SEQ_LEN)
    val_dataset = TensorDataset(val_rx, val_tx)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print("Generating test data...")
    test_rx, test_tx = generate_data(pam4_system, channel, num_symbols=500_000, seq_len=SEQ_LEN)
    test_dataset = TensorDataset(test_rx, test_tx)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    print("---------------------------\n")

    # --- 训练循环 ---
    best_val_ber = float('inf')  # 初始化一个无穷大的最佳BER
    print("--- Starting Training ---")
    for epoch in range(EPOCHS):
        model.train()  # 将模型设置为训练模式
        total_loss = 0
        for rx_batch, tx_batch in train_loader:
            rx_batch, tx_batch = rx_batch.to(device), tx_batch.to(device)

            optimizer.zero_grad()

            logits = model(rx_batch)
            loss = criterion(logits.view(-1, 4), tx_batch.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # --- 每个epoch后进行验证 ---
        val_ber = evaluate(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Validation BER: {val_ber:.6f}")

        # --- 保存性能最好的模型 ---
        if val_ber < best_val_ber:
            best_val_ber = val_ber
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved with BER: {best_val_ber:.6f}")

    print("--- Training Finished ---\n")

    # --- 最终测试 ---
    print("--- Starting Final Test ---")
    if os.path.exists(MODEL_SAVE_PATH):
        # 加载在验证集上表现最好的模型权重
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print("Loaded best model from validation for final testing.")

        # 在独立的测试集上进行最终评估
        test_ber = evaluate(model, test_loader, device)

        print(f"\nFinal Test BER on unseen data: {test_ber:.6f}")
    else:
        print("No saved model found to test.")
    print("---------------------------\n")


if __name__ == '__main__':
    main()