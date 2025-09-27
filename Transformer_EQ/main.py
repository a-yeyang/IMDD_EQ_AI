import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import matplotlib
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os  # 引入os模块用于文件路径操作

# 设置matplotlib支持中文显示和高质量绘图
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置高质量绘图参数
matplotlib.rcParams['figure.dpi'] = 300  # 设置图像分辨率
matplotlib.rcParams['savefig.dpi'] = 300  # 保存图像分辨率
matplotlib.rcParams['savefig.bbox'] = 'tight'  # 紧凑布局
matplotlib.rcParams['savefig.pad_inches'] = 0.1  # 边距
matplotlib.rcParams['figure.autolayout'] = True  # 自动调整布局

# 设置字体大小
matplotlib.rcParams['font.size'] = 14  # 基础字体大小
matplotlib.rcParams['axes.titlesize'] = 18  # 标题字体大小
matplotlib.rcParams['axes.labelsize'] = 16  # 坐标轴标签字体大小
matplotlib.rcParams['xtick.labelsize'] = 14  # x轴刻度字体大小
matplotlib.rcParams['ytick.labelsize'] = 14  # y轴刻度字体大小
matplotlib.rcParams['legend.fontsize'] = 14  # 图例字体大小

# 设置线条和标记
matplotlib.rcParams['lines.linewidth'] = 2.5  # 线条宽度
matplotlib.rcParams['lines.markersize'] = 10  # 标记大小
matplotlib.rcParams['grid.linewidth'] = 0.8  # 网格线宽度
matplotlib.rcParams['axes.linewidth'] = 1.2  # 坐标轴线宽度

# 设置颜色和样式
matplotlib.rcParams['axes.grid'] = True  # 显示网格
matplotlib.rcParams['grid.alpha'] = 0.3  # 网格透明度
matplotlib.rcParams['axes.facecolor'] = 'white'  # 背景色
matplotlib.rcParams['figure.facecolor'] = 'white'  # 图形背景色

from pam4_system import PAM4System
from channel import OpticalChannel
from model import TransformerEqualizer
from utils import calculate_ber, calculate_ser


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


        rx_signal = channel.channel1(tx_signal)
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
    ser = calculate_ser(all_preds, all_true)
    return ser


def train_model(pam4_system, device, seq_len, d_model, n_layers, n_head, epochs, batch_size, learning_rate, train_snr_db):
    """训练模型函数"""
    print(f"\n--- 开始训练 (训练信噪比: {train_snr_db} dB) ---")
    
    # 使用固定信噪比创建训练和验证信道
    train_channel = OpticalChannel(snr_db=train_snr_db)
    
    # 模型初始化
    model = TransformerEqualizer(
        d_model=d_model, n_layers=n_layers, n_head=n_head, n_position=seq_len
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 生成训练和验证数据
    print("生成训练数据...")
    train_rx, train_tx = generate_data(pam4_system, train_channel, num_symbols=500_000, seq_len=seq_len)
    train_dataset = TensorDataset(train_rx, train_tx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print("生成验证数据...")
    val_rx, val_tx = generate_data(pam4_system, train_channel, num_symbols=50_000, seq_len=seq_len)
    val_dataset = TensorDataset(val_rx, val_tx)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 训练循环
    best_val_ser = float('inf')
    model_save_path = "best_transformer_equalizer.pth"
    
    for epoch in range(epochs):
        model.train()
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
        val_ser = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Validation SER: {val_ser:.6f}")
        
        if val_ser < best_val_ser:
            best_val_ser = val_ser
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> 保存最佳模型，SER: {best_val_ser:.6f}")
    
    print("--- 训练完成 ---\n")
    return model_save_path


def test_different_snr(pam4_system, device, seq_len, batch_size, model_path, snr_range):
    """在不同信噪比下测试模型性能"""
    print("--- 开始多信噪比测试 ---")
    
    # 加载训练好的模型
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return None, None
    
    model = TransformerEqualizer(
        d_model=64, n_layers=4, n_head=8, n_position=seq_len
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    snr_values = []
    ser_values = []
    
    for snr_db in snr_range:
        print(f"测试信噪比: {snr_db} dB")
        
        # 为当前信噪比创建信道
        test_channel = OpticalChannel(snr_db=snr_db)
        
        # 生成测试数据
        test_rx, test_tx = generate_data(pam4_system, test_channel, num_symbols=100_000, seq_len=seq_len)
        test_dataset = TensorDataset(test_rx, test_tx)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # 评估模型性能
        ser = evaluate(model, test_loader, device)
        
        snr_values.append(snr_db)
        ser_values.append(ser)
        
        print(f"  信噪比 {snr_db} dB: SER = {ser:.6f}")
    
    print("--- 多信噪比测试完成 ---\n")
    return snr_values, ser_values


def plot_ser_vs_snr(snr_values, ser_values, save_path="ser_vs_snr.png", 
                   title_fontsize=20, label_fontsize=16, tick_fontsize=14, 
                   annotation_fontsize=12, linewidth=3, markersize=12):
    """绘制误符号率-信噪比关系图 - 高质量版本"""
    
    # 创建高质量图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制主曲线
    line = ax.semilogy(snr_values, ser_values, 'bo-', 
                      linewidth=linewidth, markersize=markersize, 
                      markerfacecolor='blue', markeredgecolor='darkblue',
                      markeredgewidth=1.5, alpha=0.8)
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('信噪比 (dB)', fontsize=label_fontsize, fontweight='bold')
    ax.set_ylabel('误符号率 (SER)', fontsize=label_fontsize, fontweight='bold')
    ax.set_title('Transformer均衡器性能：误符号率 vs 信噪比', 
                fontsize=title_fontsize, fontweight='bold', pad=20)
    
    # 设置坐标轴范围
    ax.set_xlim(min(snr_values) - 0.5, max(snr_values) + 0.5)
    
    # 设置网格
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)  # 将网格放在数据线后面
    
    # 设置坐标轴刻度
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, 
                  width=1.2, length=6)
    ax.tick_params(axis='both', which='minor', width=0.8, length=3)
    
    # 设置坐标轴边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # 添加数据点标注（选择性标注，避免过于拥挤）
    step = max(1, len(snr_values) // 8)  # 最多显示8个标注
    for i in range(0, len(snr_values), step):
        snr, ser = snr_values[i], ser_values[i]
        ax.annotate(f'{ser:.1e}', (snr, ser), 
                   textcoords="offset points", xytext=(0, 15), 
                   ha='center', fontsize=annotation_fontsize,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                           edgecolor='gray', alpha=0.8))
    
    # 优化布局
    plt.tight_layout()
    
    # 保存高质量图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none', 
               pad_inches=0.2, format='png')
    
    # 显示图像
    plt.show()
    print(f"高质量误符号率-信噪比关系图已保存为: {save_path}")
    print(f"图像分辨率: 300 DPI, 尺寸: 12x8 英寸")


def main():
    # --- 超参数与设置 ---
    SEQ_LEN = 128
    D_MODEL = 64
    N_LAYERS = 4
    N_HEAD = 8
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    TRAIN_SNR_DB = 15  # 训练时的固定信噪比
    TEST_SNR_RANGE = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # 测试时的信噪比范围
    
    # --- 设备自适应选择 ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("后端: NVIDIA CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("后端: Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        print("后端: CPU")
    
    # --- 系统初始化 ---
    pam4_system = PAM4System(sps=8)
    
    # --- 训练阶段 ---
    model_path = train_model(
        pam4_system=pam4_system,
        device=device,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_head=N_HEAD,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        train_snr_db=TRAIN_SNR_DB
    )
    
    # --- 测试阶段 ---
    snr_values, ser_values = test_different_snr(
        pam4_system=pam4_system,
        device=device,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        model_path=model_path,
        snr_range=TEST_SNR_RANGE
    )
    
    # --- 绘制结果图 ---
    if snr_values and ser_values:
        # 使用高质量绘图参数
        plot_ser_vs_snr(snr_values, ser_values, 
                       title_fontsize=22,    # 标题字体大小
                       label_fontsize=18,    # 坐标轴标签字体大小
                       tick_fontsize=16,     # 刻度字体大小
                       annotation_fontsize=14, # 标注字体大小
                       linewidth=3.5,        # 线条宽度
                       markersize=14)        # 标记大小
        
        # 打印结果摘要
        print("\n=== 测试结果摘要 ===")
        for snr, ser in zip(snr_values, ser_values):
            print(f"信噪比 {snr:2d} dB: SER = {ser:.2e}")
        print("===================")
    else:
        print("测试失败，无法生成结果图")


if __name__ == '__main__':
    main()