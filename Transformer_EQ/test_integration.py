#!/usr/bin/env python3
"""
测试Python与MATLAB联合仿真的数据加载功能
"""

import os
import numpy as np
from vpi_data_loader import VPIDataLoader


def create_test_data():
    """创建测试数据文件"""
    print("创建测试数据文件...")
    
    # 创建测试VPI数据
    vpi_data = np.random.randn(589824)  # 模拟VPI数据
    np.savetxt('vpi_data.txt', vpi_data, fmt='%.6f')
    print("✓ 创建 vpi_data.txt")
    
    # 创建测试符号数据
    pam4_levels = [-3, -1, 1, 3]
    
    # 训练集符号
    train_symbols = np.random.choice(pam4_levels, 65536)
    np.savetxt('Transformer_EQ/symb_train.txt', train_symbols, fmt='%d')
    print("✓ 创建 symb_train.txt")
    
    # 验证集符号
    val_symbols = np.random.choice(pam4_levels, 16384)
    np.savetxt('Transformer_EQ/symb_val.txt', val_symbols, fmt='%d')
    print("✓ 创建 symb_val.txt")
    
    # 测试集符号
    test_symbols = np.random.choice(pam4_levels, 65536)
    np.savetxt('Transformer_EQ/symb_test.txt', test_symbols, fmt='%d')
    print("✓ 创建 symb_test.txt")


def test_data_loading():
    """测试数据加载功能"""
    print("\n测试数据加载功能...")
    
    try:
        loader = VPIDataLoader()
        
        # 测试VPI数据加载
        vpi_data = loader.load_vpi_data()
        print(f"✓ VPI数据加载成功，长度: {len(vpi_data)}")
        
        # 测试符号数据加载
        symbol_data = loader.load_symbol_data()
        print(f"✓ 符号数据加载成功")
        for split, data in symbol_data.items():
            print(f"  {split}: {len(data)} 个符号")
        
        # 测试数据划分
        rx_train, rx_val, rx_test = loader.split_vpi_data(vpi_data)
        print(f"✓ VPI数据划分成功")
        print(f"  训练集: {len(rx_train)} 个采样点")
        print(f"  验证集: {len(rx_val)} 个采样点")
        print(f"  测试集: {len(rx_test)} 个采样点")
        
        # 测试序列准备
        train_rx, train_tx = loader.prepare_sequences(
            rx_train, symbol_data['train'], seq_len=128
        )
        print(f"✓ 序列准备成功")
        print(f"  训练序列形状: {train_rx.shape}")
        print(f"  训练标签形状: {train_tx.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据加载测试失败: {e}")
        return False


def test_data_loaders():
    """测试数据加载器"""
    print("\n测试数据加载器...")
    
    try:
        loader = VPIDataLoader()
        train_loader, val_loader, test_loader = loader.get_data_loaders(
            seq_len=128, batch_size=32
        )
        
        print(f"✓ 数据加载器创建成功")
        print(f"  训练集: {len(train_loader.dataset)} 个序列")
        print(f"  验证集: {len(val_loader.dataset)} 个序列")
        print(f"  测试集: {len(test_loader.dataset)} 个序列")
        
        # 测试一个批次
        for rx_batch, tx_batch in train_loader:
            print(f"✓ 批次测试成功")
            print(f"  RX批次形状: {rx_batch.shape}")
            print(f"  TX批次形状: {tx_batch.shape}")
            print(f"  RX数据范围: [{rx_batch.min():.3f}, {rx_batch.max():.3f}]")
            print(f"  TX标签范围: [{tx_batch.min()}, {tx_batch.max()}]")
            break
        
        return True
        
    except Exception as e:
        print(f"✗ 数据加载器测试失败: {e}")
        return False


def cleanup_test_data():
    """清理测试数据文件"""
    print("\n清理测试数据文件...")
    
    test_files = [
        'vpi_data.txt',
        'Transformer_EQ/symb_train.txt',
        'Transformer_EQ/symb_val.txt',
        'Transformer_EQ/symb_test.txt'
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✓ 删除 {file_path}")


def main():
    """主测试函数"""
    print("=== Python与MATLAB联合仿真数据加载测试 ===\n")
    
    # 创建测试数据
    create_test_data()
    
    # 测试数据加载
    success1 = test_data_loading()
    
    # 测试数据加载器
    success2 = test_data_loaders()
    
    # 清理测试数据
    cleanup_test_data()
    
    # 总结
    print("\n=== 测试结果 ===")
    if success1 and success2:
        print("✓ 所有测试通过！数据加载功能正常")
        print("\n下一步:")
        print("1. 运行MATLAB发射机程序生成真实符号数据")
        print("2. 使用VPI Photonics生成真实信道数据")
        print("3. 运行Python主程序进行模型训练")
    else:
        print("✗ 部分测试失败，请检查代码")
    
    print("\n=== 测试完成 ===")


if __name__ == '__main__':
    main()
