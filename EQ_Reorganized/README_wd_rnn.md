# WD-RNN 均衡器使用说明

## 文件说明

### 核心函数
- `wd_rnn.m` - WD-RNN模型训练函数
- `test_wdrnn.m` - WD-RNN模型测试函数
- `demo_wd_rnn.m` - 完整演示脚本

### 依赖文件
- `CMA.m` - 用于对比的CMA均衡器（位于 utils/ 目录）

## 快速开始

### 1. 运行演示
```matlab
% 在MATLAB中运行完整演示
demo_wd_rnn
```

### 2. 单独使用训练函数
```matlab
% 准备训练数据
rx_sym_train = your_received_symbols;  % 接收信号 (列向量)
symb_train = your_true_symbols;        % 真实符号 (列向量)

% 训练模型
modelFile = wd_rnn(rx_sym_train, symb_train);
fprintf('模型已保存至: %s\n', modelFile);
```

### 3. 单独使用测试函数
```matlab
% 准备测试数据
rx_sym_test = your_test_received_symbols;  % 测试接收信号
symb_test = your_test_true_symbols;        % 测试真实符号

% 加载模型并测试
[SER, BER, eqOut, predLevels] = test_wdrnn('wd_rnn_model.mat', rx_sym_test, symb_test);

fprintf('符号误差率: %.2e\n', SER);
fprintf('比特误差率: %.2e\n', BER);
```

## 函数接口

### wd_rnn() - 训练函数
```matlab
modelFile = wd_rnn(rx_sym_train, symb_train)
```
**输入参数:**
- `rx_sym_train`: 训练用接收信号 (列向量)
- `symb_train`: 训练用真实符号 (列向量, PAM4: -3,-1,1,3)

**输出参数:**
- `modelFile`: 保存的模型文件路径

**模型文件包含:**
- 网络权重: W1, b1, W2, b2
- 网络参数: n0, n1, k_delay
- PAM4电平: pam4_levels
- WD参数: alpha_wd, beta_wd
- 正则化参数: dropout_rate

### test_wdrnn() - 测试函数
```matlab
[SER, BER, eqOut, predLevels] = test_wdrnn(modelFile, rx_sym_test, symb_test)
```
**输入参数:**
- `modelFile`: 训练好的模型文件路径
- `rx_sym_test`: 测试用接收信号 (列向量)
- `symb_test`: 测试用真实符号 (列向量)

**输出参数:**
- `SER`: 符号误差率
- `BER`: 比特误差率
- `eqOut`: 均衡器输出信号
- `predLevels`: 预测的符号电平

## 网络架构

WD-RNN采用以下架构:
- **输入层**: 67维 (61个接收窗口样点 + 6个延迟反馈)
- **隐藏层**: 20个神经元，tanh激活函数
- **输出层**: 1个神经元，线性输出
- **训练**: Teacher forcing + Adam优化器
- **测试**: 递归反馈 + 加权判决 (WD)

## 关键特性

1. **加权判决 (Weighted Decision)**
   - 根据输出可靠性动态调整反馈权重
   - 提高非线性失真环境下的鲁棒性

2. **Teacher Forcing训练**
   - 训练时使用真实符号作为延迟反馈
   - 加速训练收敛

3. **递归反馈测试**
   - 测试时使用均衡器自身输出作为反馈
   - 实现真实的在线均衡

4. **GPU加速支持**
   - 自动检测GPU可用性
   - 训练和测试均支持GPU加速

## 参数说明

### 网络参数
- `n0 = 61`: 输入窗口大小 (论文最优值)
- `n1 = 20`: 隐藏层神经元数 (论文最优值)
- `k_delay = 6`: 延迟反馈长度 (论文推荐值)

### WD参数
- `alpha_wd = 5`: WD sigmoid函数参数
- `beta_wd = 0.14`: WD阈值参数

### 训练参数
- `maxEpochs = 30`: 训练轮数
- `miniBatch = 1024`: 批量大小
- `learnRate = 1e-3`: Adam学习率
- `dropout_rate = 0.01`: Dropout比率

## 注意事项

1. **数据格式**: 输入数据必须为列向量
2. **PAM4电平**: 使用标准电平 [-3, -1, 1, 3]
3. **Gray映射**: 00→-3, 01→-1, 11→1, 10→3
4. **内存需求**: 大数据集可能需要较大内存，建议使用GPU
5. **随机种子**: 训练函数使用固定种子保证可重现性

## 性能参考

在标准测试条件下 (SNR=20dB, 轻微非线性):
- 符号误差率: ~1e-3 级别
- 比特误差率: ~5e-4 级别
- 训练时间: ~几秒钟 (5K符号, GPU)
- 测试时间: ~1秒钟 (2K符号, GPU)

实际性能取决于信道条件、噪声水平和数据量。
