% demo_wd_rnn.m
% WD-RNN 训练和测试演示脚本
% 展示如何使用 wd_rnn() 训练模型，然后用 test_wdrnn() 测试

clear; close all; clc;

fprintf('=== WD-RNN 训练和测试演示 ===\n\n');

%% 参数设置
numTrainSymbols = 5000;  % 训练符号数（为了快速演示，使用较小数量）
numTestSymbols = 2000;   % 测试符号数
SNR_dB = 20;            % 信噪比
sps = 4;                % 每符号采样数

% PAM4 电平
pam4_levels = [-3, -1, 1, 3];

fprintf('仿真参数:\n');
fprintf('  训练符号数: %d\n', numTrainSymbols);
fprintf('  测试符号数: %d\n', numTestSymbols);
fprintf('  信噪比: %d dB\n', SNR_dB);
fprintf('  每符号采样数: %d\n', sps);

%% 生成训练数据
fprintf('\n=== 步骤1: 生成训练数据 ===\n');

% 生成随机PAM4符号
rng(42); % 固定随机种子保证可重现
symb_train = pam4_levels(randi(4, numTrainSymbols, 1));

% 简单信道模型：加噪声 + 轻微非线性
rx_sym_train = symb_train + 0.3*randn(size(symb_train));  % AWGN
rx_sym_train = rx_sym_train + 0.05*rx_sym_train.^3;       % 轻微三阶非线性

fprintf('训练数据生成完成\n');
fprintf('  符号范围: [%.1f, %.1f]\n', min(symb_train), max(symb_train));
fprintf('  接收信号SNR约: %.1f dB\n', 20*log10(std(symb_train)/std(rx_sym_train-symb_train)));

%% 训练WD-RNN模型
fprintf('\n=== 步骤2: 训练WD-RNN模型 ===\n');

tic;
modelFile = wd_rnn(rx_sym_train, symb_train);
trainingTime = toc;

fprintf('训练完成！耗时: %.2f 秒\n', trainingTime);
fprintf('模型保存至: %s\n', modelFile);

%% 生成测试数据
fprintf('\n=== 步骤3: 生成测试数据 ===\n');

% 生成测试符号（不同的随机种子）
rng(123);
symb_test = pam4_levels(randi(4, numTestSymbols, 1));

% 相同的信道模型
rx_sym_test = symb_test + 0.3*randn(size(symb_test));   % AWGN  
rx_sym_test = rx_sym_test + 0.05*rx_sym_test.^3;        % 轻微三阶非线性

fprintf('测试数据生成完成\n');
fprintf('  测试符号数: %d\n', numTestSymbols);

%% 测试WD-RNN模型
fprintf('\n=== 步骤4: 测试WD-RNN模型 ===\n');

tic;
[SER, BER, eqOut, predLevels] = test_wdrnn0(modelFile, rx_sym_test, symb_test);
testingTime = toc;

fprintf('测试完成！耗时: %.2f 秒\n', testingTime);

%% 结果可视化
fprintf('\n=== 步骤5: 结果可视化 ===\n');

figure('Position', [100, 100, 1200, 800]);

% 子图1: 原始接收信号星座图
subplot(2,3,1);
plot(rx_sym_test, zeros(size(rx_sym_test)), 'b.', 'MarkerSize', 4);
hold on;
plot(pam4_levels, zeros(size(pam4_levels)), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
grid on;
title('原始接收信号');
xlabel('幅度');
ylabel('');
legend('接收信号', 'PAM4电平', 'Location', 'best');
ylim([-0.5, 0.5]);

% 子图2: 均衡后信号星座图
subplot(2,3,2);
plot(eqOut, zeros(size(eqOut)), 'r.', 'MarkerSize', 4);
hold on;
plot(pam4_levels, zeros(size(pam4_levels)), 'go', 'MarkerSize', 8, 'LineWidth', 2);
grid on;
title('WD-RNN均衡后信号');
xlabel('幅度');
ylabel('');
legend('均衡信号', 'PAM4电平', 'Location', 'best');
ylim([-0.5, 0.5]);

% 子图3: 信号时域波形比较
subplot(2,3,3);
plotRange = 1:min(200, numTestSymbols);
plot(plotRange, rx_sym_test(plotRange), 'b-', 'LineWidth', 1);
hold on;
plot(plotRange, eqOut(plotRange), 'r-', 'LineWidth', 1);
plot(plotRange, symb_test(plotRange), 'ko', 'MarkerSize', 4);
grid on;
title('信号时域波形对比');
xlabel('符号索引');
ylabel('幅度');
legend('接收信号', '均衡输出', '真实符号', 'Location', 'best');

% 子图4: 均衡器输出直方图
subplot(2,3,4);
histogram(eqOut, 50, 'Normalization', 'probability', 'FaceAlpha', 0.7);
hold on;
for i = 1:length(pam4_levels)
    xline(pam4_levels(i), 'r--', 'LineWidth', 2);
end
grid on;
title('均衡输出直方图');
xlabel('幅度');
ylabel('概率');

% 子图5: 误差统计
subplot(2,3,5);
errorVec = abs(eqOut - symb_test);
histogram(errorVec, 30, 'Normalization', 'probability', 'FaceAlpha', 0.7);
grid on;
title('均衡误差分布');
xlabel('|均衡输出 - 真实符号|');
ylabel('概率');

% 子图6: 性能总结
subplot(2,3,6);
axis off;
textStr = {
    ['符号误差率 (SER): ' sprintf('%.2e', SER)];
    ['比特误差率 (BER): ' sprintf('%.2e', BER)];
    [''];
    ['训练时间: ' sprintf('%.2f 秒', trainingTime)];
    ['测试时间: ' sprintf('%.2f 秒', testingTime)];
    [''];
    ['训练符号数: ' sprintf('%d', numTrainSymbols)];
    ['测试符号数: ' sprintf('%d', numTestSymbols)];
    ['信噪比: ' sprintf('%d dB', SNR_dB)];
};
text(0.1, 0.9, textStr, 'FontSize', 12, 'VerticalAlignment', 'top', ...
     'FontName', 'FixedWidth');
title('性能总结');

sgtitle('WD-RNN 均衡器性能评估', 'FontSize', 16, 'FontWeight', 'bold');

fprintf('\n=== 演示完成 ===\n');
fprintf('图形窗口显示了详细的性能分析结果\n');
fprintf('模型文件: %s (可用于后续测试)\n', modelFile);

%% 清理演示用的临时文件（可选）
% 如果不需要保留模型文件，可以取消注释下面的行
% delete(modelFile);
% fprintf('临时模型文件已删除\n');
