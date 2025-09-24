% test_wd_rnn_function.m
% 测试 wd_rnn 函数的脚本

clear; close all; clc;

fprintf('Testing wd_rnn function...\n');

%% 生成测试数据
numSymbols = 1000;  % 较小的数据集用于快速测试
pam4_levels = [-3, -1, 1, 3];

% 生成随机PAM4符号
symb_train = pam4_levels(randi(4, numSymbols, 1));

% 生成带噪声的接收信号（模拟信道失真）
rx_sym_train = symb_train + 0.2*randn(size(symb_train));

fprintf('Generated %d training symbols\n', numSymbols);
fprintf('Symbol range: [%.1f, %.1f]\n', min(symb_train), max(symb_train));
fprintf('Received signal SNR approximately: %.1f dB\n', 20*log10(std(symb_train)/0.2));

%% 调用训练函数
fprintf('\nCalling wd_rnn training function...\n');
tic;
modelFile = wd_rnn(rx_sym_train, symb_train);
trainingTime = toc;

fprintf('Training completed in %.2f seconds\n', trainingTime);
fprintf('Model saved to: %s\n', modelFile);

%% 验证保存的模型
if exist(modelFile, 'file')
    fprintf('\nLoading and verifying saved model...\n');
    savedModel = load(modelFile);
    
    fprintf('Model contains the following variables:\n');
    disp(fieldnames(savedModel));
    
    fprintf('Network architecture:\n');
    fprintf('  Input dimension: %d (n0=%d + k_delay=%d)\n', ...
        savedModel.n0 + savedModel.k_delay, savedModel.n0, savedModel.k_delay);
    fprintf('  Hidden layer size: %d\n', savedModel.n1);
    fprintf('  Output dimension: 1\n');
    fprintf('  PAM4 levels: [%s]\n', num2str(savedModel.pam4_levels));
    
    fprintf('\nWeight matrix dimensions:\n');
    fprintf('  W1: %dx%d\n', size(savedModel.W1,1), size(savedModel.W1,2));
    fprintf('  b1: %dx%d\n', size(savedModel.b1,1), size(savedModel.b1,2));
    fprintf('  W2: %dx%d\n', size(savedModel.W2,1), size(savedModel.W2,2));
    fprintf('  b2: %dx%d\n', size(savedModel.b2,1), size(savedModel.b2,2));
    
    fprintf('\nWD parameters:\n');
    fprintf('  alpha_wd: %.2f\n', savedModel.alpha_wd);
    fprintf('  beta_wd: %.2f\n', savedModel.beta_wd);
    fprintf('  dropout_rate: %.3f\n', savedModel.dropout_rate);
    
    fprintf('\nTest completed successfully!\n');
else
    fprintf('ERROR: Model file not found!\n');
end
