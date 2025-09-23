% run_example.m - 使用新测试函数的示例
% 展示如何使用封装后的测试函数来测试不同模型

clear; close all; clc;
config;

%% ----------------- 系统参数设置 -----------------
useGPU = false;

% 自检当前操作系统
if ispc
    disp('当前系统为Windows。useGPU 保持为 false。');
elseif ismac
    useGPU = false;
    disp('当前系统为macOS。useGPU 设置为 false。');
else
    disp('当前系统不是Windows或macOS。useGPU 保持为 false。');
end

rngSeed = 12345;          
rng(rngSeed, 'twister');

%% 测试SNR值
SNR_dB_list = [28,24,20,16,12,8,4];

%% ----------------- 数据准备 -----------------
rx = -load('vpi_data.txt');
rx = 2*(rx-mean(rx))/mean(abs(rx));
rx = lowpass(rx, 25e9, 120e9);

rx_train = rx(1:nSymbols_train*sps);
rx_test = rx(nSymbols_test*sps+1:end);

% 匹配滤波+下采样
rx_matched_train = conv(rx_train, rrc,'same');
rx_matched_test  = conv(rx_test,  rrc,'same');
rx_sym_train = resample(rx_matched_train,Rs,Fs)';
rx_sym_test  = resample(rx_matched_test,Rs,Fs)';

symb_train = load('symb_train.txt');
symb_test = load('symb_test.txt');

%% ----------------- 训练模型 -----------------
fprintf('开始训练WD-RNN模型...\n');
modelFile = wd_rnn_cls(rx_sym_train, symb_train);

%% ----------------- 测试WD-RNN模型 -----------------
fprintf('\n开始测试WD-RNN模型性能...\n');

% 设置测试选项
test_options = struct();
test_options.useGPU = useGPU;

% 调用WD-RNN测试函数
SER_rnn = test_wdrnn(modelFile, rx_sym_test, symb_test, SNR_dB_list, test_options);

%% ----------------- 测试多个不同的WD-RNN模型示例 -----------------
% 如果需要测试多个不同的WD-RNN模型，可以这样做：
% models = {'model1.mat', 'model2.mat', 'model3.mat'};
% results = cell(length(models), 1);
% 
% for i = 1:length(models)
%     fprintf('测试模型 %s...\n', models{i});
%     results{i} = test_wdrnn(models{i}, rx_sym_test, symb_test, SNR_dB_list, test_options);
% end

fprintf('\n所有测试完成！\n');
