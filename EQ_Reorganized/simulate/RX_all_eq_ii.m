% pam4_wdrnn_1.m
% 改写：训练后保存模型，再加载模型用于测试
clear; close all; clc;
config;
%% ----------------- 仿真/网络 超参数 -----------------
useGPU = false;

% 自检当前操作系统
if ispc
    disp('当前系统为Windows。useGPU 保持为 false。');
elseif ismac
    useGPU = false;
    disp('当前系统为macOS。useGPU 设置为 false。');
else
    % 处理其他操作系统的情况，例如Linux
    disp('当前系统不是Windows或macOS。useGPU 保持为 false。');
end
rngSeed = 12345;          
rng(rngSeed, 'twister');                       


%% 测试SNR值
SNR_dB_list = [28,24,20,16,12,8,4];
numSNR = length(SNR_dB_list);
%% ===================================================
rx=-load('vpi_data.txt');
rx=2*(rx-mean(rx))/mean(abs(rx));
%% ===================================================
rx_train=rx(1:nSymbols_train*sps);
rx_test=rx(nSymbols_test*sps+1:end);

%% ----------------- 匹配滤波+下采样 ----------------- 
rx_matched_train = conv(rx_train, rrc,'same');
rx_matched_test  = conv(rx_test,  rrc,'same');
rx_sym_train = resample(rx_matched_train,Rs,Fs)';
rx_sym_test  = resample(rx_matched_test,Rs,Fs)';
symb_train=load('symb_train.txt');
symb_test=load('symb_test.txt');
modelFile1=wd_rnn(rx_sym_train,symb_train);
modelFile2=train_cwd_rnn(rx_sym_train,symb_train);

%% ----------------- 测试阶段 -----------------
% 测试无均衡器和CMA均衡器
SER_no = zeros(numel(SNR_dB_list),1);
SER_cma = zeros(numel(SNR_dB_list),1);
SER_wdrnn = zeros(numel(SNR_dB_list),1);
SER_wdrnn1 = zeros(numel(SNR_dB_list),1);
for i = 1:numel(SNR_dB_list)
    rx_sym_test_snr = awgn(rx_sym_test, SNR_dB_list(i));
    
    % 无均衡器测试
    symb_no = sign(rx_sym_test_snr) + (rx_sym_test_snr==0) + 2*(rx_sym_test_snr>1) - 2*(rx_sym_test_snr<-1);
    SER_no(i) = sum(symb_no ~= symb_test)/length(symb_test);
    
    % CMA均衡器测试
    Pol_X = 3*CMA(rx_sym_test_snr, 25)';
    t = 2;
    symb_cma = sign(Pol_X) + (Pol_X==0) + 2*(Pol_X>t) - 2*(Pol_X<-t);
    SER_cma(i) = sum(symb_cma ~= symb_test)/length(symb_test);
end

% 测试WD-RNN神经网络模型
test_options = struct();
test_options.useGPU = useGPU;

parfor ii=1:numel(SNR_dB_list)
[SER_wdrnn(ii), ~, ~, ~]= test_wdrnn     (modelFile1, awgn(rx_sym_test,SNR_dB_list(ii)), symb_test);
[SER_wdrnn1(ii), ~, ~, ~] = test_cwd_rnn    (modelFile2, awgn(rx_sym_test,SNR_dB_list(ii)), symb_test);
end
%% ----------------- 绘制比较图 -----------------
fprintf('\n绘制SNR vs SER比较图...\n');

figure('Position', [100, 100, 800, 600]);
semilogy(SNR_dB_list, SER_no, 'ro-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '无均衡器');
hold on;
semilogy(SNR_dB_list, SER_cma, 'bs-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'CMA均衡器');
semilogy(SNR_dB_list, SER_wdrnn, 'm*-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'WDRNN均衡器');
semilogy(SNR_dB_list, SER_wdrnn1, 'g^-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'WDRNN-1均衡器-惩罚力度调整');

xlabel('信噪比 (dB)', 'FontSize', 14);
ylabel('误符号率 (SER)', 'FontSize', 14);
title('PAM4系统均衡性能比较 IM/DD 20km', 'FontSize', 16);
legend('Location', 'best', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 12);

% 设置y轴范围
ylim([1e-6, 1]);

% 保存结果
save('equalizer_comparison_results.mat', 'SNR_dB_list', 'SER_no', 'SER_cma', 'SER_wdrnn1');

% 输出结果表格
fprintf('\n=== 测试结果汇总 ===\n');
fprintf('SNR(dB)\t无均衡\t\tCMA\t\tWD-RNN-1\tWD-RNN\n');
fprintf('----------------------------------------\n');
for i = 1:length(SNR_dB_list)
    fprintf('%d\t%.2e\t%.2e\t%.2e\t%.2e\n', SNR_dB_list(i), SER_no(i), SER_cma(i), SER_wdrnn1(i),SER_wdrnn(i));
end

fprintf('\n测试完成！结果已保存到 equalizer_comparison_results.mat\n');


fprintf('所有任务完成。\n');