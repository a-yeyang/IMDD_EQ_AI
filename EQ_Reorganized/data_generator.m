% pam4_wdrnn_cls.m
% 只更换均衡器模型为“WD-RNN-CLS（回归+分类双头 & 置信驱动的加权判决）”，其余流程保持与你原脚本一致的简化链路
% （RRC成形、简化信道、匹配滤波/抽样、teacher forcing 训练、WD 测试）。
clear; close all; clc;
%% ----------------- 仿真/网络 超参数（保持原设定，除模型外不改） -----------------
useGPU = true;            % 是否使用 GPU（自动检测）；若你的 MATLAB/平台 不支持 GPU，可设 false
rngSeed = 12345;          % 随机数种子（使用 Mersenne Twister）
rng(rngSeed, 'twister');

config;

% PAM4 符号列表（{-3,-1,1,3}）
pam4_levels = [-3, -1, 1, 3];

%% ----------------- 产生数据（发射端） -----------------
% 生成训练 & 测试符号（二进制 → Gray 映射 → PAM4）
numTrain = nSymbols_train;
numTest  = nSymbols_test;
totalSymbols = numTrain + numTest;

% 2 bits per symbol
bits = randi([0 1], 2*totalSymbols, 1, 'uint8');  
% Gray mapping: 00->-3, 01->-1, 11->1, 10->3
mapGray = containers.Map({'00','01','11','10'}, {-3,-1,1,3});

symb = zeros(totalSymbols,1);
for i=1:totalSymbols
    b1 = num2str(bits(2*i-1));
    b2 = num2str(bits(2*i));
    key = [b1 b2];
    symb(i) = mapGray(key);
end

% 划分训练/测试
symb_train = symb(1:numTrain);
symb_test  = symb(numTrain+1:end);

% 上采样 & RRC 成形（发射）
tx_up_train = upsample(symb_train, sps);
tx_up_test  = upsample(symb_test, sps);
tx_train = conv(tx_up_train, rrc, 'same');
tx_test  = conv(tx_up_test,  rrc, 'same');

tx_all = [tx_train; tx_test]';
symb_all = [symb_train; symb_test]';
save('tx_all_to_vpi.txt', 'tx_all','-ascii');
save('symb_all.txt', 'symb_all','-ascii');

save('symb_train.txt','symb_train','-ascii');
save('symb_test.txt','symb_test','-ascii');


