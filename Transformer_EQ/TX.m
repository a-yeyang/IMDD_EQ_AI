%% IM/DD Transformer仿真的发射机程序
clear; close all; clc;
%% ----------------- 仿真/网络 超参数（保持原设定，除模型外不改） -----------------
useGPU = true;            % 是否使用 GPU（自动检测）；若你的 MATLAB/平台 不支持 GPU，可设 false
rngSeed = 12345;          % 随机数种子（使用 Mersenne Twister）
rng(rngSeed, 'twister');
config;
% PAM4 符号列表（{-3,-1,1,3}）
pam4_levels = [-3, -1, 1, 3];

%% ----------------- 产生数据（发射端） -----------------
% 生成训练、验证 & 测试符号（二进制 → Gray 映射 → PAM4）
numTrain = nSymbols_train;
numVal   = nSymbols_train;  % 验证集大小
numTest  = nSymbols_train*2;
totalSymbols = numTrain + numVal + numTest;

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

% 划分训练/验证/测试
symb_train = symb(1:numTrain);
symb_val   = symb(numTrain+1:numTrain+numVal);
symb_test  = symb(numTrain+numVal+1:end);

% 上采样 & RRC 成形（发射）
tx_up_train = upsample(symb_train, sps);
tx_up_val   = upsample(symb_val, sps);
tx_up_test  = upsample(symb_test, sps);
tx_train = conv(tx_up_train, rrc, 'same');
tx_val   = conv(tx_up_val,   rrc, 'same');
tx_test  = conv(tx_up_test,  rrc, 'same');

tx_all = [tx_train; tx_val; tx_test]';
symb_all = [symb_train; symb_val; symb_test]';
save('tx_all_to_vpi.txt', 'tx_all','-ascii'); % 根升余弦处理后的符号，用于发送给VPI Photonics软件进行仿真
save('symb_all.txt', 'symb_all','-ascii');  % {-3,-1,1,3}符号，全部
save('symb_train.txt','symb_train','-ascii'); % {-3,-1,1,3}符号，用于训练
save('symb_val.txt','symb_val','-ascii');     % {-3,-1,1,3}符号，用于验证
save('symb_test.txt','symb_test','-ascii');   % {-3,-1,1,3}符号，用于测试


