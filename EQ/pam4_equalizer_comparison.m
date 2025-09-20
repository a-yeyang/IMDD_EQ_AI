% pam4_equalizer_comparison.m
% PAM4系统三种均衡方法比较：无均衡、CMA均衡、WD-RNN均衡
% 并行测试多个SNR值：4, 8, 12, 16, 20 dB
% 输出SNR vs BER比较图

clear; close all; clc;

%% ----------------- 仿真参数设置 -----------------
% 基本参数
useGPU = true;            % 是否使用GPU
rngSeed = 12345;          % 随机数种子
rng(rngSeed, 'twister');

% 信号参数
trainSymbols = 30000;     % 训练符号数（减少以加快测试）
testSymbols  = 30000;     % 测试符号数
sps = 4;                  % samples per symbol
rrc_rolloff = 0.1;       % RRC roll-off
rrc_span = 10;            % RRC span

% WD-RNN 架构参数
n0 = 61;                  % 输入节点数
n1 = 20;                  % 隐藏层神经元数
k_delay = 6;              % 延迟单元
alpha_wd = 5; beta_wd = 0.14;  % WD 参数
dropout_rate = 0.01;           % dropout率

% 训练参数
maxEpochs = 20;           % 减少epoch数以加快测试
miniBatch = 1024;         % 小批量大小
learnRate = 1e-3;         % 学习率

% 训练SNR（固定）
train_SNR_dB = 10;

% 测试SNR值
SNR_dB_list = [20,16,12,8,4];
numSNR = length(SNR_dB_list);

% PAM4 符号列表
pam4_levels = [-3, -1, 1, 3];

% 结果存储
BER_no_eq = zeros(numSNR, 1);      % 无均衡
BER_cma = zeros(numSNR, 1);        % CMA均衡
BER_rnn = zeros(numSNR, 1);        % RNN均衡

% WD-RNN网络参数（全局变量，训练一次后复用）
global W1_global b1_global W2_global b2_global;
W1_global = []; b1_global = []; W2_global = []; b2_global = [];

%% GPU 可用性检测
if useGPU
    try
        gcount = gpuDeviceCount;
        if gcount >= 1
            gpuInfo = gpuDevice;
            fprintf('检测到GPU: %s，将使用GPU加速\n', gpuInfo.Name);
            useGPU = true;
        else
            fprintf('未检测到GPU，使用CPU运行\n');
            useGPU = false;
        end
    catch
        fprintf('GPU检测失败，使用CPU运行\n');
        useGPU = false;
    end
else
    fprintf('用户禁用GPU，使用CPU运行\n');
end

%% ----------------- 在固定SNR下训练WD-RNN -----------------
fprintf('在SNR=%d dB条件下训练WD-RNN均衡器...\n', train_SNR_dB);

% 生成训练数据
numTrain = trainSymbols;
totalSymbols = numTrain;

% 2 bits per symbol
bits_train = randi([0 1], 2*totalSymbols, 1, 'uint8');
% Gray mapping: 00->-3, 01->-1, 11->1, 10->3
mapGray = containers.Map({'00','01','11','10'}, {-3,-1,1,3});

symb_train = zeros(totalSymbols,1);
for i=1:totalSymbols
    b1 = num2str(bits_train(2*i-1));
    b2 = num2str(bits_train(2*i));
    key = [b1 b2];
    symb_train(i) = mapGray(key);
end

% 上采样和脉冲整形
tx_up_train = upsample(symb_train, sps);
rrc = rcosdesign(rrc_rolloff, rrc_span, sps, 'sqrt');
tx_train = conv(tx_up_train, rrc, 'same');

% PAPR clipping
clipThr = 3.5;
tx_train(tx_train>clipThr)=clipThr;
tx_train(tx_train<-clipThr)=-clipThr;

% 信道模型
lpOrder = 80;
lpCut = 0.2;
h_lp = fir1(lpOrder, lpCut);
nl_a1 = 1.0;
nl_a3 = 0.01;

chan_train = filter(h_lp,1, tx_train);
chan_train = nl_a1*chan_train + nl_a3*chan_train.^3;

% 加AWGN噪声（训练SNR）
Esym = mean(symb_train.^2);
signalPower = mean(chan_train.^2);
SNR_lin = 10^(train_SNR_dB/10);
noiseVar = signalPower / SNR_lin;
noiseStd = sqrt(noiseVar);
chan_train = chan_train + noiseStd * randn(size(chan_train));

% 接收端DSP
rx_matched_train = conv(chan_train, rrc, 'same');
delay = (length(rrc)-1)/2 + (length(h_lp)-1)/2;
delay = round(delay);
startIdx = delay + 1;
padlen = 100;
rx_matched_train = [zeros(padlen,1); rx_matched_train; zeros(padlen,1)];
rx_sym_train = rx_matched_train(startIdx : sps : startIdx + sps*(numTrain-1))';
rx_sym_train = rx_sym_train(:);

% 构建训练输入
padL = floor(n0/2);
padR = n0 - padL - 1;
rx_train_pad = [zeros(padL,1); rx_sym_train; zeros(padR+k_delay,1)];

Ntrain = numTrain;
inputDim = n0 + k_delay;

Xtrain = zeros(inputDim, Ntrain);
Ttrain = zeros(1, Ntrain);

for i=1:Ntrain
    idx_center = i + padL;
    window = rx_train_pad(idx_center - floor(n0/2) : idx_center + ceil(n0/2)-1);
    if k_delay>0
        prevLabels = zeros(k_delay,1);
        for kk=1:k_delay
            if i-kk >= 1
                prevLabels(kk) = symb_train(i-kk);
            else
                prevLabels(kk) = 0;
            end
        end
    else
        prevLabels = [];
    end
    Xtrain(:,i) = [window(:); prevLabels(:)];
    Ttrain(1,i) = symb_train(i);
end

% 初始化网络参数
rng(rngSeed);
W1 = 0.1*randn(n1, inputDim);
b1 = zeros(n1,1);
W2 = 0.1*randn(1, n1);
b2 = 0;

% 将数据/参数放到GPU（如可用）
if useGPU
    Xtrain = gpuArray(single(Xtrain));
    Ttrain = gpuArray(single(Ttrain));
    W1 = gpuArray(single(W1));
    b1 = gpuArray(single(b1));
    W2 = gpuArray(single(W2));
    b2 = gpuArray(single(b2));
end

% 训练网络
mW1 = zeros(size(W1),'like',W1); vW1 = mW1;
mb1 = zeros(size(b1),'like',b1); vb1 = mb1;
mW2 = zeros(size(W2),'like',W2); vW2 = mW2;
mb2 = zeros(size(b2),'like',b2); vb2 = mb2;
beta1 = 0.9; beta2 = 0.999; epsAdam = 1e-8;
iter = 0;

numBatches = ceil(Ntrain / miniBatch);

fprintf('开始训练WD-RNN...\n');
for epoch = 1:maxEpochs
    idx = randperm(Ntrain);
    if useGPU, idx = gpuArray(idx); end
    Xsh = Xtrain(:, idx);
    Tsh = Ttrain(:, idx);
    
    for b = 1:numBatches
        iter = iter + 1;
        i1 = (b-1)*miniBatch + 1;
        i2 = min(b*miniBatch, Ntrain);
        Xbatch = Xsh(:, i1:i2);
        Tbatch = Tsh(:, i1:i2);
        
        % 前向传播
        Z1 = W1 * Xbatch + b1;
        H1 = tanh(Z1);
        if dropout_rate > 0
            mask = (rand(size(H1)) > dropout_rate);
            if useGPU, mask = gpuArray(mask); end
            H1 = H1 .* mask / (1-dropout_rate);
        end
        Ypred = W2 * H1 + b2;
        
        % 损失和梯度
        err = Ypred - Tbatch;
        loss = mean(err.^2, 'all');
        
        batchSizeCurr = size(Xbatch,2);
        dY = (2/batchSizeCurr) * err;
        
        dW2 = dY * H1';
        db2 = sum(dY,2);
        
        dH1 = (W2') * dY;
        dZ1 = dH1 .* (1 - H1.^2);
        
        dW1 = dZ1 * Xbatch';
        db1 = sum(dZ1,2);
        
        % Adam更新
        mW1 = beta1*mW1 + (1-beta1)*dW1;
        vW1 = beta2*vW1 + (1-beta2)*(dW1.^2);
        mhatW1 = mW1 / (1 - beta1^iter);
        vhatW1 = vW1 / (1 - beta2^iter);
        W1 = W1 - learnRate * mhatW1 ./ (sqrt(vhatW1) + epsAdam);
        
        mb1 = beta1*mb1 + (1-beta1)*db1;
        vb1 = beta2*vb1 + (1-beta2)*(db1.^2);
        mbhat = mb1 / (1 - beta1^iter);
        vbhat = vb1 / (1 - beta2^iter);
        b1 = b1 - learnRate * mbhat ./ (sqrt(vbhat) + epsAdam);
        
        mW2 = beta1*mW2 + (1-beta1)*dW2;
        vW2 = beta2*vW2 + (1-beta2)*(dW2.^2);
        mhatW2 = mW2 / (1 - beta1^iter);
        vhatW2 = vW2 / (1 - beta2^iter);
        W2 = W2 - learnRate * mhatW2 ./ (sqrt(vhatW2) + epsAdam);
        
        mb2 = beta1*mb2 + (1-beta1)*db2;
        vb2 = beta2*vb2 + (1-beta2)*(db2.^2);
        mb2hat = mb2 / (1 - beta1^iter);
        vb2hat = vb2 / (1 - beta2^iter);
        b2 = b2 - learnRate * mb2hat ./ (sqrt(vb2hat) + epsAdam);
    end
    
    if mod(epoch, 5) == 0
        if useGPU
            lossVal = gather(double(loss));
        else
            lossVal = double(loss);
        end
        fprintf('Epoch %d/%d - Loss: %.5e\n', epoch, maxEpochs, lossVal);
    end
end

% 保存训练好的网络参数
W1_global = W1;
b1_global = b1;
W2_global = W2;
b2_global = b2;

fprintf('WD-RNN训练完成！\n\n');

%% ----------------- 在不同SNR下测试所有均衡方法 -----------------
fprintf('开始测试 %d 个SNR值...\n', numSNR);

for snr_idx = 1:numSNR
    SNR_dB = SNR_dB_list(snr_idx);
    fprintf('\n=== 测试 SNR = %d dB ===\n', SNR_dB);
    
    %% ----------------- 生成测试数据 -----------------
    % 生成测试符号
    numTest = testSymbols;
    
    % 2 bits per symbol
    bits_test = randi([0 1], 2*numTest, 1, 'uint8');
    % Gray mapping: 00->-3, 01->-1, 11->1, 10->3
    mapGray = containers.Map({'00','01','11','10'}, {-3,-1,1,3});
    
    symb_test = zeros(numTest,1);
    for i=1:numTest
        b1 = num2str(bits_test(2*i-1));
        b2 = num2str(bits_test(2*i));
        key = [b1 b2];
        symb_test(i) = mapGray(key);
    end
    
    % 上采样和脉冲整形
    tx_up_test = upsample(symb_test, sps);
    tx_test = conv(tx_up_test, rrc, 'same');
    
    % PAPR clipping
    clipThr = 3.5;
    tx_test(tx_test>clipThr)=clipThr;
    tx_test(tx_test<-clipThr)=-clipThr;
    
    %% ----------------- 信道模型 -----------------
    % 处理测试序列
    chan_test = filter(h_lp,1, tx_test);
    chan_test = nl_a1*chan_test + nl_a3*chan_test.^3;
    
    % 加AWGN噪声（当前测试SNR）
    Esym = mean(symb_test.^2);
    signalPower = mean(chan_test.^2);
    SNR_lin = 10^(SNR_dB/10);
    noiseVar = signalPower / SNR_lin;
    noiseStd = sqrt(noiseVar);
    
    chan_test = chan_test + noiseStd * randn(size(chan_test));
    
    %% ----------------- 接收端DSP -----------------
    % 匹配滤波
    rx_matched_test = conv(chan_test, rrc, 'same');
    
    % 下采样
    rx_matched_test = [zeros(padlen,1); rx_matched_test; zeros(padlen,1)];
    rx_sym_test = rx_matched_test(startIdx : sps : startIdx + sps*(numTest-1))';
    rx_sym_test = rx_sym_test(:);
    
    %% ----------------- 1. CMA均衡器测试（优先测试） -----------------
    fprintf('  测试CMA均衡器...\n');
    % 使用CMA均衡器
    addpath('utils');  % 添加utils路径
    [Pol_X_cma, ~] = CMA(rx_sym_test, 25);
    
    % CMA均衡器判决门限设置：0, ±t, t=0.75
    t = 0.75;  % 判决门限
    predLevels_cma = zeros(numTest, 1);
    for i = 1:numTest
        if Pol_X_cma(i) > t
            predLevels_cma(i) = 3;      % 判决为3
        elseif Pol_X_cma(i) > 0
            predLevels_cma(i) = 1;      % 判决为1
        elseif Pol_X_cma(i) > -t
            predLevels_cma(i) = -1;     % 判决为-1
        else
            predLevels_cma(i) = -3;     % 判决为-3
        end
    end
    
    % 计算BER
    symErr_cma = sum(predLevels_cma ~= symb_test) / numTest;
    BER_cma(snr_idx) = symErr_cma;
    
    %% ----------------- 2. 无均衡器测试 -----------------
    fprintf('  测试无均衡器...\n');
    % 直接硬判决
    predLevels_no_eq = zeros(numTest, 1);
    for i = 1:numTest
        [~, idxMin] = min(abs(rx_sym_test(i) - pam4_levels));
        predLevels_no_eq(i) = pam4_levels(idxMin);
    end
    
    % 计算BER
    symErr_no_eq = sum(predLevels_no_eq ~= symb_test) / numTest;
    BER_no_eq(snr_idx) = symErr_no_eq;
    
    %% ----------------- 3. WD-RNN均衡器测试 -----------------
    fprintf('  测试WD-RNN均衡器...\n');
    
    % 使用训练好的网络参数
    W1 = W1_global;
    b1 = b1_global;
    W2 = W2_global;
    b2 = b2_global;
    
    % 测试阶段：递归推理
    padL = floor(n0/2);
    padR = n0 - padL - 1;
    rx_test_pad = [zeros(padL,1); rx_sym_test; zeros(padR,1)];
    Ntest = numTest;
    
    feedbackBuf = zeros(k_delay,1,'like',W1);
    predLevels_rnn = zeros(Ntest,1);
    
    for i = 1:Ntest
        idx_center = i + padL;
        window = rx_test_pad(idx_center - floor(n0/2) : idx_center + ceil(n0/2)-1);
        prevLabels = zeros(k_delay,1);
        for kk=1:k_delay
            if (i-kk) >= 1
                prevLabels(kk) = feedbackBuf(kk);
            else
                prevLabels(kk) = 0;
            end
        end
        xin = [window(:); prevLabels(:)];
        if useGPU, xin = gpuArray(single(xin)); else xin = single(xin); end
        
        % 前向传播
        z1 = W1 * xin + b1;
        h1 = tanh(z1);
        y = W2 * h1 + b2;
        if useGPU, y = gather(y); end
        y = double(y);
        
        % 硬判决
        [~, idxMin] = min(abs(y - pam4_levels));
        yhat = pam4_levels(idxMin);
        
        % 计算gamma和S(x)
        if y < -3 || y > 3
            gamma = 1;
        else
            gamma = 1 - abs(y - yhat);
            gamma = max(0, min(1, gamma));
        end
        
        Sx = 0.5 * ( 1 - exp(-alpha_wd*(gamma/beta_wd - 1)) ./ (1 + exp(-alpha_wd*(gamma/beta_wd - 1))) + 1 );
        Sx = double(Sx);
        
        ytilde = Sx * yhat + (1 - Sx) * y;
        
        % 更新反馈缓冲区
        if k_delay > 0
            feedbackBuf = [ytilde; feedbackBuf(1:end-1)];
        end
        
        predLevels_rnn(i) = yhat;
    end
    
    % 计算BER
    symErr_rnn = sum(predLevels_rnn ~= symb_test) / numTest;
    BER_rnn(snr_idx) = symErr_rnn;
    
    fprintf('  SNR=%d dB: CMA=%.2e, 无均衡=%.2e, RNN=%.2e\n', ...
            SNR_dB, BER_cma(snr_idx), BER_no_eq(snr_idx), BER_rnn(snr_idx));
end

%% ----------------- 绘制比较图 -----------------
fprintf('\n绘制SNR vs BER比较图...\n');

figure('Position', [100, 100, 800, 600]);
semilogy(SNR_dB_list, BER_no_eq, 'ro-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '无均衡器');
hold on;
semilogy(SNR_dB_list, BER_cma, 'bs-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'CMA均衡器');
semilogy(SNR_dB_list, BER_rnn, 'g^-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'WD-RNN均衡器');

xlabel('信噪比 (dB)', 'FontSize', 14);
ylabel('误码率 (BER)', 'FontSize', 14);
title('PAM4系统三种均衡方法性能比较', 'FontSize', 16);
legend('Location', 'best', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 12);

% 设置y轴范围
ylim([1e-6, 1]);

% 保存结果
save('equalizer_comparison_results.mat', 'SNR_dB_list', 'BER_no_eq', 'BER_cma', 'BER_rnn');

% 输出结果表格
fprintf('\n=== 测试结果汇总 ===\n');
fprintf('SNR(dB)\tCMA\t\t无均衡\t\tRNN\n');
fprintf('----------------------------------------\n');
for i = 1:numSNR
    fprintf('%d\t%.2e\t%.2e\t%.2e\n', SNR_dB_list(i), BER_cma(i), BER_no_eq(i), BER_rnn(i));
end

fprintf('\n测试完成！结果已保存到 equalizer_comparison_results.mat\n');
