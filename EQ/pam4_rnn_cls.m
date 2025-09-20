% pam4_wdrnn_cls.m
% 只更换均衡器模型为“WD-RNN-CLS（回归+分类双头 & 置信驱动的加权判决）”，其余流程保持与你原脚本一致的简化链路
% （RRC成形、简化信道、匹配滤波/抽样、teacher forcing 训练、WD 测试）。

clear; close all; clc;

%% ----------------- 仿真/网络 超参数（保持原设定，除模型外不改） -----------------
useGPU = true;            % 是否使用 GPU（自动检测）；若你的 MATLAB/平台 不支持 GPU，可设 false
rngSeed = 12345;          % 随机数种子（使用 Mersenne Twister）
rng(rngSeed, 'twister');

% 信号参数（保持原文件的设定）
trainSymbols = 60000;    
testSymbols  = 60000;    
sps = 4;                 
rrc_rolloff = 0.1;       
rrc_span = 10;           

% WD-RNN 架构超参（采用论文优选）
n0 = 61;                 
n1 = 20;                 
k_delay = 6;             
alpha_wd = 5; beta_wd = 0.14;    % WD 压缩函数的形状参数
dropout_rate = 0.01;             % 与论文一致的极小 dropout

% 训练超参（保持原设定）
maxEpochs = 30;         
miniBatch = 1024;       
learnRate = 1e-3;       

% 新增（仅模型相关）：分类头与置信融合权重
lambda_ce  = 0.5;       % 分类头交叉熵权重
lambda_mix = 0.5;       % γ 与置信度融合比例

% 信道 / 噪声（仿真版）
SNR_dB = 20;            

% PAM4 符号列表（{-3,-1,1,3}）
pam4_levels = [-3, -1, 1, 3];

%% ----------------- GPU 可用性检测 -----------------
if useGPU
    try
        gcount = gpuDeviceCount;
        if gcount >= 1
            gpuInfo = gpuDevice;
            fprintf('GPU detected: %s. Will use GPU arrays (若支持).\n', gpuInfo.Name);
            useGPU = true;
        else
            fprintf('No GPU detected. Running on CPU.\n');
            useGPU = false;
        end
    catch
        fprintf('GPU detection failed. Running on CPU.\n');
        useGPU = false;
    end
else
    fprintf('GPU explicitly disabled by user. Running on CPU.\n');
end

%% ----------------- 产生数据（发射端） -----------------
% 生成训练 & 测试符号（二进制 → Gray 映射 → PAM4）
numTrain = trainSymbols;
numTest  = testSymbols;
totalSymbols = numTrain + numTest;

% 2 bits per symbol
bits = randi([0 1], 2*totalSymbols, 1, 'uint8');  % Mersenne Twister 已设定
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
rrc_tx = rcosdesign(rrc_rolloff, rrc_span, sps, 'sqrt');
tx_up_train = upsample(symb_train, sps);
tx_up_test  = upsample(symb_test, sps);
tx_train = conv(tx_up_train, rrc_tx, 'same');
tx_test  = conv(tx_up_test,  rrc_tx, 'same');

% % 简单 PAPR clipping（与原脚本一致的轻量处理）
% clipThr = 3.5;
% tx_train(tx_train>clipThr)=clipThr;  tx_train(tx_train<-clipThr)=-clipThr;
% tx_test(tx_test>clipThr)=clipThr;    tx_test(tx_test<-clipThr)=-clipThr;

%% ----------------- 信道（简化模型：低通 + 三阶非线性 + AWGN） -----------------
lpOrder = 80; lpCut = 0.2; h_lp = fir1(lpOrder, lpCut);
nl_a1 = 1.0; nl_a3 = 0.02;  % 轻微三阶畸变

chan_train = filter(h_lp,1, tx_train); chan_train = nl_a1*chan_train + nl_a3*chan_train.^3;
chan_test  = filter(h_lp,1, tx_test);  chan_test  = nl_a1*chan_test  + nl_a3*chan_test.^3;

% AWGN
signalPower = mean(chan_train.^2);
SNR_lin = 10^(SNR_dB/10);
noiseVar = signalPower / SNR_lin;
noiseStd = sqrt(noiseVar);
rx_train = chan_train + noiseStd*randn(size(chan_train));
rx_test  = chan_test  + noiseStd*randn(size(chan_test));

%% ----------------- 匹配滤波 + 下采样（接收端） -----------------
rrc_rx = rrc_tx;  % 匹配滤波
rx_matched_train = conv(rx_train, rrc_rx);
rx_matched_test  = conv(rx_test,  rrc_rx);

% 简单定时：直接按发射整形的中心抽样（与原脚本一致的简化方案）
startIdx = floor(length(rrc_tx)/2) + 1;  % RRC 脉冲主峰附近
rx_sym_train = rx_matched_train(startIdx : sps : startIdx + sps*(numTrain-1))';
rx_sym_test  = rx_matched_test(startIdx  : sps : startIdx + sps*(numTest-1))';
rx_sym_train = rx_sym_train(:); rx_sym_test = rx_sym_test(:);

%% ----------------- 构建 WD-RNN 的训练输入（teacher forcing: 用 labels 作为延迟反馈） -----------------
% 构造输入：每个训练符号 i 的输入为 [n0 窗口样点; 过去 k_delay 个“标签”]
padL = floor(n0/2); padR = n0 - padL - 1;
rx_train_pad = [zeros(padL,1); rx_sym_train; zeros(padR+k_delay,1)];
Ntrain = numTrain;
inputDim = n0 + k_delay;

Xtrain = zeros(inputDim, Ntrain, 'single');
Ttrain = zeros(1, Ntrain, 'single');

for i=1:Ntrain
    idx_center = i + padL;
    window = rx_train_pad(idx_center - floor(n0/2) : idx_center + ceil(n0/2)-1);
    if k_delay>0
        prevLabels = zeros(k_delay,1);
        for kk=1:k_delay
            if i-kk >= 1, prevLabels(kk) = symb_train(i-kk); else, prevLabels(kk) = 0; end
        end
    else
        prevLabels = [];
    end
    Xtrain(:,i) = single([window(:); prevLabels(:)]);
    Ttrain(1,i) = single(symb_train(i));
end

% 训练标签的“类别索引”（1..4），供分类头用
% 映射：-3->1, -1->2, 1->3, 3->4
level2idx = containers.Map({-3,-1,1,3}, {1,2,3,4});
yclassTrain = arrayfun(@(v) level2idx(v), symb_train);

%% ----------------- 定义并初始化网络参数（WD-RNN-CLS，回归+分类双头） -----------------
% 结构：inputDim -> n1 (tanh) -> [回归头 1；分类头 4]
rng(rngSeed);  
W1  = 0.1*randn(n1, inputDim, 'single');   b1  = zeros(n1,1,'single');
W2y = 0.1*randn(1, n1, 'single');          b2y = zeros(1,1,'single');
W2c = 0.05*randn(4, n1,'single');          b2c = zeros(4,1,'single');

% 将数据/参数放到 GPU（如可用）
if useGPU
    Xtrain = gpuArray(Xtrain); Ttrain = gpuArray(Ttrain);
    W1 = gpuArray(W1); b1 = gpuArray(b1);
    W2y = gpuArray(W2y); b2y = gpuArray(b2y);
    W2c = gpuArray(W2c); b2c = gpuArray(b2c);
end

%% ----------------- 训练：自写 Adam（mini-batch），损失=MSE+lambda_ce*CE -----------------
% 我们在训练时对隐藏层应用极小 dropout（rate = 0.01）以贴近论文设置
mW1 = zeros(size(W1),'like',W1); vW1 = mW1;
mb1 = zeros(size(b1),'like',b1); vb1 = mb1;
mW2y = zeros(size(W2y),'like',W2y); vW2y = mW2y;
mb2y = zeros(size(b2y),'like',b2y); vb2y = mb2y;
mW2c = zeros(size(W2c),'like',W2c); vW2c = mW2c;
mb2c = zeros(size(b2c),'like',b2c); vb2c = mb2c;
beta1 = 0.9; beta2 = 0.999; epsAdam = 1e-8;
iter = 0;
numBatches = ceil(Ntrain / miniBatch);

fprintf('Start training: epochs=%d, batches per epoch=%d\n', maxEpochs, numBatches);

for epoch = 1:maxEpochs
    % shuffle
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

        % forward (vectorized, dual-head)
        Z1 = W1 * Xbatch + b1;          % n1 x batch
        H1 = tanh(Z1);                  % activation
        % dropout on H1 during training (inverted dropout)
        if dropout_rate > 0
            mask = (rand(size(H1)) > dropout_rate);
            if useGPU, mask = gpuArray(mask); end
            H1 = H1 .* mask / (1-dropout_rate);
        end
        Yreg   = W2y * H1 + b2y;        % 1 x batch (regression head)
        Logits = W2c * H1 + b2c;        % 4 x batch (classification head)

        % stable softmax
        Logits = Logits - max(Logits,[],1);
        ExpL   = exp(Logits);
        Pcls   = ExpL ./ sum(ExpL,1);   % 4 x batch

        % loss: MSE + lambda_ce*CE
        err    = Yreg - Tbatch;                         
        mseLoss = mean(err.^2, 'all');

        % gather labels for this batch (1..4)
        cols = 1:size(Xbatch,2);
        labels = yclassTrain(idx(i1:i2));    % int32 1..4
        if useGPU, labels = gpuArray(labels); end
        I = sub2ind(size(Pcls), double(labels(:))', cols);
        ceLoss = -mean( log( max(Pcls(I), realmin('single')) ) );

        loss = mseLoss + lambda_ce * ceLoss;

        % gradients
        batchSizeCurr = size(Xbatch,2);

        dYreg = (2/batchSizeCurr) * err;        % 1 x batch
        Yoh   = zeros(4, batchSizeCurr, 'like', Pcls); Yoh(I) = 1;
        dLogits = lambda_ce * (Pcls - Yoh) / batchSizeCurr;  % 4 x batch

        dH1 = (W2y' * dYreg) + (W2c' * dLogits);            % n1 x batch

        if dropout_rate > 0
            dH1 = dH1 .* mask / (1-dropout_rate);
        end

        dZ1 = dH1 .* (1 - tanh(Z1).^2);

        dW2y = dYreg * H1';               db2y = sum(dYreg,2);
        dW2c = dLogits * H1';             db2c = sum(dLogits,2);
        dW1  = dZ1 * Xbatch';             db1  = sum(dZ1,2);

        % Adam update (element-wise)
        % W1
        mW1 = beta1*mW1 + (1-beta1)*dW1;
        vW1 = beta2*vW1 + (1-beta2)*(dW1.^2);
        mhatW1 = mW1 / (1 - beta1^iter);
        vhatW1 = vW1 / (1 - beta2^iter);
        W1 = W1 - learnRate * mhatW1 ./ (sqrt(vhatW1) + epsAdam);

        % b1
        mb1 = beta1*mb1 + (1-beta1)*db1;
        vb1 = beta2*vb1 + (1-beta2)*(db1.^2);
        mbhat = mb1 / (1 - beta1^iter);
        vbhat = vb1 / (1 - beta2^iter);
        b1 = b1 - learnRate * mbhat ./ (sqrt(vbhat) + epsAdam);

        % W2y
        mW2y = beta1*mW2y + (1-beta1)*dW2y;
        vW2y = beta2*vW2y + (1-beta2)*(dW2y.^2);
        mhatW2y = mW2y / (1 - beta1^iter);
        vhatW2y = vW2y / (1 - beta2^iter);
        W2y = W2y - learnRate * mhatW2y ./ (sqrt(vhatW2y) + epsAdam);

        % b2y
        mb2y = beta1*mb2y + (1-beta1)*db2y;
        vb2y = beta2*vb2y + (1-beta2)*(db2y.^2);
        mb2yhat = mb2y / (1 - beta1^iter);
        vb2yhat = vb2y / (1 - beta2^iter);
        b2y = b2y - learnRate * mb2yhat ./ (sqrt(vb2yhat) + epsAdam);

        % W2c
        mW2c = beta1*mW2c + (1-beta1)*dW2c;
        vW2c = beta2*vW2c + (1-beta2)*(dW2c.^2);
        mhatW2c = mW2c / (1 - beta1^iter);
        vhatW2c = vW2c / (1 - beta2^iter);
        W2c = W2c - learnRate * mhatW2c ./ (sqrt(vhatW2c) + epsAdam);

        % b2c
        mb2c = beta1*mb2c + (1-beta1)*db2c;
        vb2c = beta2*vb2c + (1-beta2)*(db2c.^2);
        mb2chat = mb2c / (1 - beta1^iter);
        vb2chat = vb2c / (1 - beta2^iter);
        b2c = b2c - learnRate * mb2chat ./ (sqrt(vb2chat) + epsAdam);
    end

    % 每个 epoch 打印损失（近似为最后一个 batch 的 loss）
    if useGPU, lossVal = gather(double(loss)); else, lossVal = double(loss); end
    fprintf('Epoch %d/%d - Loss(approx last batch): %.5e (MSE+%g*CE) \n', epoch, maxEpochs, lossVal, lambda_ce);
end

fprintf('Training finished. \n');

%% ----------------- 测试：递归推理（用 WD 加权判决作为反馈） -----------------
% 按论文测试范式：无需标签，递归反馈 \~y
padL = floor(n0/2); padR = n0 - padL - 1;
Ntest = length(rx_sym_test);
rx_test_pad = [zeros(padL,1); rx_sym_test; zeros(padR,1)];
feedbackBuf = zeros(k_delay,1,'like',W1);

predSymbols = zeros(Ntest,1);
predLevels  = zeros(Ntest,1);
eqOut = zeros(Ntest,1);
[Pol_X,epsilon_x]=CMA(rx_test_pad,25) ;
for i = 1:Ntest
    idx_center = i + padL;
    window = rx_test_pad(idx_center - floor(n0/2) : idx_center + ceil(n0/2)-1);

    % 过去 k 个反馈
    prevLabels = zeros(k_delay,1);
    for kk=1:k_delay
        if (i-kk) >= 1, prevLabels(kk) = feedbackBuf(kk); else, prevLabels(kk) = 0; end
    end
    xin = [window(:); prevLabels(:)];
    if useGPU, xin = gpuArray(single(xin)); else, xin = single(xin); end

    % 前向（双头）
    z1 = W1 * xin + b1;
    h1 = tanh(z1);
    y = W2y * h1 + b2y;              % 回归输出
    logits = W2c * h1 + b2c;         % 分类 logits
    if useGPU, y = gather(y); logits = gather(logits); end
    y = double(y); logits = double(logits);
    eqOut(i) = y;

    % 分类置信度
    logits = logits - max(logits);
    p = exp(logits) ./ sum(exp(logits));
    conf = max(p);                   % (0,1)

    % 硬判决与 gamma
    [~, idxMin] = min(abs(y - pam4_levels));
    yhat = pam4_levels(idxMin);
    gamma = 1 - min(abs(y - yhat), 1);

    % 融合置信度得到 g，并计算 S(g)
    g = lambda_mix * gamma + (1 - lambda_mix) * conf;
    Sg = 0.5 * ( 1 - exp(-alpha_wd*(g/beta_wd - 1)) ./ (1 + exp(-alpha_wd*(g/beta_wd - 1))) + 1 );

    % 加权判决反馈
    ytilde = Sg * yhat + (1 - Sg) * y;

    % 更新反馈缓冲：最近的作为第1个
    if k_delay >= 1
        feedbackBuf = [ytilde; feedbackBuf(1:end-1)];
    end

    predSymbols(i) = yhat;
    predLevels(i)  = yhat;
end

subplot(3,1,1);
plot(rx_test_pad ,'.')
title('无均衡')
hold on
subplot(3,1,2)
title('CMA均衡')
plot(Pol_X ,'m.')
subplot(3,1,3)
plot(eqOut,'r.')
title('WD-RNN均衡')
hold off
%% ----------------- 误码统计（符号/比特） -----------------
% 符号误码率
symErrs = sum(predSymbols ~= symb_test);
SER = symErrs / length(symb_test);
fprintf('SER (symbol error rate) = %.6f\\n', SER);

% 比特误码率（以 Gray 反映射近似）
% 反灰映射：-3->00, -1->01, 1->11, 3->10
invMap = containers.Map({-3,-1,1,3}, {'00','01','11','10'});
trueBits = strings(length(symb_test),1); 
predBits = strings(length(predSymbols),1);
for i=1:length(symb_test)
    trueBits(i) = invMap(symb_test(i));
    predBits(i) = invMap(predSymbols(i));
end
trueBits = join(trueBits, ""); trueBits = char(trueBits);
predBits = join(predBits, ""); predBits = char(predBits);
trueBits = trueBits(:) - '0'; predBits = predBits(:) - '0';
bitErrs = sum(predBits ~= trueBits);
BER_bit = bitErrs / length(trueBits);
fprintf('BER (bit error rate) = %.6f\n', BER_bit);

%% ----------------- 可选：权重剪枝（weight pruning）并再次评估 -----------------
pruning_ratios = [0.0, 0.2, 0.4]; % 包含 p=0 表示不剪枝

for pi = 1:length(pruning_ratios)
    p = pruning_ratios(pi);
    if p==0
        W1p = W1; W2yp = W2y; W2cp = W2c;
    else
        % 合并所有权重并求阈值
        if useGPU
            allW = gather(abs([W1(:); W2y(:); W2c(:)]));
        else
            allW = abs([W1(:); W2y(:); W2c(:)]);
        end
        th = prctile(allW, p*100);
        % 剪枝
        W1p = W1; W2yp = W2y; W2cp = W2c;
        W1p(abs(W1p) < th) = 0;
        W2yp(abs(W2yp) < th) = 0;
        W2cp(abs(W2cp) < th) = 0;
    end
    fprintf('Pruning ratio p=%.2f applied. (To fully evaluate, re-run test inference with pruned weights.)\n', p);
end

fprintf('Script complete.\n');