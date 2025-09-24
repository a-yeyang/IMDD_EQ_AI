function modelFile=wd_rnn(rx_sym_train,symb_train)
%% ----------------- 仿真/网络 超参数（可按需调整） -----------------
useGPU = true;            % 是否使用 GPU（自动检测）；若你的 MATLAB/平台 不支持 GPU，可设 false
rngSeed = 12345;          % 随机数种子（使用 Mersenne Twister）
rng(rngSeed, 'twister');

%% 输入数据处理
rx_sym_train = rx_sym_train(:);  % 确保是列向量
symb_train = symb_train(:);      % 确保是列向量
Ntrain = length(symb_train);

%% WD-RNN 架构超参（采用论文优选）
n0 = 61;                 % 输入节点数（接收窗口样点数）。论文最优为 61。
n1 = 20;                 % 隐藏层神经元数（论文最优为 20）。
k_delay = 6;             % 延迟单元 k=6（论文推荐折中值）。

alpha_wd = 5; beta_wd = 0.14;  % WD 参数（论文建议）.
dropout_rate = 0.01;           % dropout（训练时）.

%% 训练超参
maxEpochs = 30;         % 论文选择 30 epochs（WD-RNN）.
miniBatch = 1024;       % 小批量大小
learnRate = 1e-3;       % Adam 初始学习率

%% PAM4 符号列表（论文用四电平 {-3,-1,1,3}）
pam4_levels = [-3, -1, 1, 3];

%% GPU 可用性检测
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

%% ----------------- 构建 WD-RNN 的训练输入（teacher forcing: 用 labels 作为延迟反馈） -----------------
% 输入向量 = [接收窗口 n0 个样点; k 个延迟 label]  -> 大小 (n0 + k) x N
% 这里我们选择窗口为当前符号及其前 (n0-1) 的样点（等价于过去窗口），也可用对称窗口。
% 为简便，先 pad rx 矢量两端以便索引。

padL = floor(n0/2);
padR = n0 - padL - 1;
rx_train_pad = [zeros(padL,1); rx_sym_train; zeros(padR+k_delay,1)]; % pad 额外保证延迟取值安全

inputDim = n0 + k_delay;

Xtrain = zeros(inputDim, Ntrain);
Ttrain = zeros(1, Ntrain);

for i=1:Ntrain
    idx_center = i + padL;
    window = rx_train_pad(idx_center - floor(n0/2) : idx_center + ceil(n0/2)-1);
    % labels for previous k (teacher forcing uses true labels)
    if k_delay>0
        prevLabels = zeros(k_delay,1);
        for kk=1:k_delay
            if i-kk >= 1
                prevLabels(kk) = symb_train(i-kk);
            else
                prevLabels(kk) = 0; % 边界初始用 0（也可用前置填充）
            end
        end
    else
        prevLabels = [];
    end
    Xtrain(:,i) = [window(:); prevLabels(:)];
    Ttrain(1,i) = symb_train(i);
end

%% ----------------- 定义并初始化网络参数（手写小网络，便于精确控制） -----------------
% 网络：inputDim -> n1 (tanh) -> 1 (linear)
% 参数维度：
% W1: n1 x inputDim
% b1: n1 x 1
% W2: 1 x n1
% b2: 1 x 1

rng(rngSeed);  % 保持可复现的初始权重
W1 = 0.1*randn(n1, inputDim);
b1 = zeros(n1,1);
W2 = 0.1*randn(1, n1);
b2 = 0;

% 将数据/参数放到 GPU（如可用）
if useGPU
    Xtrain = gpuArray(single(Xtrain));
    Ttrain = gpuArray(single(Ttrain));
    W1 = gpuArray(single(W1));
    b1 = gpuArray(single(b1));
    W2 = gpuArray(single(W2));
    b2 = gpuArray(single(b2));
end

%% ----------------- 训练：自写 Adam 优化器（mini-batch） -----------------
% 我们在训练时对隐藏层应用 dropout（rate = dropout_rate）来匹配论文设定（dropout=0.01）
% Loss: MSE between predicted y and target symbol

% Adam 初始化
mW1 = zeros(size(W1),'like',W1); vW1 = mW1;
mb1 = zeros(size(b1),'like',b1); vb1 = mb1;
mW2 = zeros(size(W2),'like',W2); vW2 = mW2;
mb2 = zeros(size(b2),'like',b2); vb2 = mb2;
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

        % forward (vectorized)
        Z1 = W1 * Xbatch + b1;          % n1 x batch
        H1 = tanh(Z1);                  % activation
        % dropout on H1 during training
        if dropout_rate > 0
            mask = (rand(size(H1)) > dropout_rate);
            if useGPU, mask = gpuArray(mask); end
            H1 = H1 .* mask / (1-dropout_rate); % inverted dropout
        end
        Ypred = W2 * H1 + b2;           % 1 x batch

        % loss and grads (MSE)
        err = Ypred - Tbatch;
        % --- 新增 Symbol-Aware Loss ---
        pam4_levels = [-3, -1, 1, 3];
        weights = ones(size(err), 'like', err);
        sigma_sq = 0.5; % 可调超参数
        
        for i = 1:numel(Tbatch)
            true_level = Tbatch(i);
            pred_level = Ypred(i);
        
            % 找到最近的错误电平
            dists = abs(true_level - pam4_levels);
            [~, sorted_idx] = sort(dists);
            neighbor_levels = pam4_levels(sorted_idx(2:end)); % 排除真实电平
        
            % 仅考虑最危险的邻近错误电平
            [~, min_dist_idx] = min(abs(pred_level - neighbor_levels));
            L_neighbor = neighbor_levels(min_dist_idx);
        
            % 计算惩罚权重
            penalty = exp(-(pred_level - L_neighbor)^2 / sigma_sq);
            weights(i) = 1 + penalty;
        end
        % --- 结束 ---
        % 应用权重
        weighted_err = err .* weights;



        loss = mean(weighted_err.^2, 'all');     % scalar

        % compute gradients (manual differentiation for small net)
        % dLoss/dYpred = 2*(Ypred - T) / batchSize
        batchSizeCurr = size(Xbatch,2);
        dY = (2/batchSizeCurr) * weighted_err;   % 1 x batch

        % grads for W2, b2
        dW2 = dY * H1';                 % 1 x n1
        db2 = sum(dY,2);                % 1 x 1

        % backprop to H1
        dH1 = (W2') * dY;               % n1 x batch
        dZ1 = dH1 .* (1 - H1.^2);       % tanh' = 1 - tanh^2

        % grads for W1,b1
        dW1 = dZ1 * Xbatch';            % n1 x inputDim
        db1 = sum(dZ1,2);               % n1 x 1

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

        % W2
        mW2 = beta1*mW2 + (1-beta1)*dW2;
        vW2 = beta2*vW2 + (1-beta2)*(dW2.^2);
        mhatW2 = mW2 / (1 - beta1^iter);
        vhatW2 = vW2 / (1 - beta2^iter);
        W2 = W2 - learnRate * mhatW2 ./ (sqrt(vhatW2) + epsAdam);

        % b2
        mb2 = beta1*mb2 + (1-beta1)*db2;
        vb2 = beta2*vb2 + (1-beta2)*(db2.^2);
        mb2hat = mb2 / (1 - beta1^iter);
        vb2hat = vb2 / (1 - beta2^iter);
        b2 = b2 - learnRate * mb2hat ./ (sqrt(vb2hat) + epsAdam);
    end

    % 每个 epoch 打印损失（估计用最后一个 batch 的 loss）
    if useGPU
        lossVal = gather(double(loss));
    else
        lossVal = double(loss);
    end
    fprintf('Epoch %d/%d - Loss(approx last batch): %.5e\n', epoch, maxEpochs, lossVal);
end

fprintf('Training finished.\n');

%% ----------------- 保存模型 -----------------
modelFile = 'wd_rnn_model.mat';
if useGPU
    W1 = gather(W1); b1 = gather(b1);
    W2 = gather(W2); b2 = gather(b2);
end
save(modelFile, 'W1','b1','W2','b2', ...
    'n0','n1','k_delay','pam4_levels','alpha_wd','beta_wd','dropout_rate','-v7.3');
fprintf('Model saved to %s\n', modelFile);
end