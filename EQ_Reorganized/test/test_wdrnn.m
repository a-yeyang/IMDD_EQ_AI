function [SER, BER, eqOut, predLevels] = test_wdrnn(modelFile, rx_sym_test, symb_test)
%% test_wdrnn - 使用训练好的WD-RNN模型进行测试
% 输入：
%   modelFile - 训练好的模型文件路径 (例如: 'wd_rnn_model.mat')
%   rx_sym_test - 测试接收信号 (列向量)
%   symb_test - 测试符号标签 (列向量)
% 输出：
%   SER - 符号误差率
%   BER - 比特误差率  
%   eqOut - 均衡器输出信号
%   predLevels - 预测的符号电平

%% 输入数据处理
rx_sym_test = rx_sym_test(:);  % 确保是列向量
symb_test = symb_test(:);      % 确保是列向量
numTest = length(symb_test);

%% 加载训练好的模型
if ~exist(modelFile, 'file')
    error('模型文件不存在: %s', modelFile);
end

fprintf('加载模型文件: %s\n', modelFile);
model = load(modelFile);

% 提取模型参数
W1 = model.W1;
b1 = model.b1;
W2 = model.W2;
b2 = model.b2;
n0 = model.n0;
n1 = model.n1;
k_delay = model.k_delay;
pam4_levels = model.pam4_levels;
alpha_wd = model.alpha_wd;
beta_wd = model.beta_wd;

fprintf('模型参数加载完成:\n');
fprintf('  输入维度: %d (n0=%d + k_delay=%d)\n', n0+k_delay, n0, k_delay);
fprintf('  隐藏层大小: %d\n', n1);
fprintf('  PAM4 电平: [%s]\n', num2str(pam4_levels));
fprintf('  WD 参数: alpha=%.2f, beta=%.2f\n', alpha_wd, beta_wd);

%% GPU 可用性检测
useGPU = false;
try
    gcount = gpuDeviceCount;
    if gcount >= 1
        gpuInfo = gpuDevice;
        fprintf('GPU detected: %s. Will use GPU arrays.\n', gpuInfo.Name);
        useGPU = true;
        % 将模型参数移到GPU
        W1 = gpuArray(single(W1));
        b1 = gpuArray(single(b1));
        W2 = gpuArray(single(W2));
        b2 = gpuArray(single(b2));
    else
        fprintf('No GPU detected. Running on CPU.\n');
    end
catch
    fprintf('GPU detection failed. Running on CPU.\n');
end

%% ----------------- 测试：递归推理（用 WD 加权判决作为反馈） -----------------
% 我们按论文式 (2)-(4) 在测试时使用 weighted decision 反馈（没有 labels）
% 测试流程：对每个测试符号 i：
%  1) 从 rx_sym_test 中取 n0 窗口样点（使用 pad 边界）
%  2) 收集过去 k 个 feedback（开始时用 0 填充）
%  3) 前向预测 y，进行硬判决 y_hat（最接近的 pam4 level）
%  4) 计算 gamma = 1 - |y - y_hat|（若 y 超出 [-3,3], gamma=1）
%  5) 计算 S(gamma) 按论文 (4)，得到 y_tilde = S* y_hat + (1-S)* y
%  6) 将 y_tilde 存入 feedback buffer，用于后续样点

fprintf('\n开始WD-RNN均衡测试...\n');

% 准备
padL = floor(n0/2);
padR = n0 - padL - 1;
rx_test_pad = [zeros(padL,1); rx_sym_test; zeros(padR,1)];
Ntest = numTest;

% feedback buffer：最近 k 个 y_tilde（初始化为 0）
feedbackBuf = zeros(k_delay,1);
if useGPU
    feedbackBuf = gpuArray(single(feedbackBuf));
end

predLevels = zeros(Ntest,1);
eqOut = zeros(Ntest,1); 

fprintf('处理 %d 个测试符号...\n', Ntest);

for i = 1:Ntest
    % if mod(i, 1000) == 0
    %     fprintf('  处理进度: %d/%d\n', i, Ntest);
    % end
    
    idx_center = i + padL;
    window = rx_test_pad(idx_center - floor(n0/2) : idx_center + ceil(n0/2)-1);
    
    % 构建延迟反馈
    prevLabels = zeros(k_delay,1);
    for kk=1:k_delay
        if (i-kk) >= 1
            prevLabels(kk) = feedbackBuf(kk); % 这里反馈用上一步计算出的 y_tilde（最新的）
        else
            prevLabels(kk) = 0;
        end
    end
    
    % 构建输入向量
    xin = [window(:); prevLabels(:)];
    if useGPU
        xin = gpuArray(single(xin));
    else
        xin = single(xin);
    end

    % 前向传播
    z1 = W1 * xin + b1;
    h1 = tanh(z1);
    y = W2 * h1 + b2;  % 1x1
    
    if useGPU
        y = gather(y);
    end
    y = double(y);
    eqOut(i) = y;
    
    % 硬判决（Nearest PAM level）
    [~, idxMin] = min(abs(y - pam4_levels));
    yhat = pam4_levels(idxMin);

    % gamma 计算（论文式(3)）：当 y 在 [-3,3] 之间按 1 - |y - yhat|，超范围则 gamma=1
    if y < -3 || y > 3
        gamma = 1;
    else
        gamma = 1 - abs(y - yhat);
        gamma = max(0, min(1, gamma)); % 限幅
    end

    % S(x) 压缩 sigmoid（论文式(4)）
    Sx = 0.5 * ( 1 - exp(-alpha_wd*(gamma/beta_wd - 1)) ./ (1 + exp(-alpha_wd*(gamma/beta_wd - 1))) + 1 );
    % 注意：上式按论文直接实现；数值边界小心
    Sx = double(Sx);

    ytilde = Sx * yhat + (1 - Sx) * y;

    % 更新 feedbackBuf（最新放最前）
    if k_delay > 0
        if useGPU
            ytilde_gpu = gpuArray(single(ytilde));
            feedbackBuf = [ytilde_gpu; feedbackBuf(1:end-1)];
        else
            feedbackBuf = [ytilde; feedbackBuf(1:end-1)];
        end
    end

    % 保存预测
    predLevels(i) = yhat;
end

fprintf('WD-RNN均衡处理完成.\n');

%% 计算误差率
% 符号误差率
symErr = sum(predLevels ~= symb_test);
SER = symErr / Ntest;

fprintf('\n=== 测试结果 ===\n');
fprintf('符号误差数: %d / %d\n', symErr, Ntest);
fprintf('符号误差率 (SER): %.6f (%.2e)\n', SER, SER);

% 比特误差率（基于 Gray 映射）
% Gray mapping: 00->-3, 01->-1, 11->1, 10->3
invMap = containers.Map([-3 -1 1 3], {'00','01','11','10'});

% 生成真实比特和预测比特
trueBits = zeros(2*Ntest,1,'uint8');
predBits = zeros(2*Ntest,1,'uint8');

for i=1:Ntest
    % 真实符号对应的比特
    str_true = invMap(symb_test(i));
    trueBits(2*i-1) = uint8(str2double(str_true(1))); % 第1位
    trueBits(2*i)   = uint8(str2double(str_true(2))); % 第2位
    
    % 预测符号对应的比特
    str_pred = invMap(predLevels(i));
    predBits(2*i-1) = uint8(str2double(str_pred(1))); % 第1位
    predBits(2*i)   = uint8(str2double(str_pred(2))); % 第2位
end

% 计算比特误差率
bitErrs = sum(predBits ~= trueBits);
BER = bitErrs / length(trueBits);

fprintf('比特误差数: %d / %d\n', bitErrs, length(trueBits));
fprintf('比特误差率 (BER): %.6f (%.2e)\n', BER, BER);

fprintf('\n测试完成！\n');

end