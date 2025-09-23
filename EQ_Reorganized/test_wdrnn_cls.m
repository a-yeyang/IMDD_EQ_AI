function SER_rnn_cls = test_wdrnn_cls(modelFile, rx_sym_test, symb_test, SNR_dB_list, options)
% test_wdrnn - 测试WD-RNN-CLS神经网络均衡器性能
%
% 输入参数：
%   modelFile    - 训练好的WD-RNN-CLS模型文件路径
%   rx_sym_test  - 测试接收符号
%   symb_test    - 测试符号标签
%   SNR_dB_list  - 测试SNR列表
%   options      - 可选参数结构体（可选）
%     .useGPU        - 是否使用GPU (默认: false)
%
% 输出参数：
%   SER_rnn_cls      - WD-RNN-CLS均衡器的误符号率

%% 参数解析
if nargin < 5
    options = struct();
end

% 设置默认选项
if ~isfield(options, 'useGPU'), options.useGPU = false; end

useGPU = options.useGPU;

%% GPU检测和设置
if useGPU
    try
        gcount = gpuDeviceCount;
        if gcount >= 1
            gpuInfo = gpuDevice;
            fprintf('GPU detected: %s. Will use GPU arrays.\n', gpuInfo.Name);
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
    fprintf('GPU disabled. Running on CPU.\n');
end

%% 加载WD-RNN-CLS模型
fprintf('Loading WD-RNN-CLS model from %s\n', modelFile);
try
    modelData = load(modelFile, 'W1','b1','W2y','b2y','W2c','b2c', ...
        'n0','n1','k_delay','pam4_levels','alpha_wd','beta_wd','lambda_mix');
    
    W1 = modelData.W1; b1 = modelData.b1;
    W2y = modelData.W2y; b2y = modelData.b2y;
    W2c = modelData.W2c; b2c = modelData.b2c;
    n0 = modelData.n0; k_delay = modelData.k_delay;
    pam4_levels = modelData.pam4_levels;
    alpha_wd = modelData.alpha_wd; beta_wd = modelData.beta_wd;
    lambda_mix = modelData.lambda_mix;
    
    fprintf('WD-RNN-CLS model loaded successfully.\n');
catch ME
    error('Failed to load WD-RNN-CLS model: %s', ME.message);
end

%% 初始化结果数组
numSNR = length(SNR_dB_list);
SER_rnn_cls = zeros(numSNR, 1);

fprintf('开始测试WD-RNN-CLS，SNR范围：[%s] dB\n', num2str(SNR_dB_list));

%% 并行测试不同SNR
parfor i = 1:numSNR
    snr_db = SNR_dB_list(i);
    fprintf('Testing WD-RNN-CLS at SNR = %d dB...\n', snr_db);
    
    % 添加噪声
    rx_sym_test_snr = awgn(rx_sym_test, snr_db);
    
    % 测试WD-RNN-CLS
    SER_rnn_cls(i) = test_wdrnn_single_snr(rx_sym_test_snr, symb_test, ...
        W1, b1, W2y, b2y, W2c, b2c, n0, k_delay, pam4_levels, ...
        alpha_wd, beta_wd, lambda_mix, useGPU);
    
    fprintf('SNR %d dB: WD-RNN-CLS SER_rnn_cls = %.2e\n', snr_db, SER_rnn_cls(i));
end

%% 输出测试结果
fprintf('\n=== WD-RNN-CLS测试结果 ===\n');
fprintf('SNR(dB)\tWD-RNN-CLS SER\n');
fprintf('--------------------\n');
for i = 1:numSNR
    fprintf('%d\t%.2e\n', SNR_dB_list(i), SER_rnn_cls(i));
end

fprintf('\nWD-RNN-CLS测试完成！\n');

end

%% 辅助函数：单个SNR的WD-RNN-CLS测试
function SER_rnn_cls = test_wdrnn_single_snr(rx_sym_test_snr, symb_test, ...
    W1, b1, W2y, b2y, W2c, b2c, n0, k_delay, pam4_levels, ...
    alpha_wd, beta_wd, lambda_mix, useGPU)

% 准备输入数据
padL = floor(n0/2); padR = n0 - padL - 1;
Ntest = length(rx_sym_test_snr);
rx_test_pad = [zeros(padL,1); rx_sym_test_snr; zeros(padR,1)];
feedbackBuf = zeros(k_delay,1,'like',W1);

predSymbols = zeros(Ntest,1);

% 递归推理
for j = 1:Ntest
    idx_center = j + padL;
    window = rx_test_pad(idx_center - floor(n0/2) : idx_center + ceil(n0/2)-1);
    
    % 过去 k 个反馈
    prevLabels = zeros(k_delay,1);
    for kk = 1:k_delay
        if (j-kk) >= 1
            prevLabels(kk) = feedbackBuf(kk);
        else
            prevLabels(kk) = 0;
        end
    end
    
    xin = [window(:); prevLabels(:)];
    if useGPU
        xin = gpuArray(single(xin));
    else
        xin = single(xin);
    end
    
    % 前向传播（双头）
    z1 = W1 * xin + b1;
    h1 = tanh(z1);
    y = W2y * h1 + b2y;              % 回归输出
    logits = W2c * h1 + b2c;         % 分类 logits
    
    if useGPU
        y = gather(y);
        logits = gather(logits);
    end
    y = double(y);
    logits = double(logits);
    
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
    Sg = 0.5 * (1 - exp(-alpha_wd*(g/beta_wd - 1)) ./ (1 + exp(-alpha_wd*(g/beta_wd - 1))) + 1);
    
    % 加权判决反馈
    ytilde = Sg * yhat + (1 - Sg) * y;
    
    % 更新反馈缓冲：最近的作为第1个
    if k_delay >= 1
        feedbackBuf = [ytilde; feedbackBuf(1:end-1)];
    end
    
    predSymbols(j) = yhat;
end

% 计算误符号率
symErrs = sum(predSymbols ~= symb_test);
SER_rnn_cls = symErrs / length(symb_test);

end
