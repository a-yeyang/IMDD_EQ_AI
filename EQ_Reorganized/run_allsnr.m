% pam4_wdrnn_cls.m
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


% 训练SNR（固定）
train_SNR_dB = 10;

%% 测试SNR值
SNR_dB_list = [28,24,20,16,12,8,4];
numSNR = length(SNR_dB_list);
%% ===================================================
rx=-load('vpi_data.txt');
rx=2*(rx-mean(rx))/mean(abs(rx));
%% ===================================================
rx = lowpass(rx, 25e9, 120e9);
rx_train=rx(1:nSymbols_train*sps);
rx_test=rx(nSymbols_test*sps+1:end);

%% ----------------- 匹配滤波+下采样 ----------------- 
rx_matched_train = conv(rx_train, rrc,'same');
rx_matched_test  = conv(rx_test,  rrc,'same');
rx_sym_train = resample(rx_matched_train,Rs,Fs)';
rx_sym_test  = resample(rx_matched_test,Rs,Fs)';
symb_train=load('symb_train.txt');
symb_test=load('symb_test.txt');
modelFile=wd_rnn_cls(rx_sym_train,symb_train)
%% ----------------- 加载模型 -----------------
clear W1 b1 W2y b2y W2c b2c
load(modelFile, 'W1','b1','W2y','b2y','W2c','b2c', ...
    'n0','n1','k_delay','pam4_levels','alpha_wd','beta_wd','lambda_mix');
fprintf('Model loaded from %s\n', modelFile);

%% ----------------- 测试阶段 (递归推理/误符号率统计/绘图) -----------------
% 保持和你原来的一致 ...
% (此处省略测试推理部分，直接接你原来的测试代码即可)
 

%% ----------------- 测试：递归推理（用 WD 加权判决作为反馈） -----------------
% 按论文测试范式：无需标签，递归反馈 \~y
SER_no=zeros(numel(SNR_dB_list),1);
SER_cma=zeros(numel(SNR_dB_list),1);
SER_rnn_cls=zeros(numel(SNR_dB_list),1);

parfor i=1:numel(SNR_dB_list)
    rx_sym_test_snr=awgn(rx_sym_test,SNR_dB_list(i))
    padL = floor(n0/2); padR = n0 - padL - 1;
    Ntest = length(rx_sym_test_snr);
    rx_test_pad = [zeros(padL,1); rx_sym_test_snr; zeros(padR,1)];
    feedbackBuf = zeros(k_delay,1,'like',W1);
    
    predSymbols = zeros(Ntest,1);
    predLevels  = zeros(Ntest,1);
    eqOut = zeros(Ntest,1);
    Pol_X=3*CMA(rx_sym_test_snr,25)' ;
    rx_sym_test1=rx_sym_test_snr';
    t=2;
    symb_cma=sign(Pol_X) + (Pol_X==0) + 2*(Pol_X>t) - 2*(Pol_X<-t);
    symb_no=sign(rx_sym_test_snr) + (rx_sym_test_snr==0) + 2*(rx_sym_test_snr>1) - 2*(rx_sym_test_snr<-1);
    SER_cma(i)=sum(symb_cma ~= symb_test)/length(symb_test)
    SER_no(i)=sum(symb_no ~= symb_test)/length(symb_test)
    for j = 1:Ntest
        idx_center = j + padL;
        window = rx_test_pad(idx_center - floor(n0/2) : idx_center + ceil(n0/2)-1);
    
        % 过去 k 个反馈
        prevLabels = zeros(k_delay,1);
        for kk=1:k_delay
            if (j-kk) >= 1, prevLabels(kk) = feedbackBuf(kk); else, prevLabels(kk) = 0; end
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
    
        predSymbols(j) = yhat;
        predLevels(j)  = yhat;
    end
    symErrs = sum(predSymbols ~= symb_test);
    SER_rnn_cls(i) = symErrs / length(symb_test)
end

fprintf('SER_rnn_cls (symbol error rate) = %.6f\n', SER_rnn_cls);
fprintf('SER_cma (symbol error rate) = %.6f\n', SER_cma);
%% ----------------- 绘制比较图 -----------------
fprintf('\n绘制SNR vs SER比较图...\n');

figure('Position', [100, 100, 800, 600]);
semilogy(SNR_dB_list, SER_no, 'ro-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '无均衡器');
hold on;
semilogy(SNR_dB_list, SER_cma, 'bs-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'CMA均衡器');
semilogy(SNR_dB_list, SER_rnn_cls, 'g^-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'WD-RNN-CLS均衡器');

xlabel('信噪比 (dB)', 'FontSize', 14);
ylabel('误符号率 (SER)', 'FontSize', 14);
title('PAM4系统均衡性能比较 IM/DD 20km', 'FontSize', 16);
legend('Location', 'best', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 12);

% 设置y轴范围
ylim([1e-6, 1]);

% 保存结果
save('equalizer_comparison_results.mat', 'SNR_dB_list', 'SER_no', 'SER_cma', 'SER_rnn_cls');

% 输出结果表格
fprintf('\n=== 测试结果汇总 ===\n');
fprintf('SNR(dB)\tCMA\t\t无均衡\t\tRNN_CLS\n');
fprintf('----------------------------------------\n');
for i = 1:numSNR
    fprintf('%d\t%.2e\t%.2e\t%.2e\n', SNR_dB_list(i), SER_cma(i), SER_no(i), SER_rnn_cls(i));
end

fprintf('\n测试完成！结果已保存到 equalizer_comparison_results.mat\n');


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