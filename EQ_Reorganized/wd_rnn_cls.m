 %% ----------------- 构建训练输入 -----------------
function modelFile=wd_rnn_cls(rx_sym_train,symb_train)
config;
rngSeed = 12345;  
n0 = 61;                 
n1 = 20;                 
k_delay = 6;             
alpha_wd = 5; beta_wd = 0.14;    
dropout_rate = 0.01;             

maxEpochs = 30;         
miniBatch = 1024;       
learnRate = 1e-3;       

lambda_ce  = 0.5;       
lambda_mix = 0.5;       
          
pam4_levels = [-3, -1, 1, 3];

%% ----------------- GPU 检测 -----------------
useGPU = true;    
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
    fprintf('GPU disabled by user. Running on CPU.\n');
end

%% ----------------- GPU 检测 -----------------
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
    fprintf('GPU disabled by user. Running on CPU.\n');
end
padL = floor(n0/2); padR = n0 - padL - 1;
rx_train_pad = [zeros(padL,1); rx_sym_train; zeros(padR+k_delay,1)];
Ntrain = nSymbols_train;
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

level2idx = containers.Map({-3,-1,1,3}, {1,2,3,4});
yclassTrain = arrayfun(@(v) level2idx(v), symb_train);

%% ----------------- 网络参数初始化 -----------------
rng(rngSeed);  
W1  = 0.1*randn(n1, inputDim, 'single');   b1  = zeros(n1,1,'single');
W2y = 0.1*randn(1, n1, 'single');          b2y = zeros(1,1,'single');
W2c = 0.05*randn(4, n1,'single');          b2c = zeros(4,1,'single');

if useGPU
    Xtrain = gpuArray(Xtrain); Ttrain = gpuArray(Ttrain);
    W1 = gpuArray(W1); b1 = gpuArray(b1);
    W2y = gpuArray(W2y); b2y = gpuArray(b2y);
    W2c = gpuArray(W2c); b2c = gpuArray(b2c);
end

%% ----------------- Adam 训练 -----------------
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

        % forward
        Z1 = W1 * Xbatch + b1;          
        H1 = tanh(Z1);                  
        if dropout_rate > 0
            mask = (rand(size(H1)) > dropout_rate);
            if useGPU, mask = gpuArray(mask); end
            H1 = H1 .* mask / (1-dropout_rate);
        end
        Yreg   = W2y * H1 + b2y;        
        Logits = W2c * H1 + b2c;        

        Logits = Logits - max(Logits,[],1);
        ExpL   = exp(Logits);
        Pcls   = ExpL ./ sum(ExpL,1);   

        err    = Yreg - Tbatch;                         
        mseLoss = mean(err.^2, 'all');

        cols = 1:size(Xbatch,2);
        labels = yclassTrain(idx(i1:i2));    
        if useGPU, labels = gpuArray(labels); end
        I = sub2ind(size(Pcls), double(labels(:))', cols);
        ceLoss = -mean( log( max(Pcls(I), realmin('single')) ) );

        loss = mseLoss + lambda_ce * ceLoss;

        batchSizeCurr = size(Xbatch,2);

        dYreg = (2/batchSizeCurr) * err;        
        Yoh   = zeros(4, batchSizeCurr, 'like', Pcls); Yoh(I) = 1;
        dLogits = lambda_ce * (Pcls - Yoh) / batchSizeCurr;  

        dH1 = (W2y' * dYreg) + (W2c' * dLogits);            

        if dropout_rate > 0
            dH1 = dH1 .* mask / (1-dropout_rate);
        end

        dZ1 = dH1 .* (1 - tanh(Z1).^2);

        dW2y = dYreg * H1';               db2y = sum(dYreg,2);
        dW2c = dLogits * H1';             db2c = sum(dLogits,2);
        dW1  = dZ1 * Xbatch';             db1  = sum(dZ1,2);

        % Adam updates (略，和你原来一样)
        % -----------------
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

        mW2y = beta1*mW2y + (1-beta1)*dW2y;
        vW2y = beta2*vW2y + (1-beta2)*(dW2y.^2);
        mhatW2y = mW2y / (1 - beta1^iter);
        vhatW2y = vW2y / (1 - beta2^iter);
        W2y = W2y - learnRate * mhatW2y ./ (sqrt(vhatW2y) + epsAdam);

        mb2y = beta1*mb2y + (1-beta1)*db2y;
        vb2y = beta2*vb2y + (1-beta2)*(db2y.^2);
        mb2yhat = mb2y / (1 - beta1^iter);
        vb2yhat = vb2y / (1 - beta2^iter);
        b2y = b2y - learnRate * mb2yhat ./ (sqrt(vb2yhat) + epsAdam);

        mW2c = beta1*mW2c + (1-beta1)*dW2c;
        vW2c = beta2*vW2c + (1-beta2)*(dW2c.^2);
        mhatW2c = mW2c / (1 - beta1^iter);
        vhatW2c = vW2c / (1 - beta2^iter);
        W2c = W2c - learnRate * mhatW2c ./ (sqrt(vhatW2c) + epsAdam);

        mb2c = beta1*mb2c + (1-beta1)*db2c;
        vb2c = beta2*vb2c + (1-beta2)*(db2c.^2);
        mb2chat = mb2c / (1 - beta1^iter);
        vb2chat = vb2c / (1 - beta2^iter);
        b2c = b2c - learnRate * mb2chat ./ (sqrt(vb2chat) + epsAdam);
    end

    if useGPU, lossVal = gather(double(loss)); else, lossVal = double(loss); end
    fprintf('Epoch %d/%d - Loss: %.5e \n', epoch, maxEpochs, lossVal);
end

fprintf('Training finished. \n');

%% ----------------- 保存模型 -----------------
modelFile = 'wd_rnn_cls_model.mat';
if useGPU
    W1 = gather(W1); b1 = gather(b1);
    W2y = gather(W2y); b2y = gather(b2y);
    W2c = gather(W2c); b2c = gather(b2c);
end
save(modelFile, 'W1','b1','W2y','b2y','W2c','b2c', ...
    'n0','n1','k_delay','pam4_levels','alpha_wd','beta_wd','lambda_mix','-v7.3');
fprintf('Model saved to %s\n', modelFile);

end