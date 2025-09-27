%% ===================================================
% 加载VPI Photonics生成的信道数据
rx=-load('vpi_data.txt');
rx=2*(rx-mean(rx))/mean(abs(rx));
%% ===================================================
config;
% 数据集划分（训练:验证:测试 = 65536:65536:131072）
nSymbols_val = nSymbols_train;  % 验证集大小
rx_train = rx(1:nSymbols_train*sps);
rx_val   = rx(nSymbols_train*sps+1:(nSymbols_train+nSymbols_val)*sps);
rx_test  = rx((nSymbols_train+nSymbols_val)*sps+1:end);

%% ----------------- 匹配滤波+下采样 ----------------- 
rx_matched_train = conv(rx_train, rrc,'same');
rx_matched_val   = conv(rx_val,   rrc,'same');
rx_matched_test  = conv(rx_test,  rrc,'same');
rx_sym_train = resample(rx_matched_train,Rs,Fs)';
rx_sym_val   = resample(rx_matched_val,  Rs,Fs)';
rx_sym_test  = resample(rx_matched_test, Rs,Fs)';

% 加载对应的符号数据
symb_train=load('symb_train.txt');
symb_val  =load('symb_val.txt');
symb_test =load('symb_test.txt');

% 保存处理后的接收信号数据供Python使用
save('Transformer_EQ/rx_sym_train.txt', 'rx_sym_train', '-ascii');
save('Transformer_EQ/rx_sym_val.txt',   'rx_sym_val',   '-ascii');
save('Transformer_EQ/rx_sym_test.txt',  'rx_sym_test',  '-ascii');