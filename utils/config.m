global span rolloff Fs Rs GBt nSymbols_train nSymbols_test nSymbols_all sps rrc UXR;
span  = 6;         % Raised cosine (combined Tx/Rx) delay 
rolloff  = 0.1;        % Raised cosine roll-off factor (0.1-0.5)
GBt=Rs;
Fs=120; 
Rs=30;
UXR=256;
sps=Fs/Rs;
nSymbols_train=65536;
nSymbols_test=65536;
nSymbols_all=nSymbols_train+nSymbols_test;
rrc = rcosdesign(rolloff, span, sps, 'sqrt');


