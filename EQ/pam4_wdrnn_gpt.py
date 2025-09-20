"""
WD-RNN PAM4 分类版复现
改进点:
  1) 输出层: 4 分类 (softmax)，CrossEntropyLoss
  2) 推理: 使用 softmax 概率作为可靠度 γ
  3) feedback: WD 融合 argmax 电平 与 softmax 均值
"""

import numpy as np
from scipy import signal
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# 设备选择
# -------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

# -------------------------
# 系统参数
# -------------------------
SYMBOL_RATE = 50e9
SPS = 2
FS = SYMBOL_RATE * SPS
ROLL_OFF = 0.01

N0 = 61
N1 = 20
K_DELAY = 6
ALPHA = 5
BETA = 0.14

TRAIN_SYMS = 20000
TEST_SYMS = 40000
EPOCHS = 20
BATCH_SIZE = 512
LR = 1e-3

# 符号映射
SYMBOLS = np.array([-3., -1., 1., 3.])
SYMBOL2IDX = {-3.:0, -1.:1, 1.:2, 3.:3}
IDX2SYMBOL = {v:k for k,v in SYMBOL2IDX.items()}

BITS2SYMBOL = {(0,0):-3., (0,1):-1., (1,1):1., (1,0):3.}
SYMBOL2BITS = {v:k for k,v in BITS2SYMBOL.items()}

def bits_to_symbols(bits):
    return np.array([BITS2SYMBOL[tuple(b)] for b in bits], dtype=np.float32)

def symbols_to_bits(symbols):
    out = []
    for s in symbols:
        out.append(SYMBOL2BITS[s])
    return np.array(out, dtype=np.int8)

# -------------------------
# RRC 滤波器
# -------------------------
def rrc_filter(span_symbols, sps, beta):
    N = span_symbols * sps
    t = np.arange(-N/2, N/2 + 1) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - beta + 4*beta/np.pi
        elif beta != 0 and np.isclose(abs(ti), 1/(4*beta)):
            h[i] = (beta/np.sqrt(2)) * ((1+2/np.pi)*np.sin(np.pi/(4*beta)) + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
        else:
            numerator = np.sin(np.pi*ti*(1-beta)) + 4*beta*ti*np.cos(np.pi*ti*(1+beta))
            denominator = np.pi*ti*(1-(4*beta*ti)**2)
            h[i] = numerator/denominator
    return h/np.sqrt(np.sum(h**2))

# -------------------------
# 发射机
# -------------------------
def generate_tx_waveform(nsym, seed=0):
    rng = np.random.RandomState(seed)
    bits = rng.randint(0,2, size=(nsym,2)).astype(np.int8)
    symbols = bits_to_symbols(bits)
    up = np.zeros(len(symbols)*SPS)
    up[::SPS] = symbols
    rrc = rrc_filter(10, SPS, ROLL_OFF)
    tx = np.convolve(up, rrc, mode='same')
    return bits, symbols, tx

# -------------------------
# 信道
# -------------------------
def dml_channel(sig, fs=FS, f_10db=20.2e9, nonlin_coeff=0.02):
    n = 4
    fc = f_10db/(9.0**(1.0/(2*n)))
    wn = fc/(fs/2)
    b,a = signal.butter(n, wn)
    y = signal.lfilter(b,a,sig)
    if nonlin_coeff>0:
        y = y + nonlin_coeff*(y**3)
    return y

def awgn(sig, snr_db):
    p = np.mean(sig**2)
    n = np.sqrt(p/10**(snr_db/10))*np.random.randn(*sig.shape)
    return sig+n

def rx_process(rx):
    rrc = rrc_filter(10, SPS, ROLL_OFF)
    filt = np.convolve(rx, rrc, mode='same')
    return filt[::SPS]

# -------------------------
# WD 函数
# -------------------------
def S_compressed(x, alpha=ALPHA, beta=BETA):
    z = alpha*(x/beta - 1)
    ex = np.exp(-z)
    return 0.5*((1-ex)/(1+ex)+1)

def wd_feedback(probs):
    # probs: softmax 输出 (4,)
    y_hat_idx = np.argmax(probs)
    y_hat = IDX2SYMBOL[y_hat_idx]
    gamma = np.max(probs)   # 用最大 softmax 作为可靠度
    Sg = S_compressed(gamma)
    y_soft = np.dot(probs, SYMBOLS)  # soft 输出的均值
    y_tilde = Sg*y_hat + (1-Sg)*y_soft
    return y_tilde, y_hat

# -------------------------
# WD-RNN 模型 (分类版)
# -------------------------
class WD_RNN(nn.Module):
    def __init__(self, n0=N0, n1=N1, k=K_DELAY):
        super().__init__()
        self.fc1 = nn.Linear(n0+k, n1)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(n1,4)   # 分类输出
    def forward(self,x):
        h = self.act(self.fc1(x))
        return self.fc2(h)           # logits

# -------------------------
# 构建数据集
# -------------------------
def build_dataset(rx_down, labels, n0=N0, k=K_DELAY):
    rx_norm = (rx_down-np.mean(rx_down))/np.std(rx_down)
    lab_idx = np.array([SYMBOL2IDX[s] for s in labels],dtype=np.int64)
    start = n0-1+k
    X,y = [],[]
    for i in range(start,len(rx_down)):
        xwin = rx_norm[i-(n0-1):i+1]
        yd = lab_idx[i-1:i-1-k:-1]
        if len(yd)<k:
            yd = np.pad(yd,(0,k-len(yd)))
        # feedback 在训练时用真实标签 (one-hot)
        yd_embed = [IDX2SYMBOL[j] for j in yd]  # 用电平值代替
        X.append(np.concatenate([xwin,yd_embed]))
        y.append(lab_idx[i])
    return np.array(X,np.float32), np.array(y,np.int64)

# -------------------------
# 训练
# -------------------------
def train_model(model,X,y,epochs=EPOCHS,lr=LR):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
    for ep in range(epochs):
        total=0
        for xb,yb in loader:
            xb,yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits,yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()*xb.size(0)
        print(f"Epoch {ep+1}/{epochs}, Loss {total/len(dataset):.4e}")
    return model

# -------------------------
# 推理 (WD)
# -------------------------
def infer(model,rx_down,labels,n0=N0,k=K_DELAY):
    rx_norm = (rx_down-np.mean(rx_down))/np.std(rx_down)
    N = len(rx_down)
    pred_syms = np.zeros(N)
    # warm-up: 用真实标签
    y_feedback = list(labels[:k])
    start=n0-1
    for i in range(start,N):
        xwin = rx_norm[i-(n0-1):i+1]
        xfull = np.concatenate([xwin,y_feedback[:k]])
        with torch.no_grad():
            logits = model(torch.from_numpy(xfull).float().unsqueeze(0).to(device))
            probs = torch.softmax(logits,dim=1).cpu().numpy().squeeze()
        y_tilde,y_hat = wd_feedback(probs)
        pred_syms[i]=y_hat
        y_feedback=[y_tilde]+y_feedback[:-1]
    return pred_syms

# -------------------------
# 实验流程
# -------------------------
def run():
    bits_tr,syms_tr,tx_tr = generate_tx_waveform(TRAIN_SYMS,seed=1)
    ch_tr = dml_channel(tx_tr)
    rx_tr = awgn(ch_tr,20)
    rx_down_tr = rx_process(rx_tr)
    Xtr,ytr = build_dataset(rx_down_tr,syms_tr)
    model = WD_RNN()
    model = train_model(model,Xtr,ytr)
    # test
    bits_te,syms_te,tx_te = generate_tx_waveform(TEST_SYMS,seed=2)
    ch_te = dml_channel(tx_te)
    rx_te = awgn(ch_te,15)
    rx_down_te = rx_process(rx_te)
    y_pred = infer(model,rx_down_te,syms_te)
    # BER
    bits_est = symbols_to_bits(y_pred[N0-1:])
    bits_ref = bits_te[N0-1:]
    ber = np.mean(bits_est!=bits_ref)
    print("Test BER:",ber)

if __name__=="__main__":
    run()
