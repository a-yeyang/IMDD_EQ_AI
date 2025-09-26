# main.py (Corrected)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from pam4_system import PAM4System
from channel import OpticalChannel
from model import TransformerEqualizer
from utils import calculate_ber


# MODIFIED FUNCTION
def generate_data(system, channel, num_symbols, seq_len):
    all_rx_symbols = []
    all_tx_indices = []

    num_sequences = num_symbols // seq_len
    for _ in range(num_sequences):
        tx_indices = np.random.randint(0, 4, seq_len)
        tx_signal, _ = system.transmit(tx_indices)
        rx_signal = channel.propagate(tx_signal)
        # Pass seq_len to the receive function
        rx_symbols = system.receive(rx_signal, seq_len)

        # Add a check to ensure dimensions match
        if len(rx_symbols) == seq_len:
            all_rx_symbols.append(rx_symbols)
            all_tx_indices.append(tx_indices)

    return torch.FloatTensor(np.array(all_rx_symbols)), torch.LongTensor(np.array(all_tx_indices))


def main():
    # --- Parameters ---
    SEQ_LEN = 128
    D_MODEL = 64
    N_LAYERS = 4
    N_HEAD = 8
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    SNR_DB = 18

    # --- System Setup ---
    # ------------------- START OF MODIFICATION -------------------

    # Auto-detect device for cross-platform compatibility
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Backend: NVIDIA CUDA")
    # Check for Apple Silicon (M1/M2/M3) MPS backend
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Backend: Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        print("Backend: CPU")

    # -------------------- END OF MODIFICATION --------------------

    pam4_system = PAM4System(sps=8)
    channel = OpticalChannel(snr_db=SNR_DB)
    model = TransformerEqualizer(
        d_model=D_MODEL, n_layers=N_LAYERS, n_head=N_HEAD, n_position=SEQ_LEN
    ).to(device) # The .to(device) call now sends the model to the correct hardware

########################################################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Data Generation ---
    print("Generating training data...")
    train_rx, train_tx = generate_data(pam4_system, channel, num_symbols=500000, seq_len=SEQ_LEN)
    train_dataset = TensorDataset(train_rx, train_tx)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Generating validation data...")
    val_rx, val_tx = generate_data(pam4_system, channel, num_symbols=50000, seq_len=SEQ_LEN)
    val_dataset = TensorDataset(val_rx, val_tx)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for rx_batch, tx_batch in train_loader:
            rx_batch, tx_batch = rx_batch.to(device), tx_batch.to(device)

            optimizer.zero_grad()

            logits = model(rx_batch)
            loss = criterion(logits.view(-1, 4), tx_batch.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for rx_batch, tx_batch in val_loader:
                rx_batch = rx_batch.to(device)
                logits = model(rx_batch)
                preds = torch.argmax(logits, dim=-1)
                all_preds.append(preds.cpu().numpy())
                all_true.append(tx_batch.numpy())

        all_preds = np.concatenate(all_preds).flatten()
        all_true = np.concatenate(all_true).flatten()
        ber = calculate_ber(all_preds, all_true)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, BER: {ber:.6f}")


if __name__ == '__main__':
    main()