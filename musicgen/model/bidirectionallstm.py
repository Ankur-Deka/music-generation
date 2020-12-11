import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from base import BaseModel

class BiDirectionalLSTM(nn.Module):
    def __init__(self, \
            num_classes, \
            embed_dim, \
            lstm_dim, \
            ):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_dim, \
                num_layers=4, \
                bidirectional = True,
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_dim, lstm_dim),
            nn.LeakyReLU(),
            nn.Linear(lstm_dim, lstm_dim),
            nn.LeakyReLU(),
            nn.Linear(lstm_dim, lstm_dim),
            nn.LeakyReLU(),
            nn.Linear(lstm_dim, num_classes)
        )
        self.lstm_dim = lstm_dim


    def forward(self, X):
        # x is of size [B, S]
        x = X['seq']
        B = x.shape[0]
        embed = self.embedding(x)   # [B, S, E]

        embedpermute = embed.permute(1, 0, 2)  # [S, B, E]
        # Get output
        _, (hn, cn) = self.lstm(embedpermute)   # [S, B, E]
        # hn is bidirectional
        hn = hn.view(-1, 2, B, self.lstm_dim)  # [L, 2, B, E]
        out = hn[-1].mean(0)  # [B, E]

        # Final output
        out = self.fc(out)  # [B, C]
        return {
                'out': out,
        }


    def generate_notes(self, x, N=500):
        # Given sequence, generate notes
        seq = x['seq']  # [B, S]
        notes = []
        for i in range(N):
            out = self.forward({'seq': seq})['out']      # Get output [B, C]
            outmax = torch.argmax(out, 1)[..., None]     # argmax [B, 1]
            notes.append(outmax.data.cpu().numpy())
            # Append it to sequence
            seq = torch.cat([seq, outmax], 1)            # [B, S+1]
            seq = seq[:, 1:]                             # [B, S]
        # Process them
        notes = np.concatenate(notes, -1).astype(int)    # [B, N]
        return notes



if __name__ == '__main__':
    x = torch.randint(100, size=(4, 100))
    model = BiDirectionalLSTM(100, 32, 32)
    out = model({'seq': x})
    print(out['out'].shape)
