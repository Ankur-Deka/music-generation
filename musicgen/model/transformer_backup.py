import torch.nn as nn
import torch
import torch.nn.functional as F
from base import BaseModel

class Transformer(nn.Module):
    def __init__(self, \
            num_classes, \
            embed_dim, \
            ):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(embed_dim*2, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=8)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, num_classes)
        )
        # Positional embedding
        self.pos_w = nn.Parameter(torch.randn(embed_dim))
        self.pos_b = nn.Parameter(torch.randn(embed_dim))

    def forward(self, X):
        # x is of size [B, S]
        x = X['seq']
        pos = self.pos_embedding(x) # [B, S, E]
        embed = self.embedding(x)   # [B, S, E]
        embed = torch.cat([embed, pos], -1)  # [B, S, 2E]

        embedpermute = embed.permute(1, 0, 2)  # [S, B, 2E]
        # Get output
        out = self.encoder(embedpermute)   # [S, B, 2E]
        out = out.mean(0)  # [B, 2E]
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
        return notes


    def pos_embedding(self, inp):
        # inp is of size [B, S]
        # xfill = x[..., None]            # [B, S, 1]
        B, S = inp.shape
        x = torch.arange(S)[None].to(inp.device)   # [1, S]
        x = torch.cat([x]*B, 0)[..., None]             # [B, S, 1]

        w = self.pos_w[None, None, :]   # [1, 1, E]
        b = self.pos_b[None, None, :]   # [1, 1, E]
        # get a embedding
        embed = x*w + b
        e1 = embed[:, :, :1]
        e2 = torch.sin(embed[:, :, 1:])
        finalembed = torch.cat([e1, e2], -1)  # [B, S, E]
        return finalembed


if __name__ == '__main__':
    x = torch.randint(100, size=(4, 100))
    model = Transformer(100, 32)
    out = model({'seq': x})
    print(out['out'].shape)
