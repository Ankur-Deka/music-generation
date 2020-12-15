import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from base import BaseModel
from collections import OrderedDict
try:
    from transformer import Transformer
except:
    from .transformer import Transformer

class PredictiveModel(nn.Module):
    def __init__(self, \
            num_classes, \
            num_artists, \
            embed_dim, \
            transformerfiles, \
            ):
        super().__init__()
        # Load transformers
        self.transformers = []
        self.transformerfiles = transformerfiles
        for i in range(num_artists):
            self.transformers.append(Transformer(num_classes, embed_dim))
        # Module list
        self.transformers = nn.ModuleList(self.transformers)

        # The classifier takes sequence of notes and predicts artist
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, num_artists)
        )
        # Positional embedding
        self.pos_w = nn.Parameter(torch.randn(embed_dim))
        self.pos_b = nn.Parameter(torch.randn(embed_dim))

        # Load transformers
        for i, files in enumerate(self.transformerfiles):
            with open(files, 'rb') as fi:
                state_dict = torch.load(fi)['state_dict']
                newdict = OrderedDict()
                for k, v in state_dict.items():
                    newdict[k.replace('module.', '')] = v
            self.transformers[i].load_state_dict(newdict)
        print("Loaded transformers.")


    def forward(self, X):
        # x is of size [B, S]
        x = X['seq']
        out = self.forward_classifier(X)
        # Predict next note
        # nextnote = self.generate_notes(X, 1, convert_torch=True).to(x.device).squeeze()   # [B, ]
        nextnoteprob = self.get_next_node_probability(X)
        return {
                'out': out,
                'nextnote': nextnoteprob,
        }


    def forward_classifier(self, X):
        x = X['seq']
        pos = self.pos_embedding(x) # [B, S, E]
        embed = self.embedding(x)   # [B, S, E]
        embed = embed + pos

        embedpermute = embed.permute(1, 0, 2)  # [S, B, E]
        # Get output
        out = self.encoder(embedpermute)   # [S, B, E]
        out = out.mean(0)  # [B, 2E]
        # Final output
        out = self.fc(out)  # [B, C]
        return out


    def get_next_node_probability(self, X):
        x = X['seq']
        B = x.shape[0]
        out = self.forward_classifier(X)
        outmax = torch.argmax(out, 1)
        noteprobs = []
        for tra in self.transformers:
            _nextnode = tra.forward(X)['out'][:, None]    # [B, 1, N]
            noteprobs.append(_nextnode)

        noteprobs = torch.cat(noteprobs, 1)           # [B, numartists, N]
        noteprobs = noteprobs[torch.arange(B), outmax, :]  # [B, N]
        return noteprobs


    def generate_notes(self, x, N=500, convert_torch=False):
        # Given sequence, generate notes
        seq = x['seq']  # [B, S]
        B = seq.shape[0]
        notes = []
        for i in range(N):
            out = self.forward_classifier({'seq': seq})      # Get output [B, C]
            outmax = torch.argmax(out, 1)     # argmax [B]
            # Get next sequence predictions
            candidatenotes = []
            for tra in self.transformers:
                _nextnode = tra.forward({'seq': seq})['out']  # Get output [B, N]
                candidatenotes.append(torch.argmax(_nextnode, 1)[..., None])   # appended [B, 1]
            candidatenotes = torch.cat(candidatenotes, 1)   # [B, numartists]
            candidatenotes = candidatenotes[torch.arange(B), outmax][..., None]  # [B, 1]
            notes.append(candidatenotes.data.cpu().numpy())
            # Append it to sequence
            seq = torch.cat([seq, candidatenotes], 1)            # [B, S+1]
            seq = seq[:, 1:]                             # [B, S]
        # Process them
        notes = np.concatenate(notes, -1).astype(int)    # [B, N]
        if convert_torch:
            notes = torch.LongTensor(notes)
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
    files = ["/serverdata/rohit/musicgeneration/models/AlbTransformerMidiV1/predictive/checkpoint-epoch1000.pth", \
                "/serverdata/rohit/musicgeneration/models/MendelTransformerMidiV1/predictive/checkpoint-epoch1000.pth", \
            "/serverdata/rohit/musicgeneration/models/MussTransformerMidiV1/predictive/checkpoint-epoch1000.pth"
            ]
    x = torch.randint(372, size=(4, 100))
    model = PredictiveModel(372, 3, 128, files)
    out = model({'seq': x})
    print(out['out'].shape)
    print(out['nextnote'].shape)
    print(out['nextnote'])

