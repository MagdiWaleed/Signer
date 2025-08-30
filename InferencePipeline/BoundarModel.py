import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x


class BoundaryDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_projection = nn.Sequential(
            nn.LayerNorm(33*2),
            nn.Linear(33*2,128),
            nn.Linear(128,256)
        )
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=256,
                nhead=4,
                dim_feedforward=512,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)


        self.positionalEmbedding = PositionalEncoding(256)
        self.output_layer = nn.Sequential(
            nn.Linear(256,128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128,2)
        )

    def forward(self,landmarks):
        batch_size,seq_len, _,_ = landmarks.size()
        landmarks = landmarks.reshape(batch_size, seq_len,-1)
        landmarks = self.src_projection(landmarks)

        landmarks = self.positionalEmbedding(landmarks)

        transformer_out = self.transformer(landmarks)
        logits = self.output_layer(transformer_out)
        return logits
