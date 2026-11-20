"""Small, causal analogues of the five published architecture families."""
import torch
from torch import nn
from torch.nn import functional as F
from .data import N_PHASES

ARCHITECTURES = ('lstm', 'lstm_attention', 'lstm_fcn', 'inceptiontime', 'transformer')


class CausalConv(nn.Conv1d):
    def forward(self, x):
        return super().forward(F.pad(x, (self.kernel_size[0]-1, 0)))


class ETCModel(nn.Module):
    def __init__(self, architecture, width=32):
        super().__init__()
        if architecture not in ARCHITECTURES or width < 4 or width % 4:
            raise ValueError('unknown architecture or width not a positive multiple of four')
        self.architecture = architecture
        self.embed = nn.Embedding(N_PHASES+1, width, padding_idx=N_PHASES)
        self.input = nn.Linear(width+1, width)
        if architecture.startswith('lstm'):
            self.lstm = nn.LSTM(width, width, batch_first=True)
        if architecture == 'lstm_attention':
            self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        if architecture == 'lstm_fcn':
            self.conv = nn.Sequential(CausalConv(width,width,5), nn.ReLU(),
                                      CausalConv(width,width,3), nn.ReLU())
        if architecture == 'inceptiontime':
            self.branches = nn.ModuleList(CausalConv(width,width//4,k) for k in (1,3,5,9))
            self.mix = nn.Linear(width,width)
        if architecture == 'transformer':
            layer = nn.TransformerEncoderLayer(width,4,2*width,dropout=0.,batch_first=True)
            self.transformer = nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.head = nn.Linear(2*width if architecture=='lstm_fcn' else width,1)

    def forward(self, phases):
        # At time t only phase history through t and elapsed sample index are available.
        ids = phases.masked_fill(phases == -1, N_PHASES)
        e = self.embed(ids)
        t = torch.arange(phases.shape[1],device=phases.device,dtype=e.dtype)
        elapsed = (t / 100.).view(1,-1,1).expand(phases.shape[0],-1,-1)
        x = self.input(torch.cat([e,elapsed],dim=-1))
        a = self.architecture
        if a.startswith('lstm'):
            h,_ = self.lstm(x)
            if a == 'lstm_attention':
                mask = torch.ones(x.shape[1],x.shape[1],dtype=torch.bool,device=x.device).triu(1)
                q,_ = self.attention(h,h,h,attn_mask=mask,need_weights=False)
                h = h+q
            elif a == 'lstm_fcn':
                h = torch.cat([h,self.conv(x.transpose(1,2)).transpose(1,2)],dim=-1)
        elif a == 'inceptiontime':
            z = x.transpose(1,2)
            h = x + self.mix(torch.cat([F.relu(c(z)) for c in self.branches],dim=1).transpose(1,2))
        else:
            mask = torch.ones(x.shape[1],x.shape[1],dtype=torch.bool,device=x.device).triu(1)
            h = self.transformer(x,mask=mask)
        return F.softplus(self.head(h).squeeze(-1))


def masked_mae(pred, target, scale=600.):
    if pred.shape != target.shape or scale <= 0:
        raise ValueError('aligned prediction and target, and positive scale required')
    valid = target >= 0
    if not valid.any():
        raise ValueError('empty target')
    return (pred[valid]-target[valid]/scale).abs().mean()
