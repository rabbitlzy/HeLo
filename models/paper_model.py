"""Explicit interpretation of HeLo equations, NOT an author-verified implementation.

Underspecified choices: uniform OT marginals, epsilon=0.1, 50 Sinkhorn steps,
unscaled T.T @ physiological features, shared transformer, dropout=0.5 from
released defaults. Eq.7's scalar label cosine is deliberately literal: for
positive PANAS distributions the target correlation is all ones. Do not silently
replace it with the released code's normalized outer product.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def sinkhorn_plan(source, target, epsilon=0.1, iterations=50):
    """Differentiable log-domain balanced Sinkhorn with uniform marginals.

    Rows index source tokens; columns index target tokens. The transpose maps
    source representations to target coordinates. Regularization is not
    specified in the paper and must be recorded in every experiment.
    """
    cost = (1 - torch.bmm(F.normalize(source, dim=-1),
                         F.normalize(target, dim=-1).transpose(1, 2))).clamp(0, 2)
    log_kernel = -cost / epsilon
    n, m = source.shape[1], target.shape[1]
    u = source.new_zeros(source.shape[0], n)
    v = target.new_zeros(target.shape[0], m)
    for _ in range(iterations):
        u = -math.log(n) - torch.logsumexp(log_kernel + v[:, None, :], dim=2)
        v = -math.log(m) - torch.logsumexp(log_kernel + u[:, :, None], dim=1)
    return (log_kernel + u[:, :, None] + v[:, None, :]).exp()


class PhysiologicalAttention(nn.Module):
    """Equations 1-3: EEG query, peripheral keys/values, output projection + LN(y)."""
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        b, n, d = x.shape
        q = self.q(x).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(y).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(y).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        weights = (q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)).softmax(-1)
        value = (self.dropout(weights) @ v).transpose(1, 2).reshape(b, n, d)
        return self.output(value) + self.norm(y)


class PaperHeLo(nn.Module):
    """Paper-equation interpretation; see protocol report for remaining ambiguity."""
    def __init__(self, args):
        super().__init__()
        c, d, labels = args.n_channels, args.feature_dim, args.emotion_classes
        self.channels, self.dim, self.labels = c, d, labels
        self.epsilon = getattr(args, "sinkhorn_epsilon", 0.1)
        self.iterations = getattr(args, "sinkhorn_iterations", 50)
        self.eeg = nn.Linear(args.dim_eeg, d)
        self.gsr = nn.Linear(args.dim_gsr, c * d)
        self.ppg = nn.Linear(args.dim_ppg, c * d)
        self.video = nn.Linear(args.dim_video, 2 * c * d)
        self.cross_gsr = PhysiologicalAttention(d, args.heads, args.dropout)
        self.cross_ppg = PhysiologicalAttention(d, args.heads, args.dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, args.heads, args.hidden_size, args.dropout,
                                       activation="gelu", batch_first=True), args.depth)
        self.fused_projection = nn.Linear(4 * c, labels)
        self.label_embedding = nn.Parameter(torch.rand(1, labels, d))
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.Classifier = nn.Sequential(nn.Linear(labels * d, d), nn.ReLU(),
                                        nn.Linear(d, d), nn.ReLU(), nn.Linear(d, labels))

    def forward(self, eeg, gsr, ppg, video, labels):
        b, c, d = len(eeg), self.channels, self.dim
        e = self.eeg(eeg.reshape(b, c, -1))
        g = self.gsr(gsr.reshape(b, -1)).reshape(b, c, d)
        p = self.ppg(ppg.reshape(b, -1)).reshape(b, c, d)
        physiological = torch.cat([self.cross_gsr(e, g), self.cross_ppg(e, p)], dim=1)
        behavioral = self.video(video).reshape(b, 2 * c, d)
        plan = sinkhorn_plan(physiological, behavioral, self.epsilon, self.iterations)
        aligned = plan.transpose(1, 2) @ physiological
        fused = torch.cat([self.encoder(aligned), self.encoder(behavioral)], dim=1)
        fused = self.fused_projection(fused.transpose(1, 2)).transpose(1, 2)
        embedding = self.label_embedding.expand(b, -1, -1)
        normalized = F.normalize(embedding, dim=-1)
        learned = normalized @ normalized.transpose(1, 2)
        label_vectors = F.normalize(labels.unsqueeze(-1), dim=-1)
        target = label_vectors @ label_vectors.transpose(1, 2)
        weights = ((self.q(embedding) @ self.k(fused).transpose(1, 2) + learned)
                   / math.sqrt(d)).softmax(-1)
        output = weights @ self.v(fused)
        return self.Classifier(output.flatten(1)).softmax(-1), learned, target
