"""Inspect current label learning and eval randomness without editing the model.

All inputs are synthetic. The cosine comparison is a mathematical reference,
not a proposed replacement for the paper's ambiguous supervision definition.
"""

import torch
import torch.nn.functional as F

from models.model import HeLo
from params import Config


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Activate the CUDA-enabled helo environment first.")
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    config = Config()
    config.initialize()
    args = config.parser.parse_args([])
    model = HeLo(args).to(device)
    batch_size = 2

    print("[1] Persistent label embedding")
    names = [name for name, _ in model.named_parameters() if "label_embedding" in name]
    print("Registered label embedding parameters:", names)

    print("\n[2] Correlation normalization (synthetic input)")
    embedding = torch.rand(batch_size, args.emotion_classes, args.feature_dim, device=device)
    labels = torch.rand(batch_size, args.emotion_classes, 1, device=device) + 0.1
    labels = labels / labels.sum(dim=1, keepdim=True)
    with torch.no_grad():
        actual, target = model.label_correlation(embedding, labels)
        unit_rows = F.normalize(embedding, p=2, dim=-1)
        cosine = unit_rows @ unit_rows.transpose(1, 2)
    for name, matrix in (("Current learned matrix", actual), ("Target matrix", target), ("Per-label cosine reference", cosine)):
        diagonal = matrix.diagonal(dim1=-2, dim2=-1)
        print("%s: diagonal mean=%.6f, range=[%.6f, %.6f]" % (name, diagonal.mean().item(), matrix.min().item(), matrix.max().item()))
    print("Current CC MSE: %.6f" % F.mse_loss(actual, target).item())
    print("Cosine-reference CC MSE: %.6f" % F.mse_loss(cosine, target).item())
    print("NOTE: A smaller MSE here does not validate a replacement definition.")
    print("NOTE: The target diagonal is not generally 1, unlike a cosine matrix.")

    print("\n[3] Attention randomness in eval mode")
    model.eval()
    q = torch.rand(batch_size, args.emotion_classes, args.feature_dim, device=device)
    k = torch.rand_like(q)
    v = torch.rand_like(q)
    bias = torch.zeros_like(actual)
    with torch.no_grad():
        a = model.lcd_cross_attention.attention(q, k, v, bias, scale=args.feature_dim ** -0.5)
        b = model.lcd_cross_attention.attention(q, k, v, bias, scale=args.feature_dim ** -0.5)
    print("Attention repeated-output max difference: %.8f" % (a - b).abs().max().item())

    print("\n[4] Full-model repeated prediction (same input, no optimizer step)")
    inputs = (
        torch.rand(batch_size, args.n_channels, args.dim_eeg, device=device),
        torch.rand(batch_size, 1, args.dim_gsr, device=device),
        torch.rand(batch_size, 1, args.dim_ppg, device=device),
        torch.rand(batch_size, args.dim_video, device=device),
        labels.squeeze(-1),
    )
    captured = []

    def capture_embedding(module, hook_inputs):
        captured.append(hook_inputs[0].detach().clone())

    hook = model.lcd_cross_attention.register_forward_pre_hook(capture_embedding)
    try:
        with torch.no_grad():
            pred_a = model(*inputs)[0]
            pred_b = model(*inputs)[0]
    finally:
        hook.remove()
    if len(captured) != 2:
        raise RuntimeError("Expected to capture one label embedding per forward call.")
    for value in (a, b, pred_a, pred_b):
        if not torch.isfinite(value).all().item():
            raise RuntimeError("Diagnostic output contains NaN or Inf.")
    print("Label embeddings identical:", torch.equal(captured[0], captured[1]))
    print("Prediction max difference: %.8f" % (pred_a - pred_b).abs().max().item())
    print("Predictions close (atol=1e-6):", torch.allclose(pred_a, pred_b, atol=1e-6, rtol=0))
    print("\nDIAGNOSTICS FINISHED; this script did not modify model source files.")


if __name__ == "__main__":
    main()
