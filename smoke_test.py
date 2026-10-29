"""Check the unmodified HeLo model with synthetic data; no DMER data required."""

import torch
import torch.nn.functional as F

from models.model import HeLo
from params import Config


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    require(torch.cuda.is_available(), "CUDA is unavailable in this Python environment.")
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    # Keep upstream model dimensions; reduce only the synthetic batch size.
    config = Config()
    config.initialize()
    args = config.parser.parse_args([])
    batch_size = 2
    model = HeLo(args).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    eeg = torch.rand(batch_size, args.n_channels, args.dim_eeg, device=device)
    gsr = torch.rand(batch_size, 1, args.dim_gsr, device=device)
    ppg = torch.rand(batch_size, 1, args.dim_ppg, device=device)
    video = torch.rand(batch_size, args.dim_video, device=device)
    labels = torch.rand(batch_size, args.emotion_classes, device=device) + 0.1
    labels = labels / labels.sum(dim=1, keepdim=True)

    print("PyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(device))
    print("Synthetic batch size:", batch_size)
    print("Input shapes:", tuple(eeg.shape), tuple(gsr.shape), tuple(ppg.shape), tuple(video.shape))
    print("Running forward pass...", flush=True)
    predict, learned_corr, gt_corr = model(eeg, gsr, ppg, video, labels)

    require(tuple(predict.shape) == (batch_size, args.emotion_classes), "Unexpected output shape.")
    for name, value in (("prediction", predict), ("learned correlation", learned_corr), ("target correlation", gt_corr)):
        require(torch.isfinite(value).all().item(), name + " contains NaN or Inf.")
    require((predict > 0).all().item(), "Prediction contains zero or negative values before log().")
    require(torch.allclose(predict.sum(dim=1), torch.ones(batch_size, device=device), atol=1e-5), "Prediction rows do not sum to 1.")
    require(tuple(learned_corr.shape) == (batch_size, args.emotion_classes, args.emotion_classes), "Unexpected learned correlation shape.")
    require(gt_corr.shape == learned_corr.shape, "Correlation shapes differ.")

    # Match upstream loss reductions: KL sum and correlation MSE mean.
    kl_loss = F.kl_div(predict.log(), labels, reduction="sum")
    cc_loss = F.mse_loss(learned_corr, gt_corr, reduction="mean")
    loss = kl_loss + cc_loss
    require(torch.isfinite(loss).item(), "Loss is not finite.")
    print("Output shape:", tuple(predict.shape))
    print("Row sums:", predict.detach().sum(dim=1).cpu().tolist())
    print("Correlation shape:", tuple(learned_corr.shape))
    print("KL loss: %.6f | CC loss: %.6f | Total loss: %.6f" % (kl_loss.item(), cc_loss.item(), loss.item()))

    before = model.Classifier[-1].weight.detach().clone()
    optimizer.zero_grad()
    print("Running backward pass...", flush=True)
    loss.backward()
    grads = [(name, param.grad) for name, param in model.named_parameters() if param.grad is not None]
    require(bool(grads), "No registered parameter received a gradient.")
    for name, grad in grads:
        require(torch.isfinite(grad).all().item(), "Non-finite gradient: " + name)
    require(any(torch.count_nonzero(grad).item() > 0 for _, grad in grads), "All gradients are zero.")
    optimizer.step()
    require(not torch.equal(before, model.Classifier[-1].weight.detach()), "Classifier weights did not update.")
    for name, param in model.named_parameters():
        require(torch.isfinite(param).all().item(), "Non-finite parameter after update: " + name)
    torch.cuda.synchronize(device)
    print("Backward: OK")
    print("Optimizer update: OK")
    print("Peak PyTorch allocated memory: %.1f MiB" % (torch.cuda.max_memory_allocated(device) / 1024**2))

    registered = any("label_embedding" in name for name, _ in model.named_parameters())
    print("Label embedding registered:", registered)
    if not registered:
        print("NOTE: Upstream creates label_embedding inside forward(); it is not a persistent model parameter.")
    print("SMOKE TEST PASSED (execution only; not evidence of paper reproduction).")


if __name__ == "__main__":
    main()
