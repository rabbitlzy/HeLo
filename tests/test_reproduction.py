import unittest
from types import SimpleNamespace
import numpy as np
import torch

from models.paper_model import PaperHeLo, sinkhorn_plan
from reproduction_metrics import distribution_metrics
from utils import compute_distance


class ReproductionTests(unittest.TestCase):
    def test_all_rows_match_independent_original_metric_formulas(self):
        p = np.array([[.1, .2, .7], [.6, .1, .3], [.2, .5, .3]])
        q = np.array([[.2, .3, .5], [.2, .5, .3], [.3, .3, .4]])
        result = distribution_metrics(p, q)
        np.testing.assert_allclose(result, np.stack([compute_distance(a, b) for a, b in zip(p, q)]), atol=1e-12)
        self.assertFalse(np.allclose(result.mean(0), result[0]))

    def test_zero_probability_convention(self):
        result = distribution_metrics([[1, 0]], [[1, 0]])
        np.testing.assert_allclose(result, [[0, 0, 0, 0, 1, 1]])
        self.assertTrue(np.isinf(distribution_metrics([[1, 0]], [[0, 1]])[0, 3]))

    def test_sinkhorn_marginals_and_gradient(self):
        torch.manual_seed(7)
        x = torch.rand(2, 4, 8, requires_grad=True)
        y = torch.rand(2, 5, 8, requires_grad=True)
        plan = sinkhorn_plan(x, y)
        self.assertTrue(torch.allclose(plan.sum(2), torch.full((2, 4), .25), atol=1e-4))
        self.assertTrue(torch.allclose(plan.sum(1), torch.full((2, 5), .2), atol=1e-4))
        (plan.transpose(1, 2) @ x).square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(y.grad).all())

    def test_literal_scalar_cosine_and_no_label_leakage(self):
        torch.manual_seed(7)
        args = SimpleNamespace(n_channels=2, feature_dim=8, emotion_classes=3, dim_eeg=5,
                               dim_gsr=4, dim_ppg=3, dim_video=6, heads=2, dropout=.5, hidden_size=4, depth=1)
        model = PaperHeLo(args).eval()
        inputs = (torch.rand(2, 2, 5), torch.rand(2, 4), torch.rand(2, 3), torch.rand(2, 6))
        labels = torch.tensor([[.1, .2, .7], [.3, .4, .3]])
        prediction, learned, target = model(*inputs, labels)
        second = model(*inputs, labels.flip(1))[0]
        self.assertTrue(torch.equal(target, torch.ones_like(target)))
        self.assertTrue(torch.allclose(learned.diagonal(dim1=1, dim2=2), torch.ones(2, 3)))
        self.assertTrue(torch.equal(prediction, second))
        loss = -prediction.log().mean() + (learned-target).square().sum((1, 2)).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(model.label_embedding.grad).all())
        self.assertGreater(model.label_embedding.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
