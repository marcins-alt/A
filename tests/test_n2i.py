"""
Tests for the N2I utility classes and functions.

How to run
----------
From the repository root::

    pip install pytest
    pytest tests/test_n2i.py -v

These tests cover the pure-Python / NumPy / PyTorch helpers defined in N2I.
They do NOT require CUDA, real TIFF data, or the noise2inverse / msd_pytorch
packages — only numpy, torch, and matplotlib are needed.
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for CI

import numpy as np
import pytest
import torch

# ── Load N2I as a module (it has no .py extension) ──────────────────────────
_n2i_path = Path(__file__).parent.parent / "N2I"
_spec = importlib.util.spec_from_file_location("n2i_module", _n2i_path)
_n2i = importlib.util.module_from_spec(_spec)
sys.modules["n2i_module"] = _n2i
_spec.loader.exec_module(_n2i)

SkipDataset      = _n2i.SkipDataset
PairedRandomCrop = _n2i.PairedRandomCrop
PatchDataset     = _n2i.PatchDataset
select_rois      = _n2i.select_rois
Plots            = _n2i.Plots
TrainValSplit     = _n2i.TrainValSplit


# ── Helpers ──────────────────────────────────────────────────────────────────

class ListDataset(torch.utils.data.Dataset):
    """Minimal dataset backed by a Python list of (inp, tgt) pairs."""
    def __init__(self, pairs):
        self._pairs = pairs
    def __len__(self):
        return len(self._pairs)
    def __getitem__(self, idx):
        return self._pairs[idx]


def make_pairs(n, h=64, w=64):
    """Create *n* random (inp, tgt) tensor pairs of shape (1, H, W)."""
    return [(torch.rand(1, h, w), torch.rand(1, h, w)) for _ in range(n)]


# ── SkipDataset ──────────────────────────────────────────────────────────────

class TestSkipDataset:
    def test_len_reduced(self):
        ds = ListDataset(make_pairs(200))
        assert len(SkipDataset(ds, skip_start=10, skip_end=10)) == 180

    def test_first_item_matches_offset(self):
        ds = ListDataset(make_pairs(200))
        skip_ds = SkipDataset(ds, skip_start=5, skip_end=0)
        inp_skip, _ = skip_ds[0]
        inp_orig, _ = ds[5]
        assert torch.equal(inp_skip, inp_orig)

    def test_zero_skip(self):
        ds = ListDataset(make_pairs(50))
        assert len(SkipDataset(ds, skip_start=0, skip_end=0)) == 50

    def test_skip_larger_than_dataset_gives_empty(self):
        ds = ListDataset(make_pairs(10))
        assert len(SkipDataset(ds, skip_start=8, skip_end=8)) == 0

    def test_skip_end_excludes_last_items(self):
        ds = ListDataset(make_pairs(100))
        skip_ds = SkipDataset(ds, skip_start=0, skip_end=10)
        # last item of skip_ds should be ds[89]
        inp_skip, _ = skip_ds[-1]
        inp_orig, _ = ds[89]
        assert torch.equal(inp_skip, inp_orig)


# ── PairedRandomCrop ─────────────────────────────────────────────────────────

class TestPairedRandomCrop:
    def test_output_shape(self):
        crop = PairedRandomCrop(32)
        c_inp, c_tgt = crop(torch.rand(1, 64, 64), torch.rand(1, 64, 64))
        assert c_inp.shape == (1, 32, 32)
        assert c_tgt.shape == (1, 32, 32)

    def test_both_tensors_cropped_identically(self):
        torch.manual_seed(0)
        crop = PairedRandomCrop(32)
        img = torch.arange(64 * 64, dtype=torch.float32).reshape(1, 64, 64)
        c_inp, c_tgt = crop(img, img.clone())
        assert torch.equal(c_inp, c_tgt)

    def test_2d_input_promoted_to_3d(self):
        crop = PairedRandomCrop(16)
        c_inp, c_tgt = crop(torch.rand(64, 64), torch.rand(64, 64))
        assert c_inp.dim() == 3
        assert c_tgt.dim() == 3

    def test_invalid_dim_raises(self):
        crop = PairedRandomCrop(8)
        with pytest.raises(ValueError):
            crop(torch.rand(2, 3, 4, 5), torch.rand(2, 3, 4, 5))


# ── PatchDataset ─────────────────────────────────────────────────────────────

class TestPatchDataset:
    def test_len_matches_base(self):
        ds = ListDataset(make_pairs(20))
        assert len(PatchDataset(ds, patch_size=32)) == 20

    def test_output_shape(self):
        ds = ListDataset(make_pairs(5, h=128, w=128))
        inp, tgt = PatchDataset(ds, patch_size=64)[0]
        assert inp.shape == (1, 64, 64)
        assert tgt.shape == (1, 64, 64)


# ── select_rois ───────────────────────────────────────────────────────────────

class TestSelectRois:
    def test_returns_three_values(self):
        result = select_rois(np.random.rand(256, 256), roi_size=64)
        assert len(result) == 3

    def test_flat_roi_within_image_bounds(self):
        img = np.random.rand(256, 256)
        flat_rc, _, rs = select_rois(img, roi_size=64)
        r, c = flat_rc
        assert 0 <= r and r + rs <= 256
        assert 0 <= c and c + rs <= 256

    def test_edge_roi_within_image_bounds(self):
        img = np.random.rand(256, 256)
        _, edge_rc, rs = select_rois(img, roi_size=64)
        r, c = edge_rc
        assert 0 <= r and r + rs <= 256
        assert 0 <= c and c + rs <= 256

    def test_flat_roi_has_lower_gradient_than_edge_roi(self):
        # Construct image: top half is uniform (flat), bottom has a sharp edge
        img = np.zeros((128, 128), dtype=float)
        img[64:, :] = 1.0
        flat_rc, edge_rc, rs = select_rois(img, roi_size=32)
        grad = np.abs(np.gradient(img, axis=0))
        flat_score = grad[flat_rc[0]:flat_rc[0]+rs, flat_rc[1]:flat_rc[1]+rs].mean()
        edge_score = grad[edge_rc[0]:edge_rc[0]+rs, edge_rc[1]:edge_rc[1]+rs].mean()
        assert flat_score <= edge_score

    def test_3d_channel_first_input(self):
        img = np.random.rand(1, 128, 128)
        flat_rc, edge_rc, rs = select_rois(img, roi_size=32)
        assert isinstance(flat_rc, tuple) and len(flat_rc) == 2

    def test_roi_size_echoed(self):
        _, _, rs = select_rois(np.zeros((128, 128)), roi_size=48)
        assert rs == 48


# ── TrainValSplit ─────────────────────────────────────────────────────────────

class TestTrainValSplit:
    def test_sizes_sum_to_total(self):
        ds = ListDataset(make_pairs(100))
        split = TrainValSplit(ds, train_fraction=0.80)
        assert len(split.train_dataset) + len(split.val_dataset) == 100

    def test_train_fraction_respected(self):
        ds = ListDataset(make_pairs(100))
        split = TrainValSplit(ds, train_fraction=0.80)
        assert len(split.train_dataset) == 80
        assert len(split.val_dataset)   == 20

    def test_no_index_overlap(self):
        ds = ListDataset(make_pairs(50))
        split = TrainValSplit(ds, train_fraction=0.60)
        train_idx = set(split.train_dataset.indices)
        val_idx   = set(split.val_dataset.indices)
        assert train_idx.isdisjoint(val_idx)

    def test_train_before_val_by_index(self):
        ds = ListDataset(make_pairs(100))
        split = TrainValSplit(ds, train_fraction=0.70)
        assert max(split.train_dataset.indices) < min(split.val_dataset.indices)

    def test_default_fraction_is_80_percent(self):
        ds = ListDataset(make_pairs(100))
        split = TrainValSplit(ds)
        assert len(split.train_dataset) == 80


# ── Plots ─────────────────────────────────────────────────────────────────────

class TestPlots:
    def test_update_appends_to_all_lists(self, tmp_path):
        p = Plots(output_dir=tmp_path, run_label="test")
        p.update(0.5, 0.6, {"noise_std": 0.1, "noise_reduction": 2.0})
        assert p.train_losses     == [0.5]
        assert p.val_losses       == [0.6]
        assert p.noise_stds       == [0.1]
        assert p.noise_reductions == [2.0]

    def test_update_missing_metric_uses_nan(self, tmp_path):
        p = Plots(output_dir=tmp_path, run_label="test")
        p.update(0.5, 0.6, {})
        assert np.isnan(p.noise_stds[0])

    def test_save_curves_creates_png(self, tmp_path):
        p = Plots(output_dir=tmp_path, run_label="run1")
        for i in range(5):
            p.update(1.0 / (i + 1), 1.1 / (i + 1),
                     {"noise_std": 0.1, "noise_reduction": 2.0})
        p.save_curves()
        assert (tmp_path / "run1_curves.png").exists()

    def test_save_summary_keys(self, tmp_path):
        p = Plots(output_dir=tmp_path, run_label="run2")
        for i in range(3):
            p.update(float(i + 1), float(i + 1) + 0.1,
                     {"noise_std": 0.3 - i * 0.1, "noise_reduction": 2.0 + i})
        summary = p.save_summary()
        for key in ("best_epoch_by_noise", "best_noise_std",
                    "noise_reduction_at_best", "final_train_loss"):
            assert key in summary

    def test_save_summary_best_epoch(self, tmp_path):
        p = Plots(output_dir=tmp_path, run_label="run3")
        # noise_stds: [0.3, 0.1, 0.2] — best (lowest) is epoch 2
        noise_vals = [0.3, 0.1, 0.2]
        for ns in noise_vals:
            p.update(0.5, 0.6, {"noise_std": ns, "noise_reduction": 1.0})
        summary = p.save_summary()
        assert summary["best_epoch_by_noise"] == 2

    def test_save_image_grid_creates_png(self, tmp_path):
        p = Plots(output_dir=tmp_path, run_label="img_test")
        img = np.random.rand(256, 256).astype(np.float32)
        p.save_image_grid(img, img * 0.9, img * 1.1, epoch=1, roi_size=64)
        assert (tmp_path / "img_test_epoch001_grid.png").exists()
