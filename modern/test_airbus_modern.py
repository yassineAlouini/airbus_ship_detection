import numpy as np
import pytest

from airbus_modern import (
    border_from_labels,
    gate_sources,
    image_f2,
    instances_from_probs,
    labels_from_rles,
    rle_decode,
    rle_encode,
    rles_from_labels,
    snap_to_rectangles,
)


def _box(labels, k, y0, y1, x0, x1):
    labels[y0:y1, x0:x1] = k
    return labels


def test_rle_roundtrip_matches_competition_convention():
    mask = np.zeros((768, 768), dtype=np.uint8)
    mask[10:20, 5] = 1  # column-major: one run starting at 5 * 768 + 10 + 1
    assert rle_encode(mask) == f"{5 * 768 + 11} 10"
    assert (rle_decode(rle_encode(mask)) == mask).all()


def test_f2_empty_image_rules():
    empty = np.zeros((768, 768), dtype=np.int32)
    ship = _box(empty.copy(), 1, 0, 10, 0, 10)
    assert image_f2(empty, empty) == 1.0
    assert image_f2(empty, ship) == 0.0  # one false positive on an empty image costs the whole image
    assert image_f2(ship, empty) == 0.0


def test_f2_partial_match():
    truth = _box(_box(np.zeros((768, 768), np.int32), 1, 0, 10, 0, 10), 2, 100, 110, 100, 110)
    pred = _box(np.zeros((768, 768), np.int32), 1, 0, 10, 0, 10)
    # 1 TP, 1 FN at every threshold: 5 / (5 + 4) = 0.5556
    assert image_f2(truth, pred) == pytest.approx(5 / 9)
    # IoU = 0.8 (10 x 8 inside 10 x 10) -> matched for thresholds 0.5 ... 0.75, i.e. 6 of 10 thresholds.
    shifted = _box(np.zeros((768, 768), np.int32), 1, 0, 10, 0, 8)
    single = _box(np.zeros((768, 768), np.int32), 1, 0, 10, 0, 10)
    assert image_f2(single, shifted) == pytest.approx(0.6)


def test_touching_ships_are_split_and_never_overlap():
    truth = _box(_box(np.zeros((768, 768), np.int32), 1, 100, 110, 100, 160), 2, 110, 120, 100, 160)
    body = (truth > 0).astype(np.float32)
    border = border_from_labels(truth).astype(np.float32)
    labels = instances_from_probs(body, border, ship_prob=0.9, gate_thr=0.5, mask_thr=0.5, min_area=0)
    assert labels.max() == 2
    assert image_f2(truth, labels) > 0.5
    rles = rles_from_labels(labels)
    decoded = sum(rle_decode(r).astype(int) for r in rles)
    assert decoded.max() == 1  # disjoint instances


def test_gate_and_min_area():
    body = np.zeros((768, 768), np.float32)
    body[0:3, 0:3] = 1  # 9 px blob
    body[50:70, 50:70] = 1
    border = np.zeros_like(body)
    assert instances_from_probs(body, border, 0.2, 0.5, 0.5, 0).max() == 0
    assert instances_from_probs(body, border, 0.9, 0.5, 0.5, 10).max() == 1
    assert instances_from_probs(body, border, 0.9, 0.5, 0.5, 0).max() == 2


def test_labels_from_rles_ignores_nan():
    assert labels_from_rles([np.nan]).max() == 0


def test_gate_sources_include_mean_with_aux_head():
    sources = gate_sources(np.array([0.2, 0.8]), {"vit": np.array([0.6, 1.0])})
    assert set(sources) == {"aux", "vit", "mean_aux_vit"}
    assert np.allclose(sources["mean_aux_vit"], [0.4, 0.9])


def test_snap_squares_off_a_ragged_rotated_ship():
    import cv2

    truth = np.zeros((768, 768), np.uint8)
    box = cv2.boxPoints(((300, 300), (60, 14), 30)).astype(np.int32)
    cv2.fillPoly(truth, [box], 1)
    ragged = truth.copy()
    ragged[::3, ::5] = 0  # holes along the hull and inside
    ragged = cv2.erode(ragged, np.ones((2, 2), np.uint8))
    truth_labels = truth.astype(np.int32)
    snapped = snap_to_rectangles(ragged.astype(np.int32))
    assert image_f2(truth_labels, snapped) > image_f2(truth_labels, ragged.astype(np.int32))
    assert snapped.max() == 1


def test_snap_keeps_instances_disjoint_and_respects_min_area():
    labels = _box(_box(np.zeros((768, 768), np.int32), 1, 100, 110, 100, 160), 2, 110, 120, 100, 160)
    snapped = snap_to_rectangles(labels)
    assert snapped.max() == 2
    assert image_f2(labels, snapped) == 1.0
    tiny = _box(np.zeros((768, 768), np.int32), 1, 5, 7, 5, 7)
    assert (snap_to_rectangles(tiny, min_area=10) == tiny).all()


def test_dihedral_tta_is_exact_for_an_equivariant_model():
    import torch

    from airbus_modern import predict_dihedral, predict_tta

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x[:, :2] * 4 - 2, x.mean((1, 2, 3))[:, None]

    x = torch.rand(2, 3, 32, 32)
    seg4, cls4, seg8, cls8 = predict_dihedral(Identity(), x)
    ref_seg, ref_cls = predict_tta(Identity(), x)
    expected = torch.sigmoid(x[:, :2] * 4 - 2)
    assert torch.allclose(seg4, ref_seg, atol=1e-6) and torch.allclose(cls4, ref_cls, atol=1e-6)
    assert torch.allclose(seg8, expected, atol=1e-6)
