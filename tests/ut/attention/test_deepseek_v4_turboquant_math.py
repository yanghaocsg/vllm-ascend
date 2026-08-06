import math

import torch

from vllm_ascend.turboquant import tq_latent_store


def _sylvester_hadamard(order: int) -> torch.Tensor:
    matrix = torch.ones(1, 1, dtype=torch.float64)
    while matrix.shape[0] < order:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(order)


def test_shared_kv_hadamard_attention_is_basis_invariant():
    torch.manual_seed(0)
    head_dim = 512
    transform = _sylvester_hadamard(head_dim)
    signs = torch.where(torch.arange(head_dim) % 3 == 0, -1.0, 1.0).to(torch.float64)
    transform = signs.unsqueeze(1) * transform

    query = torch.randn(3, head_dim, dtype=torch.float64)
    shared_kv = torch.randn(7, head_dim, dtype=torch.float64)
    scale = 1.0 / math.sqrt(head_dim)

    reference = torch.softmax(query @ shared_kv.T * scale, dim=-1) @ shared_kv
    transformed_query = query @ transform
    transformed_kv = shared_kv @ transform
    transformed_output = torch.softmax(transformed_query @ transformed_kv.T * scale, dim=-1) @ transformed_kv
    restored_output = transformed_output @ transform.T

    torch.testing.assert_close(restored_output, reference, rtol=1e-10, atol=1e-10)


def test_turboquant_slot_layout_is_320_bytes():
    head_dim = 512
    packed_bytes = head_dim // 2
    scale_bytes = 2
    slot_alignment = 64
    slot_size = (packed_bytes + scale_bytes + slot_alignment - 1) // slot_alignment * slot_alignment

    assert packed_bytes == 256
    assert slot_size == 320
    assert (1024 - slot_size) / 1024 == 0.6875


def test_turboquant_code_norm_correction_preserves_slot_layout(monkeypatch):
    head_dim = 8
    packed_bytes = head_dim // 2
    slot_size = tq_latent_store.base_slot_size(head_dim)
    slot = torch.zeros(2, slot_size, dtype=torch.uint8)
    codes = torch.tensor(
        [
            [0x10, 0x32, 0x54, 0x76],
            [0x89, 0xAB, 0xCD, 0xEF],
        ],
        dtype=torch.uint8,
    )
    slot[:, :packed_bytes] = codes
    raw_norm = torch.tensor([2.0, 3.0], dtype=torch.float16)
    slot[:, packed_bytes : packed_bytes + 2] = raw_norm.view(torch.uint8).view(-1, 2)
    slot[:, packed_bytes + 2 :] = 0x5A

    byte_lutsq = torch.arange(1, 257, dtype=torch.float32)
    monkeypatch.setattr(tq_latent_store, "lutsq", lambda device, head_dim: byte_lutsq)

    corrected = tq_latent_store.apply_code_norm_correction(slot, head_dim=head_dim)

    expected_code_norm = torch.sqrt(byte_lutsq[codes.long()].sum(-1))
    expected_scale = (raw_norm.float() / expected_code_norm).to(torch.float16)
    actual_scale = corrected[:, packed_bytes : packed_bytes + 2].contiguous().view(torch.float16).view(-1)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(corrected[:, :packed_bytes], codes, rtol=0, atol=0)
    assert torch.equal(corrected[:, packed_bytes + 2 :], slot[:, packed_bytes + 2 :])
    assert corrected.shape == slot.shape
