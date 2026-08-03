import math

import torch


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
