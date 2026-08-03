from types import SimpleNamespace

from vllm_ascend.models.deepseek_v4 import _is_extra_main_model_layer_weight


def test_identifies_weights_outside_truncated_main_model():
    config = SimpleNamespace(num_hidden_layers=4)

    assert not _is_extra_main_model_layer_weight(config, "model.layers.0.self_attn.wq_a.weight")
    assert not _is_extra_main_model_layer_weight(config, "model.layers.3.mlp.experts.0.w1.weight")
    assert _is_extra_main_model_layer_weight(config, "model.layers.4.self_attn.attn_sink")
    assert _is_extra_main_model_layer_weight(config, "model.layers.42.mlp.gate.weight")


def test_does_not_classify_non_main_layer_weights_as_truncated():
    config = SimpleNamespace(num_hidden_layers=4)

    assert not _is_extra_main_model_layer_weight(config, "model.embed_tokens.weight")
    assert not _is_extra_main_model_layer_weight(config, "lm_head.weight")
    assert not _is_extra_main_model_layer_weight(config, "mtp.0.self_attn.wq_a.weight")
