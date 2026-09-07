"""
Tests for nanochat/architectures/: the layer outside modelcore that turns a --depth dial
(presets.py) or an old checkpoint (legacy.py) into a modelcore.ModelConfig. See the Stage 7 plan
in docs/roadmap.md.

python -m pytest tests/test_architectures.py -v
"""
import hashlib
import json
import os

import pytest
import torch

from modelcore import ModelManager
from modelcore.model import Model
from modelcore.tests import GOLDENS_DIR as MODELCORE_GOLDENS_DIR, TINY_DIR as MODELCORE_TINY_DIR

from nanochat.architectures import legacy, presets

GOLDENS_DIR = os.path.join(os.path.dirname(__file__), "goldens")


def _tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _load_golden(name):
    with open(os.path.join(GOLDENS_DIR, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def _load_composed_golden(preset):
    """The tiny_composed_* goldens are modelcore's own baseline (moved into modelcore/tests/goldens
    at Stage 8, see docs/roadmap.md) -- this is the one place nanochat's own tests read modelcore's
    test data, since presets.expand's job is exactly to reproduce that same tree."""
    with open(os.path.join(MODELCORE_GOLDENS_DIR, f"tiny_composed_{preset}.json"), encoding="utf-8") as f:
        return json.load(f)


def _assert_matches_composed_golden(manager, config, golden):
    stats = manager.stats(config)
    acc = golden["accounting"]
    assert stats.num_params == acc["num_scaling_params"]["total"]
    assert stats.num_matmul_params == acc["num_matmul_params"]
    assert stats.flops_per_token == acc["estimate_flops"]
    assert stats.kv_bytes_per_token() == acc["kv_bytes_per_token"]
    assert stats.kv_cache_spec == acc["kv_cache_spec"]


@pytest.fixture
def manager():
    return ModelManager()


# -----------------------------------------------------------------------------
# presets.expand -- reproduces the four deleted native architectures' own derivation

PRESET_CASES = [
    ("gpt", dict(depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32, vocab_size=128, window_pattern="SSSL")),
    ("llama", dict(depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32, vocab_size=128, window_pattern="L")),
    ("llama_kvshare", dict(depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32, vocab_size=128, window_pattern="L", kv_share_frac=0.5)),
    ("llama_kvshare_win", dict(depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32, vocab_size=128, window_pattern="SSSL", kv_share_frac=0.5)),
]


@pytest.mark.parametrize("preset,kwargs", PRESET_CASES)
def test_expand_matches_pre_refactor_golden(manager, preset, kwargs):
    config = presets.expand(preset, **kwargs)
    report = manager.validate_config(config)
    assert report.ok, report.errors

    golden = _load_composed_golden(preset)
    _assert_matches_composed_golden(manager, config, golden)

    state = torch.load(os.path.join(MODELCORE_TINY_DIR, f"tiny_composed_{preset}", "model_000000.pt"), map_location="cpu")
    with torch.device("meta"):
        model = Model(config)
    model.to_empty(device="cpu")
    model.init_weights()
    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    T = min(8, config.sequence_len)
    idx = (torch.arange(T) % config.vocab_size).long().unsqueeze(0)
    with torch.no_grad():
        logits = model(idx)
    assert _tensor_hash(logits.float()) == golden["logits_hash"]


def test_expand_llama_kvshare_materializes_derive_compute_kv_slots_exactly():
    from nanochat.architectures.derive import compute_kv_slots

    n_layer, kv_share_frac = 8, 0.5
    config = presets.expand_llama_kvshare(depth=n_layer, aspect_ratio=16, head_dim=32,
                                           max_seq_len=32, vocab_size=128, kv_share_frac=kv_share_frac)
    expected_slots = compute_kv_slots(n_layer, kv_share_frac)
    n_own = max(expected_slots) + 1
    actual_slots = [b.params["kv_slot"] for b in config.body.params["blocks"]]
    actual_produces_kv = [b.params["produces_kv"] for b in config.body.params["blocks"]]
    assert actual_slots == expected_slots
    assert actual_produces_kv == [slot == i for i, slot in enumerate(expected_slots)]
    assert sum(actual_produces_kv) == n_own < n_layer


def test_expand_llama_kvshare_win_windows_match_compute_window_sizes():
    from nanochat.architectures.derive import compute_window_sizes

    n_layer, pattern, seq_len = 8, "SSSL", 512
    config = presets.expand_llama_kvshare_win(depth=n_layer, aspect_ratio=16, head_dim=32,
                                               max_seq_len=seq_len, vocab_size=128,
                                               window_pattern=pattern, kv_share_frac=0.5)
    expected = compute_window_sizes(pattern, n_layer, seq_len)
    actual = [b.params["window"] for b in config.body.params["blocks"]]
    assert actual == expected
    assert actual[-1] == seq_len  # final layer always forced to full context


# -----------------------------------------------------------------------------
# derive.compute_window_sizes itself (moved from tests/test_modelcore_components.py at Stage 8 --
# it's a depth-dial/window-pattern policy rule, not a modelcore component)

def test_compute_window_sizes_full_context():
    from nanochat.architectures.derive import compute_window_sizes
    ws = compute_window_sizes("L", n_layer=4, sequence_len=512)
    assert ws == [512] * 4


def test_compute_window_sizes_last_layer_always_full_context():
    from nanochat.architectures.derive import compute_window_sizes
    # Every layer requests a short window, but the final layer is always forced to full context.
    ws = compute_window_sizes("SS", n_layer=3, sequence_len=1024)
    assert ws[0] < 1024 and ws[1] < 1024
    assert ws[-1] == 1024


def test_compute_window_sizes_tiles_pattern_across_layers():
    from nanochat.architectures.derive import compute_window_sizes
    ws = compute_window_sizes("SL", n_layer=4, sequence_len=2048)
    assert ws[0] < 2048  # S
    assert ws[1] == 2048  # L
    assert ws[2] < 2048  # S (pattern repeats)
    assert ws[3] == 2048  # L, also forced as the last layer


def test_compute_window_sizes_invalid_chars_assert():
    from nanochat.architectures.derive import compute_window_sizes
    with pytest.raises(AssertionError):
        compute_window_sizes("X", n_layer=2, sequence_len=128)


def test_kv_sharing_strictly_shrinks_params_and_kv_bytes_vs_plain_llama(manager):
    """Same shape, only kv_share_frac differs from plain llama's implicit "no sharing" -- sharing
    should have strictly fewer params (dropped c_k/c_v on consumer layers) and strictly fewer
    KV-cache bytes/token (fewer distinct slots)."""
    dims = dict(depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32, vocab_size=128)
    llama = presets.expand_llama(**dims)
    kvshare = presets.expand_llama_kvshare(**dims, kv_share_frac=0.5)
    llama_stats = manager.stats(llama)
    kvshare_stats = manager.stats(kvshare)
    assert kvshare_stats.num_params < llama_stats.num_params
    assert kvshare_stats.kv_bytes_per_token() < llama_stats.kv_bytes_per_token()


def test_windowing_lowers_flops_at_identical_param_count(manager):
    """Windowing changes nothing about parameter count (it's a mask, not a shape change) but
    strictly lowers FLOPs/token relative to full-context at the same shape. sequence_len=512
    because compute_window_sizes's short-window formula rounds up to a 128-token tile, so at
    tiny sequence lengths "short" already has no effect once _effective_window caps it."""
    dims = dict(depth=4, aspect_ratio=16, head_dim=32, max_seq_len=512, vocab_size=128, kv_share_frac=0.5)
    full_context = presets.expand_llama_kvshare(**dims, window_pattern="L")
    windowed = presets.expand_llama_kvshare_win(**dims, window_pattern="SL")
    assert manager.stats(windowed).num_params == manager.stats(full_context).num_params
    assert manager.stats(windowed).flops_per_token < manager.stats(full_context).flops_per_token


def test_expand_unknown_preset_raises(manager):
    with pytest.raises(ValueError, match="Unknown preset"):
        presets.expand("nonexistent", depth=4)


def test_resolve_model_config_from_json_file(tmp_path, manager):
    config = presets.expand("llama", depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32, vocab_size=128)
    path = tmp_path / "tree.json"
    path.write_text(json.dumps(manager.config_to_dict(config)))
    resolved = presets.resolve_model_config(str(path), depth=999, aspect_ratio=1, head_dim=1, max_seq_len=1, vocab_size=1)
    assert resolved == config


def test_resolve_model_config_rejects_arch_opt_with_json_file(tmp_path, manager):
    config = presets.expand("llama", depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32, vocab_size=128)
    path = tmp_path / "tree.json"
    path.write_text(json.dumps(manager.config_to_dict(config)))
    with pytest.raises(AssertionError):
        presets.resolve_model_config(str(path), depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32,
                                      vocab_size=128, arch_opts={"kv_share_frac": 0.5})


def test_resolve_reference_config_re_derives_at_a_different_depth():
    config = presets.expand("llama_kvshare", depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32,
                             vocab_size=128, kv_share_frac=0.5)
    ref = presets.resolve_reference_config(config, ref_depth=8)
    assert ref.n_layer == 8
    assert ref.reference["preset"] == "llama_kvshare"


# -----------------------------------------------------------------------------
# legacy.migrate_config -- flat dict (any pre-modelcore generation) -> ModelConfig

def test_migrate_config_is_pass_through_for_current_format(manager):
    config = presets.expand("gpt", depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32, vocab_size=128)
    d = manager.config_to_dict(config)
    assert legacy.migrate_config(d) == config


def test_patch_missing_config_keys_backfills_window_pattern():
    d = {"sequence_len": 8, "vocab_size": 16, "n_layer": 2, "n_head": 1, "n_kv_head": 1, "n_embd": 4}
    patched = legacy.patch_missing_config_keys(dict(d))
    assert patched["window_pattern"] == "L"


def test_migrate_config_defaults_missing_arch_to_gpt(manager):
    config = presets.expand("gpt", depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32, vocab_size=128, window_pattern="L")
    flat = {"sequence_len": config.sequence_len, "vocab_size": config.vocab_size, "n_layer": 4,
            "n_head": config.n_embd // 32, "n_kv_head": config.n_embd // 32, "n_embd": config.n_embd,
            "window_pattern": "L"}  # no "arch" key at all, like d6
    migrated = legacy.migrate_config(flat)
    assert migrated.n_layer == 4
    assert manager.validate_config(migrated).ok


@pytest.mark.parametrize("arch", ["gpt", "llama", "llama_kvshare", "llama_kvshare_win"])
def test_migrate_config_matches_direct_assembly_for_every_flat_arch(manager, arch):
    n_layer, n_head, n_embd = 4, 2, 64
    flat = {"sequence_len": 32, "vocab_size": 128, "n_layer": n_layer, "n_head": n_head,
            "n_kv_head": n_head, "n_embd": n_embd, "window_pattern": "L", "arch": arch}
    if "kvshare" in arch:
        flat["kv_share_frac"] = 0.5
    migrated = legacy.migrate_config(flat)
    assert manager.validate_config(migrated).ok
    assert migrated.n_layer == n_layer
    assert migrated.n_embd == n_embd


# -----------------------------------------------------------------------------
# legacy state-dict migrations -- synthetic, mirroring the old (deleted)
# nanochat/model/gpt/migrations.py test suite

def _tiny_gpt_and_state(manager):
    config = presets.expand("gpt", depth=4, aspect_ratio=16, head_dim=32, max_seq_len=32, vocab_size=128, window_pattern="L")
    model = manager.create_model(config, device=torch.device("cpu"), seed=0)
    return config, model


def _old_layout_state_dict(model_state, n_layer):
    """Hand-builds the pre-Stage-2 flat layout an equivalent gpt checkpoint would have had,
    reusing the current model's own tensor values (via the current body-prefixed keys)."""
    old_sd = {
        "transformer.wte.weight": model_state["embedding.wte.weight"],
        "smear_gate.weight": model_state["embedding.smear.gate.weight"],
        "smear_lambda": model_state["embedding.smear.lambda_"],
        "lm_head.weight": model_state["unembedding.lm_head.weight"],
        "backout_lambda": model_state["body.backout_lambda"],
    }
    old_sd["resid_lambdas"] = torch.stack([model_state[f"body.blocks.{i}.resid_lambda"] for i in range(n_layer)])
    old_sd["x0_lambdas"] = torch.stack([model_state[f"body.blocks.{i}.x0_lambda"] for i in range(n_layer)])
    for key, value in model_state.items():
        if not key.startswith("body.blocks."):
            continue
        _, _, i, rest = key.split(".", 3)
        if rest in ("resid_lambda", "x0_lambda"):
            continue
        if rest.startswith("attn.value_embed."):
            old_sd[f"value_embeds.{i}.{rest.removeprefix('attn.value_embed.')}"] = value
        else:
            old_sd[f"transformer.h.{i}.{rest}"] = value
    return old_sd


def test_patch_gpt_state_dict_layout_renames_old_keys_and_preserves_values():
    manager = ModelManager()
    config, model = _tiny_gpt_and_state(manager)
    n_layer = config.n_layer
    real_sd = model.state_dict()
    old_sd = _old_layout_state_dict(real_sd, n_layer)
    assert len(old_sd) == len(real_sd) - 2 * (n_layer - 1)  # resid/x0 collapse to 1 tensor each

    new_sd = legacy.patch_gpt_state_dict_layout(dict(old_sd), n_layer, log=lambda msg: None)
    new_sd = legacy.patch_body_prefix(new_sd, "gpt", log=lambda msg: None)
    assert set(new_sd.keys()) == set(real_sd.keys())
    for key, value in real_sd.items():
        assert torch.equal(new_sd[key], value), f"mismatch in {key}"


def test_patch_gpt_state_dict_layout_is_idempotent_on_new_layout():
    manager = ModelManager()
    _, model = _tiny_gpt_and_state(manager)
    real_sd = model.state_dict()  # already body-prefixed, no transformer.* keys at all
    patched = legacy.patch_gpt_state_dict_layout(dict(real_sd), model.config.n_layer, log=lambda msg: None)
    assert patched.keys() == real_sd.keys()
    for key in real_sd:
        assert torch.equal(patched[key], real_sd[key])


def test_patch_body_prefix_is_idempotent():
    manager = ModelManager()
    _, model = _tiny_gpt_and_state(manager)
    real_sd = model.state_dict()
    patched = legacy.patch_body_prefix(dict(real_sd), "gpt", log=lambda msg: None)
    assert patched.keys() == real_sd.keys()


def test_patch_missing_state_keys_backfills_identity_values():
    n_layer = 4
    data = {"blocks_placeholder": None}  # deliberately missing "blocks.0.resid_lambda"
    del data["blocks_placeholder"]
    patched = legacy.patch_missing_state_keys(dict(data), n_layer, log=lambda msg: None)
    assert torch.equal(patched["resid_lambdas"], torch.ones(n_layer))
    assert torch.equal(patched["x0_lambdas"], torch.zeros(n_layer))


def test_patch_missing_state_keys_is_a_noop_on_new_layout():
    data = {"blocks.0.resid_lambda": torch.tensor(1.0)}
    patched = legacy.patch_missing_state_keys(dict(data), 4, log=lambda msg: None)
    assert patched is data or patched == data


# -----------------------------------------------------------------------------
# legacy optimizer-state migrations -- synthetic, mirroring the old test suite exactly

def _old_optimizer_data(n_layer, n_value_embedding=3, n_smear=3, matrix_shape_counts=(2, 3)):
    """Synthetic OLD-layout (old GPT.setup_optimizer policy order) optimizer_data:
    [unembedding(1), embedding(1), value_embedding(n_value_embedding), resid_scalar(1, an
    [n_layer]-shaped tensor), x0_scalar(1, ditto), smear(n_smear, the 3rd being backout_lambda
    when n_smear=3), *matrix(one group per shape)]."""
    idx = 0
    groups = []
    state = {}

    def add_group(kind, count):
        nonlocal idx
        params = list(range(idx, idx + count))
        for p in params:
            state[p] = {"step": 7, "exp_avg": torch.tensor(float(p)), "exp_avg_sq": torch.tensor(float(p) ** 2)}
        idx += count
        groups.append({"kind": kind, "lr": 0.1, "betas": (0.8, 0.95), "eps": 1e-10, "weight_decay": 0.0, "params": params})
        return params

    add_group("adamw", 1)  # 0: unembedding
    add_group("adamw", 1)  # 1: embedding
    add_group("adamw", n_value_embedding)  # 2: value_embedding
    resid_params = add_group("adamw", 1)  # 3: resid_scalar
    state[resid_params[0]] = {"step": 7, "exp_avg": torch.arange(n_layer, dtype=torch.float32), "exp_avg_sq": torch.arange(n_layer, dtype=torch.float32) ** 2}
    x0_params = add_group("adamw", 1)  # 4: x0_scalar
    state[x0_params[0]] = {"step": 7, "exp_avg": torch.arange(n_layer, dtype=torch.float32) * 10, "exp_avg_sq": torch.arange(n_layer, dtype=torch.float32) * 100}
    add_group("adamw", n_smear)  # 5: smear (3rd member is backout_lambda, if n_smear == 3)
    for count in matrix_shape_counts:  # 6+: matrix, one group per shape
        add_group("muon", count)
    return {"state": state, "param_groups": groups}


def test_patch_resid_x0_split_moves_state_and_renumbers():
    n_layer = 4
    data = _old_optimizer_data(n_layer)
    resid_old_idx = data["param_groups"][3]["params"][0]
    x0_old_idx = data["param_groups"][4]["params"][0]
    resid_exp_avg_old = data["state"][resid_old_idx]["exp_avg"].clone()
    x0_exp_avg_old = data["state"][x0_old_idx]["exp_avg"].clone()
    later_group_old_params = [p for g in data["param_groups"][5:] for p in g["params"]]
    original_later_state = {p: data["state"][p] for p in later_group_old_params}

    patched = legacy._patch_resid_x0_split(data, n_layer, log=lambda msg: None)

    resid_group = patched["param_groups"][3]
    x0_group = patched["param_groups"][4]
    assert len(resid_group["params"]) == n_layer
    assert len(x0_group["params"]) == n_layer
    for i, p in enumerate(resid_group["params"]):
        assert patched["state"][p]["exp_avg"].item() == resid_exp_avg_old[i].item()
    for i, p in enumerate(x0_group["params"]):
        assert patched["state"][p]["exp_avg"].item() == x0_exp_avg_old[i].item()

    later_group_new_params = [p for g in patched["param_groups"][5:] for p in g["params"]]
    assert len(later_group_new_params) == len(later_group_old_params)
    for old_p, new_p in zip(later_group_old_params, later_group_new_params):
        assert patched["state"][new_p] is original_later_state[old_p]


def test_patch_resid_x0_split_is_idempotent():
    n_layer = 4
    data = _old_optimizer_data(n_layer)
    once = legacy._patch_resid_x0_split(data, n_layer, log=lambda msg: None)
    twice = legacy._patch_resid_x0_split(once, n_layer, log=lambda msg: None)
    assert twice is once


def test_split_backout_lambda_preserves_state_and_flat_indices():
    n_layer = 4
    data = _old_optimizer_data(n_layer)
    smear_group = data["param_groups"][5]
    gate_idx, lambda_idx, backout_idx = smear_group["params"]
    backout_state_before = data["state"][backout_idx]

    patched = legacy._split_backout_lambda_from_smear(data, log=lambda msg: None)
    groups = patched["param_groups"]
    assert len(groups) == len(data["param_groups"]) + 1
    new_smear, backout_group = groups[5], groups[6]
    assert new_smear["params"] == [gate_idx, lambda_idx]
    assert backout_group["params"] == [backout_idx]
    # Flat index (and its state) is untouched -- only the group boundary moved.
    assert patched["state"][backout_idx] is backout_state_before
    # Every other group's flat indices are unaffected.
    assert groups[7:] == data["param_groups"][6:]


def test_split_backout_lambda_is_a_noop_without_a_3_member_smear_group():
    n_layer = 4
    data = _old_optimizer_data(n_layer, n_smear=2)  # no backout_lambda in the smear group
    patched = legacy._split_backout_lambda_from_smear(data, log=lambda msg: None)
    assert patched is data


def test_migrate_optimizer_state_is_noop_for_non_gpt_arch():
    n_layer = 4
    data = _old_optimizer_data(n_layer, n_smear=0)
    flat_config = {"n_layer": n_layer, "arch": "llama"}
    migrated = legacy.migrate_optimizer_state(data, flat_config, "llama", n_layer)
    assert migrated is data


def test_migrate_optimizer_state_is_noop_for_current_format():
    n_layer = 4
    data = _old_optimizer_data(n_layer)
    current_config = {"format": "modelcore.v1"}
    migrated = legacy.migrate_optimizer_state(data, current_config, "gpt", n_layer)
    assert migrated is data


def test_migrate_optimizer_state_applies_both_fixes_together():
    """The resid/x0 split and the backout_lambda re-group compose correctly when both run in
    sequence (migrate_optimizer_state's actual call order) -- total param count is preserved, no
    state is lost, and the group count grows by exactly one (backout_scalar carved out). The real
    end-to-end proof (migrated state actually loads into a real optimizer) is
    test_legacy_migration_matches_real_checkpoint_golden below, against d6's real shard."""
    n_layer = 4
    data = _old_optimizer_data(n_layer)
    total_params_before = sum(len(g["params"]) for g in data["param_groups"])

    migrated = legacy.migrate_optimizer_state(data, {"n_layer": n_layer}, "gpt", n_layer)

    total_params_after = sum(len(g["params"]) for g in migrated["param_groups"])
    # resid/x0 grow from 1 param each to n_layer each; backout_lambda moves group, doesn't vanish.
    assert total_params_after == total_params_before + 2 * (n_layer - 1)
    assert len(migrated["param_groups"]) == len(data["param_groups"]) + 1
    assert len(migrated["state"]) == len(data["state"]) + 2 * (n_layer - 1)


# -----------------------------------------------------------------------------
# Full migrate_checkpoint against the real on-disk checkpoints (skip if this machine doesn't have
# them) -- the authoritative proof that the whole chain (config + state dict + optimizer state)
# reproduces tests/goldens/*.json exactly.

REAL_LEGACY_SOURCES = {
    "d6": ("base_checkpoints", "d6", "005000"),
    "contest_h100run_gpt_d12": ("base_checkpoints", "contest_h100run_gpt_d12", "002511"),
    "contest_h100run_llama_d12": ("base_checkpoints", "contest_h100run_llama_d12", "002150"),
    "contest_h100run_llama_kvshare_win_d12": ("base_checkpoints", "contest_h100run_llama_kvshare_win_d12", "002659"),
    "contest_d12test_llama_kvshare_d12": ("base_checkpoints", "contest_d12test_llama_kvshare_d12", "002258"),
}


@pytest.mark.parametrize("golden_name", sorted(REAL_LEGACY_SOURCES))
def test_legacy_migration_matches_real_checkpoint_golden(manager, golden_name):
    from nanochat.common import get_base_dir

    subdir, tag, step = REAL_LEGACY_SOURCES[golden_name]
    checkpoint_dir = os.path.join(get_base_dir(), subdir, tag)
    model_path = os.path.join(checkpoint_dir, f"model_{step}.pt")
    if not os.path.isfile(model_path):
        pytest.skip(f"{model_path} not present on this machine")

    golden = _load_golden(golden_name)
    raw_config = dict(golden["meta_model_config"])
    raw_state = torch.load(model_path, map_location="cpu")
    raw_state = {k: (v.float() if v.dtype == torch.bfloat16 else v) for k, v in raw_state.items()}

    config, migrated_state = legacy.migrate_checkpoint(raw_config, raw_state)
    assert manager.validate_config(config).ok

    model = manager.create_model(config, device=torch.device("cpu"))
    model.load_state_dict(migrated_state, strict=True, assign=True)
    model.eval()

    new_multiset = sorted([[list(v.shape), str(v.dtype), _tensor_hash(v)] for v in model.state_dict().values()])
    assert new_multiset == golden["state_dict_fingerprint"]["multiset"]
    _assert_matches_composed_golden(manager, config, golden)

    T = min(8, config.sequence_len)
    idx = (torch.arange(T) % config.vocab_size).long().unsqueeze(0)
    with torch.no_grad():
        logits = model(idx)
    assert _tensor_hash(logits.float()) == golden["logits_hash"]

    optim_path = os.path.join(checkpoint_dir, f"optim_{step}_rank0.pt")
    if os.path.isfile(optim_path):
        raw_optimizer_data = torch.load(optim_path, map_location="cpu")
        arch = golden["meta_model_config"].get("arch", "gpt")
        migrated_optimizer_data = legacy.migrate_optimizer_state(raw_optimizer_data, raw_config, arch, config.n_layer)
        optimizer = manager.create_optimizer(model)
        optimizer.load_state_dict(migrated_optimizer_data)  # must not raise
        state = optimizer.state_dict()
        assert len(state["state"]) == golden["optimizer_shard"]["num_state_entries"]
