"""Offline tests for the calibration plumbing (no Ollama needed).

LOG is copied from a real llama-server log (Ollama 0.40.6) for three requests: a cold
prompt, the same prompt extended by a short suffix (cache hit), and a 1-token warmup.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "calibration"))
from harness import parse_tasks, words  # noqa: E402

from kvsim import CostModel  # noqa: E402

LOG = """\
slot   operator(): id  0 | task 0 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 1
slot   operator(): id  0 | task 0 | cached n_tokens = 0, memory_seq_rm [0, end)
slot print_timing: id  0 | task 0 | prompt eval time =     159.05 ms /     1 tokens (  159.05 ms per token,     6.29 tokens per second)
slot   operator(): id  0 | task 6 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 723
slot   operator(): id  0 | task 6 | cached n_tokens = 0, memory_seq_rm [0, end)
slot   operator(): id  0 | task 6 | cached n_tokens = 512, memory_seq_rm [512, end)
slot print_timing: id  0 | task 6 | prompt eval time =    1170.41 ms /   723 tokens (    1.62 ms per token,   617.73 tokens per second)
slot   operator(): id  0 | task 32 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 730
slot   operator(): id  0 | task 32 | cached n_tokens = 716, memory_seq_rm [716, end)
slot print_timing: id  0 | task 32 | prompt eval time =      42.53 ms /    14 tokens (    3.04 ms per token,   329.18 tokens per second)
"""


def test_parse_tasks_reads_cached_and_evaluated_tokens():
    tasks = parse_tasks(LOG)
    assert [t.task for t in tasks] == [6, 32]                 # the 1-token warmup is dropped
    cold, hit = tasks
    assert (cold.n_tokens, cold.cached, cold.evaluated) == (723, 0, 723)   # later 512-line is a batch checkpoint
    assert (hit.n_tokens, hit.cached, hit.evaluated) == (730, 716, 14)
    assert hit.n_tokens == hit.cached + hit.evaluated
    assert all(t.complete for t in tasks)


def test_incomplete_task_is_not_complete():
    partial = "\n".join(LOG.splitlines()[3:5])
    (t,) = parse_tasks(partial)
    assert not t.complete


def test_words_are_deterministic_and_prefix_stable():
    assert words(50, 1) == words(50, 1)
    assert words(50, 1) != words(50, 2)
    assert words(80, 3).startswith(words(50, 3))               # same seed -> shared prefix


def test_cost_model_from_calibration(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"cost_model": {"prefill_tokens_per_s": 870.0,
                                            "fixed_overhead_s": 0.04,
                                            "attn_s_per_token_pair": 0.0}}))
    m = CostModel.from_calibration(p, block_size=8)
    assert m.prefill_tokens_per_s == 870.0 and m.block_size == 8
    # 100 uncached tokens at 870 tok/s + overhead
    assert abs(m.prefill_time(100 / 8) - (0.04 + 100 / 870)) < 1e-9
