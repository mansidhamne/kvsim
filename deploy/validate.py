"""Validation of every step of a GPU run. A run's numbers are only reported if this passes.

Each check returns PASS, WARN or FAIL:
  FAIL  the data can't be trusted as is (errors, missing measurements, broken model output,
        a "cold" request that hit the cache, a fit that doesn't describe the data)
  WARN  worth a look, doesn't invalidate the numbers (noisy repeats, small metric mismatches)

Checks are grouped by step: smoke, text_cold, image_cold, hits, load, server_metrics, fit.
`validate_run(smoke, raw, fitted)` returns a report; `report_ok(report)` is False on any FAIL.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Check:
    step: str
    name: str
    status: str
    detail: str = ""


def looks_like_text(s: str, min_len: int = 1) -> tuple[bool, str]:
    """Catch degenerate output (fp16 overflow typically gives '!!!!!!' or one token repeated)."""
    if s is None or len(s.strip()) < min_len:
        return False, "empty"
    if "�" in s:
        return False, "contains U+FFFD replacement characters"
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in s) / len(s)
    if printable < 0.9:
        return False, f"only {printable:.0%} printable"
    run, longest, prev = 0, 0, None
    for ch in s:
        run = run + 1 if ch == prev else 1
        longest, prev = max(longest, run), ch
    if longest >= 10:
        return False, f"one character repeated {longest} times"
    words = s.split()
    if len(words) >= 8 and len(set(words)) / len(words) < 0.2:
        return False, "the same few words repeated"
    return True, ""


def _rows_ok(step: str, rows: list[dict], checks: list[Check]) -> None:
    errs = [r for r in rows if r.get("error") or r.get("status") != 200]
    checks.append(Check(step, "no request errors", FAIL if errs else PASS,
                        f"{len(errs)} of {len(rows)} failed: {(errs[0].get('error') or '')[:120]}" if errs else f"{len(rows)} requests"))
    no_ttft = [r for r in rows if r.get("ttft_s") is None and not r.get("error")]
    checks.append(Check(step, "every request has a TTFT", FAIL if no_ttft else PASS,
                        f"{len(no_ttft)} of {len(rows)} missing" if no_ttft else ""))
    no_usage = [r for r in rows if r.get("prompt_tokens") is None or r.get("cached_tokens") is None]
    checks.append(Check(step, "server reported prompt and cached tokens", FAIL if no_usage else PASS,
                        f"{len(no_usage)} rows without usage details (is --enable-prompt-tokens-details set?)"
                        if no_usage else ""))
    wrong_len = [r for r in rows if r.get("completion_tokens") is not None
                 and r.get("max_tokens") is not None and r["completion_tokens"] != r["max_tokens"]]
    checks.append(Check(step, "output length exactly as requested", FAIL if wrong_len else PASS,
                        f"{len(wrong_len)} rows differ" if wrong_len else ""))


def _repeat_spread(step: str, rows: list[dict], key, checks: list[Check], limit: float = 1.3) -> None:
    groups = defaultdict(list)
    for r in rows:
        if r.get("ttft_s"):
            groups[key(r)].append(r["ttft_s"])
    worst, where = 1.0, None
    for k, v in groups.items():
        if len(v) > 1 and min(v) > 0:
            ratio = max(v) / min(v)
            if ratio > worst:
                worst, where = ratio, k
    checks.append(Check(step, f"repeats agree (max/min <= {limit})", WARN if worst > limit else PASS,
                        f"worst {worst:.2f}x at {where}"))


def check_smoke(sm: dict) -> list[Check]:
    c: list[Check] = []
    for k, v in sm.get("checks", {}).items():
        c.append(Check("smoke", k, PASS if v else (FAIL if k in sm.get("blocking", []) else WARN)))
    reply = sm.get("sanity_reply")
    if reply is None:
        c.append(Check("smoke", "model answers 'capital of France' correctly", FAIL,
                       "no sanity reply recorded (old run format)"))
    else:
        ok = "paris" in reply.lower()
        c.append(Check("smoke", "model answers 'capital of France' correctly", PASS if ok else FAIL,
                       repr(reply[:80])))
    for size, v in (sm.get("images") or {}).items():
        text = (v.get("first") or {}).get("output_text")
        if text is not None:
            good, why = looks_like_text(text)
            c.append(Check("smoke", f"image {size} reply is readable text", PASS if good else FAIL,
                           repr(text[:60]) if good else why))
    return c


def check_text_cold(rows: list[dict]) -> list[Check]:
    c: list[Check] = []
    _rows_ok("text_cold", rows, c)
    hit = [r for r in rows if (r.get("cached_tokens") or 0) > 0]
    c.append(Check("text_cold", "cold requests really were cold (0 cached)", FAIL if hit else PASS,
                   f"{len(hit)} rows had cache hits" if hit else ""))
    by_words = defaultdict(list)
    for r in rows:
        if r.get("prompt_tokens"):
            by_words[r["words"]].append(r["prompt_tokens"])
    ws = sorted(by_words)
    mono = all(min(by_words[a]) < min(by_words[b]) for a, b in zip(ws, ws[1:]))
    c.append(Check("text_cold", "prompt tokens grow with prompt length", PASS if mono else FAIL))
    _repeat_spread("text_cold", rows, lambda r: r["words"], c)
    return c


def check_image_cold(rows: list[dict], tokens_per_image: dict[str, int] | None,
                     earlier_uuids: set[str] | None = None) -> list[Check]:
    c: list[Check] = []
    _rows_ok("image_cold", rows, c)
    seen = set(earlier_uuids or ())
    reused = []
    for r in rows:
        for u in r.get("image_uuids") or []:
            if u in seen:
                reused.append(u)
            seen.add(u)
    if any("image_uuids" in r for r in rows):
        c.append(Check("image_cold", "every cold image is new (no uuid sent earlier in the run)",
                       FAIL if reused else PASS,
                       f"reused: {reused[:3]} (the encoder cache makes these look cheaper)" if reused else ""))
    hit = [r for r in rows if (r.get("cached_tokens") or 0) > 0]
    c.append(Check("image_cold", "cold requests really were cold (0 cached)", FAIL if hit else PASS,
                   f"{len(hit)} rows had cache hits" if hit else ""))
    if tokens_per_image:
        bad = []
        for r in rows:
            size = "x".join(map(str, r["size"]))
            want = tokens_per_image.get(size)
            got = (r.get("mm_tokens") or {}).get("image")
            if want and got != want * r["images"]:
                bad.append((size, r["images"], got))
        c.append(Check("image_cold", "image tokens = images x tokens per image", FAIL if bad else PASS,
                       f"mismatches: {bad[:3]}" if bad else ""))
    _repeat_spread("image_cold", rows, lambda r: (tuple(r["size"]), r["images"]), c)
    return c


def check_hits(rows: list[dict]) -> list[Check]:
    c: list[Check] = []
    _rows_ok("hits", rows, c)
    groups = defaultdict(dict)
    for r in rows:
        groups[(r["exp"], r.get("prefix_words", r.get("images")), r["rep"])][r["step"]] = r
    problems = []
    for key, g in groups.items():
        cold, newq, ident = g.get("cold"), g.get("new_question"), g.get("identical")
        if not (cold and newq and ident):
            problems.append(f"{key}: incomplete")
            continue
        if (cold.get("cached_tokens") or 0) > 0:
            problems.append(f"{key}: cold step hit the cache")
        if (newq.get("cached_tokens") or 0) < 0.9 * (newq.get("prompt_tokens") or 1):
            problems.append(f"{key}: new question cached only {newq.get('cached_tokens')} of {newq.get('prompt_tokens')}")
        if (ident.get("cached_tokens") or 0) < (newq.get("cached_tokens") or 0):
            problems.append(f"{key}: identical resend cached less than new question")
        if cold.get("ttft_s") and newq.get("ttft_s") and newq["ttft_s"] >= cold["ttft_s"]:
            problems.append(f"{key}: cache hit was not faster than cold")
    c.append(Check("hits", "cold -> hit -> identical behave as expected", FAIL if problems else PASS,
                   "; ".join(problems[:3])))
    return c


def check_load(rows: list[dict], output_tokens: int, block_size: int = 16) -> list[Check]:
    c: list[Check] = []
    for r in rows:
        n = r["concurrency"]
        lost = r["requests"] - r["ok"]
        c.append(Check("load", f"concurrency {n}: no failed requests", FAIL if lost else PASS,
                       f"{lost} of {r['requests']} failed" if lost else f"{r['requests']} requests"))
        if r.get("missing_ttft"):
            c.append(Check("load", f"concurrency {n}: every request has a TTFT", FAIL,
                           f"{r['missing_ttft']} missing"))
        ct = [x for x in (r.get("completion_tokens") or []) if x is not None]
        if ct:
            bad = [x for x in ct if x != output_tokens]
            c.append(Check("load", f"concurrency {n}: {output_tokens} output tokens each",
                           FAIL if bad else PASS, f"{len(bad)} differ" if bad else ""))
        cached = [x for x in (r.get("cached_tokens") or []) if x is not None]
        if any(x >= block_size for x in cached):
            c.append(Check("load", f"concurrency {n}: prompts were cold", FAIL,
                           f"max cached {max(cached)} tokens"))
        for text in r.get("sample_outputs") or []:
            good, why = looks_like_text(text)
            if not good:
                c.append(Check("load", f"concurrency {n}: outputs are readable text", FAIL,
                               f"{why}: {text[:60]!r}"))
                break
    tput = [r.get("req_per_s") or 0 for r in rows]
    drops = [(rows[i]["concurrency"], rows[i + 1]["concurrency"]) for i in range(len(rows) - 1)
             if tput[i + 1] < 0.9 * tput[i]]
    c.append(Check("load", "throughput does not fall as concurrency rises", WARN if drops else PASS,
                   f"drops at {drops}" if drops else ""))
    return c


def check_server_metrics(raw: dict, block_size: int = 16) -> list[Check]:
    """Client-side cached-token sums vs the server's prefix_cache_hits counter per experiment."""
    c: list[Check] = []
    for name, d in (raw.get("server_metrics") or {}).items():
        rows = raw.get(name) or []
        if name == "load":
            client_hits = sum(x or 0 for r in rows for x in (r.get("cached_tokens") or []))
            client_q = None
        else:
            client_hits = sum(r.get("cached_tokens") or 0 for r in rows)
            client_q = sum(r.get("prompt_tokens") or 0 for r in rows)
        server_hits, server_q = d.get("vllm:prefix_cache_hits"), d.get("vllm:prefix_cache_queries")
        if server_hits is None:
            c.append(Check("server_metrics", f"{name}: server counters readable", WARN, "missing"))
            continue
        tol = max(0.05 * max(client_hits, server_hits), block_size * max(1, len(rows)))
        ok = abs(server_hits - client_hits) <= tol
        c.append(Check("server_metrics", f"{name}: server cache hits match client", PASS if ok else WARN,
                       f"server {server_hits:.0f} vs client {client_hits:.0f} tokens"))
        if client_q is not None and server_q is not None:
            okq = abs(server_q - client_q) <= max(0.05 * client_q, block_size * len(rows))
            c.append(Check("server_metrics", f"{name}: server prompt tokens match client",
                           PASS if okq else WARN, f"server {server_q:.0f} vs client {client_q:.0f}"))
    return c


def check_fit(fitted: dict, raw: dict) -> list[Check]:
    c: list[Check] = []
    f, cm = fitted["fit"], fitted["cost_model"]
    c.append(Check("fit", "fit describes the data (R2 >= 0.98)", PASS if f["r2"] >= 0.98 else FAIL,
                   f"R2 {f['r2']:.4f}"))
    # A coefficient may come out slightly negative when its true value is ~0 (noise). Only fail
    # when the negative term matters: > 2% of the largest measured TTFT and > 5 ms.
    rows = [r for r in raw.get("text_cold", []) + raw.get("image_cold", []) if r.get("ttft_s")]
    n_max = max((r.get("prompt_tokens") or 0 for r in rows), default=0)
    t_max = max((r["ttft_s"] for r in rows), default=1.0)
    size = {"overhead_s": abs(f.get("overhead_s", 0)),
            "s_per_text_token": abs(f.get("s_per_text_token", 0)) * n_max,
            "attn_s_per_token_pair": abs(f.get("attn_s_per_token_pair", 0)) * n_max ** 2 / 2}
    neg = [k for k in size if f.get(k, 0) < 0 and size[k] > max(0.02 * t_max, 0.005)]
    noise = [k for k in size if f.get(k, 0) < 0 and k not in neg]
    c.append(Check("fit", "fitted coefficients are non-negative", FAIL if neg else PASS,
                   f"negative: {neg}" if neg else (f"~0 within noise: {noise}" if noise else "")))
    planned = raw.get("image_cold")
    if planned:
        measured = [r for r in planned if r.get("ttft_s") is not None]
        enough = len(measured) >= max(2, 0.8 * len(planned))
        c.append(Check("fit", "image cost based on >= 80% of planned image requests",
                       PASS if enough else FAIL, f"{len(measured)} of {len(planned)} have a TTFT; "
                       f"multiplier {cm.get('image_cost_multiplier'):.2f}"))
    return c


def validate_run(smoke: dict | None, raw: dict | None, fitted: dict | None,
                 output_tokens: int = 64, block_size: int = 16) -> dict:
    checks: list[Check] = []
    tokens_per_image = None
    if smoke:
        checks += check_smoke(smoke)
        tokens_per_image = {k: v.get("image_tokens") for k, v in (smoke.get("images") or {}).items()}
    if raw:
        if "text_cold" in raw:
            checks += check_text_cold(raw["text_cold"])
        if "image_cold" in raw:
            smoke_uuids = {u for v in ((smoke or {}).get("images") or {}).values()
                           for u in ((v.get("first") or {}).get("image_uuids") or [])}
            checks += check_image_cold(raw["image_cold"], tokens_per_image, smoke_uuids)
        if "hits" in raw:
            checks += check_hits(raw["hits"])
        if "load" in raw:
            out_tok = (raw.get("config") or {}).get("load_output_tokens", output_tokens)
            checks += check_load(raw["load"], out_tok, block_size)
        checks += check_server_metrics(raw, block_size)
    if fitted and raw:
        checks += check_fit(fitted, raw)
    counts = {s: sum(ch.status == s for ch in checks) for s in (PASS, WARN, FAIL)}
    return {"ok": counts[FAIL] == 0, "counts": counts, "checks": [asdict(ch) for ch in checks]}


def report_ok(report: dict) -> bool:
    return bool(report.get("ok"))


def print_report(report: dict, show_pass: bool = False) -> None:
    print(f"\nValidation: {report['counts']['PASS']} pass, {report['counts']['WARN']} warn, "
          f"{report['counts']['FAIL']} fail")
    for ch in report["checks"]:
        if ch["status"] != PASS or show_pass:
            print(f"  {ch['status']}  [{ch['step']}] {ch['name']}" + (f": {ch['detail']}" if ch["detail"] else ""))
    print("  RESULT:", "VALID" if report["ok"] else "NOT VALID: do not use these numbers")


# ------------------------------------------------------------------------------- phase 3

def validate_phase3(cells: list[dict], block_size: int = 16) -> dict:
    """Checks for a routing experiment: every policy saw the same trace on freshly reset
    workers, and every measured request is complete and sane."""
    checks: list[Check] = []
    by_load = defaultdict(list)
    for c in cells:
        by_load[c["active_convs"]].append(c)
    for n, group in by_load.items():
        fps = {c["trace_fingerprint"] for c in group}
        checks.append(Check("phase3", f"{n} convs: every policy replayed the same trace",
                            PASS if len(fps) == 1 else FAIL, f"fingerprints {sorted(fps)}"))
    for c in cells:
        tag = f"{c['active_convs']} convs / {c['policy']}"
        s = c["summary"]
        checks.append(Check("phase3", f"{tag}: worker caches reset before the run",
                            PASS if all(c.get("reset_ok") or [False]) else FAIL, str(c.get("reset_ok"))))
        checks.append(Check("phase3", f"{tag}: no failed requests", FAIL if s["errors"] else PASS,
                            f"{s['errors']} of {s['requests']}" if s["errors"] else f"{s['requests']} measured"))
        if s.get("missing_ttft"):
            checks.append(Check("phase3", f"{tag}: every request has a TTFT", FAIL, f"{s['missing_ttft']} missing"))
        if s.get("wrong_output_len"):
            checks.append(Check("phase3", f"{tag}: output length as requested", FAIL,
                                f"{s['wrong_output_len']} requests differ"))
        if s.get("hit_rate") is None:
            checks.append(Check("phase3", f"{tag}: server reported cached tokens", FAIL, "no usage details"))
        for text in s.get("sample_outputs") or []:
            good, why = looks_like_text(text)
            if not good:
                checks.append(Check("phase3", f"{tag}: outputs are readable text", FAIL, f"{why}: {text[:50]!r}"))
                break
        srv, cli = (c.get("server") or {}).get("prefix_hits"), (c.get("client_all") or {}).get("cached_tokens")
        if srv is None:
            checks.append(Check("phase3", f"{tag}: server cache counter readable", WARN, "missing"))
        else:
            tol = max(0.05 * max(srv, cli), block_size * c["client_all"]["requests"])
            checks.append(Check("phase3", f"{tag}: server cache hits match client",
                                PASS if abs(srv - cli) <= tol else WARN, f"server {srv:.0f} vs client {cli:.0f}"))
        if (s.get("late_frac") or 0) > 0.2:
            checks.append(Check("phase3", f"{tag}: follow-ups sent on schedule", WARN,
                                f"{s['late_frac']:.0%} of turns > 1 s late (system overloaded for this trace)"))
        if s.get("prediction_share_within5pp") is not None and c["policy"] == "cache_aware":
            checks.append(Check("phase3", f"{tag}: router's predicted reuse matches the engine",
                                PASS if s["prediction_share_within5pp"] >= 0.9 else WARN,
                                f"{s['prediction_share_within5pp']:.0%} of requests within 5 points of the "
                                f"engine's cached share; bias {s.get('prediction_bias_tokens') or 0:+.0f} tokens"))
        # Policies that act on prefill-time estimates are only meaningful if the estimates are
        # realistic. Caught the run-1 bug (image weight inside the quadratic term, ~9x too high).
        r = s.get("estimate_ratio_median")
        if r is not None and s.get("estimate_ratio_n", 0) >= 10:
            uses = c["policy"] in ("least_loaded", "session_affinity", "cache_aware")
            good = 0.5 <= r <= 2.0
            checks.append(Check("phase3", f"{tag}: router's prefill estimates are realistic",
                                PASS if good else (FAIL if uses else WARN),
                                f"median estimate / measured TTFT = {r:.2f} over {s['estimate_ratio_n']} "
                                f"mostly-uncached requests with no backlog (should be 0.5-2)"))
    counts = {k: sum(ch.status == k for ch in checks) for k in (PASS, WARN, FAIL)}
    return {"ok": counts[FAIL] == 0, "counts": counts, "checks": [asdict(ch) for ch in checks]}


# ------------------------------------------------------------------------------- phase 3b

def validate_phase3b(cells: list[dict], block_size: int = 16) -> dict:
    """Phase 3 checks per cell, plus: all three caches reset, policy order actually rotated
    across seeds, preemptions and router retries reported."""
    checks: list[Check] = []
    shim = []
    for c in cells:
        c2 = dict(c)
        c2["active_convs"] = f"{c['workload']} cache={c['cache']} load={c['load']:g} seed={c['seed']}"
        c2["reset_ok"] = [r.get("prefix", False) for r in c.get("resets") or [{}]]
        shim.append(c2)
    base = validate_phase3(shim, block_size)
    checks += [Check(**ch) for ch in base["checks"]]
    for c in shim:
        tag = f"{c['active_convs']} / {c['policy']}"
        res = c.get("resets") or []
        other = all(r.get("mm") and r.get("encoder") for r in res) if res else False
        checks.append(Check("phase3b", f"{tag}: multimodal and encoder caches reset",
                            PASS if other else FAIL, str(res)))
        pre = (c.get("server") or {}).get("preemptions")
        if pre:
            checks.append(Check("phase3b", f"{tag}: no preemptions", WARN,
                                f"{pre:.0f} requests preempted (KV pool too small for the running "
                                "requests: latency includes recomputation)"))
        if c.get("router_retries"):
            checks.append(Check("phase3b", f"{tag}: router connection retries", WARN,
                                f"{c['router_retries']} retried (stale keep-alive), none lost"))
    # With the model's own replies in the history, a follow-up's prompt is identical across
    # policies only if every earlier reply was identical (greedy decoding, but GPU batching is not
    # bit-exact). Report how many requests had identical replies in every policy.
    by_seed = defaultdict(list)
    for c in cells:
        if c.get("history") == "own" and c.get("reply_shas"):
            by_seed[(c["workload"], c["cache"], c["load"], c["seed"])].append(c)
    for key, cs in by_seed.items():
        if len(cs) < 2:
            continue
        keys = set.intersection(*(set(c["reply_shas"]) for c in cs))
        same = sum(1 for k in keys if len({c["reply_shas"][k] for c in cs}) == 1)
        share = same / len(keys) if keys else 0.0
        checks.append(Check("phase3b", f"{key[0]} cache={key[1]} load={key[2]:g} seed={key[3]}: "
                            "replies identical across policies", PASS if share >= 0.9 else WARN,
                            f"{share:.1%} of {len(keys)} requests (own-reply history: differing "
                            "replies make later prompts differ)"))
    groups = defaultdict(lambda: defaultdict(set))
    seeds = defaultdict(set)
    for c in cells:
        groups[(c["workload"], c["cache"], c["load"])][c["policy"]].add(c.get("order_index"))
        seeds[(c["workload"], c["cache"], c["load"])].add(c["seed"])
    for key, by_pol in groups.items():
        if len(seeds[key]) >= 2:
            fixed = [p for p, idx in by_pol.items() if len(idx) < 2]
            checks.append(Check("phase3b", f"{key[0]} cache={key[1]} load={key[2]:g}: policy order rotated across seeds",
                                FAIL if fixed else PASS, f"same position every time: {fixed}" if fixed else ""))
    counts = {k: sum(ch.status == k for ch in checks) for k in (PASS, WARN, FAIL)}
    return {"ok": counts[FAIL] == 0, "counts": counts, "checks": [asdict(ch) for ch in checks]}
