"""Calibration 2: how much more does an image token cost than a text token, and is an
image's KV state reusable?

Needs a small vision model:   ollama pull moondream

  1. fit the model's TEXT prefill cost  (wall = a + b_text * tokens)
  2. send unique random-noise images with a short question; the log tells us how many
     tokens the image became. Attribute the extra time to those tokens:
         image_cost_multiplier = (time per image token) / (time per text token)
  3. same image, different question: is the image's KV reused (cached tokens, wall time)?

Merges `image_cost_multiplier` into calibration/results/calibration.json.

    python calibration/02_vision.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from harness import RESULTS, VISION_MODEL, OllamaServer, noise_png_b64, words  # noqa: E402

CAL = RESULTS / "calibration.json"


def merge(update_cost: dict, vision: dict):
    cal = json.loads(CAL.read_text()) if CAL.exists() else {"cost_model": {}}
    cal["cost_model"].update(update_cost)
    cal["vision"] = vision
    CAL.write_text(json.dumps(cal, indent=1))


def main():
    with OllamaServer(11530, num_parallel=1, num_ctx=2048, tag="calib2") as srv:
        try:
            print(f"loading {VISION_MODEL} + warm-up ...")
            srv.generate("hi", model=VISION_MODEL, raw=False, read_log=False)
            for i in range(3):
                srv.generate(words(40 + 40 * i, 777 + i), model=VISION_MODEL, raw=False,
                             read_log=False)
        except Exception as e:                                    # model missing / unsupported
            print("vision model unavailable:", e)
            merge({}, {"error": str(e)})
            return

        # ---- 1. text-only cost for this model (templated prompts, so tokens come from the log)
        T = []
        for n in (20, 60, 120, 200, 300):
            for r in range(3):
                res = srv.generate(words(n, 10_000 + n * 7 + r), model=VISION_MODEL, raw=False)
                if res.task and res.task.complete:
                    T.append((res.task.evaluated, res.wall_s))
        x, y = np.array(T, float).T
        b_text, a_text = np.polyfit(x, y, 1)
        print(f"text-only: {1 / b_text:.0f} tok/s, overhead {a_text * 1e3:.0f} ms ({len(T)} samples)")

        # ---- 2. image prompts
        Q = "What is in this picture?"
        text_only = srv.generate(Q, model=VISION_MODEL, raw=False)
        q_tokens = text_only.task.n_tokens if text_only.task else None
        I = []
        for i in range(6):
            res = srv.generate(Q, model=VISION_MODEL, raw=False, images=[noise_png_b64(1000 + i)])
            if res.task and res.task.complete:
                I.append({"wall": res.wall_s, "n_total": res.task.n_tokens,
                          "evaluated": res.task.evaluated, "cached": res.task.cached})
        if not I or q_tokens is None:
            print("could not read image-prompt accounting from the server log; aborting")
            merge({}, {"error": "no log accounting for image prompts"})
            return
        img_tokens = float(np.median([s["n_total"] for s in I])) - q_tokens
        wall = float(np.median([s["wall"] for s in I]))
        extra = wall - (a_text + b_text * q_tokens)
        b_img = extra / img_tokens
        mult = b_img / b_text
        # what the same total length would cost as text
        same_len_text = a_text + b_text * float(np.median([s["n_total"] for s in I]))
        print(f"image prompt: ~{img_tokens:.0f} image tokens, wall {wall * 1e3:.0f} ms "
              f"vs {same_len_text * 1e3:.0f} ms for an equally long TEXT prompt "
              f"-> image_cost_multiplier = {mult:.2f}")

        # ---- 3. image reuse
        img = noise_png_b64(5555)
        first = srv.generate(Q, model=VISION_MODEL, raw=False, images=[img])
        again = srv.generate("Describe the colors.", model=VISION_MODEL, raw=False, images=[img])
        reuse = {"first_wall_s": first.wall_s,
                 "first_cached": first.task.cached if first.task else None,
                 "second_wall_s": again.wall_s,
                 "second_cached": again.task.cached if again.task else None,
                 "second_total": again.task.n_tokens if again.task else None}
        print(f"same image, new question: first {first.wall_s * 1e3:.0f} ms, second "
              f"{again.wall_s * 1e3:.0f} ms, cached tokens on 2nd = {reuse['second_cached']} "
              f"of {reuse['second_total']}")

        # ---- 4. several images in one request (does the model accept it, what does it cost?)
        multi = None
        try:
            res = srv.generate(Q, model=VISION_MODEL, raw=False,
                               images=[noise_png_b64(8000), noise_png_b64(8001)])
            multi = {"wall_s": res.wall_s, "n_total": res.task.n_tokens if res.task else None}
            print("2 images in one request:", multi)
        except Exception as e:
            print("multi-image request not supported by this model/runtime:", str(e)[:80])

    vision = {"model": VISION_MODEL, "text_tokens_per_s": 1 / b_text, "text_overhead_s": a_text,
              "image_tokens": img_tokens, "image_prompt_wall_s": wall, "image_cost_multiplier": mult,
              "same_length_text_wall_s": same_len_text, "reuse": reuse, "multi_image": multi,
              "note": "multiplier measured on a different (vision) model than the text fit"}
    merge({"image_cost_multiplier": round(float(mult), 3)}, vision)
    print("merged image_cost_multiplier into", CAL)


if __name__ == "__main__":
    main()
