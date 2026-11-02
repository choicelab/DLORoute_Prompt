#!/usr/bin/env python3
"""The VLM prompts of LongHorRouting_v45, reproduced byte for byte from the committed code (git HEAD 945268b).

The original builds them in LearningBaseline/rl_rope/vlm_utils/hlvlmhandler.py (text) and gpt4_interface.py (message
format). A conversation is a JSON list of OpenAI chat messages. Only queries are added; the model's replies are not,
except an empty reply, which is kept as an empty assistant message before the query is sent again.
  [1] system message      role "developer": sysaddon + dev_default_prompt_flat.txt, then 2 in-context examples
                          ("If you see the following, you should command pull/insert" + an image)
  [2] order verification  role "user": orderaddon + order_prompt.txt (colour: order_prompt_color.txt) + full image.
                          Asked once per episode; the answer is assumed correct.
  [3] step prompt         role "user": "You can now predict action ..." on the first step, then "Remind that the last
                          action is <action> and the target clip is <clip> clip. ...", + full image + zoom-in.
                          The reply ends with "<yes/no> <Insert/Pull/Flatten> <clip>" (skill and its parameter).

  python make_prompts.py                    write outputs/<scenario>.json
  python make_prompts.py --show spatial 2   print the prompts of one config
  python make_prompts.py --example DIR      rebuild a logged episode (DIR has chat_*.json, log_*.json, image_*.png,
                                            zoomin_*.png) as a concrete example to read: images by file name, each
                                            query followed by the model's logged reply. Printed, written to
                                            outputs/examples/<DIR name>.json, and checked byte for byte against
                                            the logged chat_*.json
"""
import argparse
import base64
import glob
import io
import json
import os
import re

import numpy as np
from PIL import Image

from scenarios import SCENARIOS

HERE = os.path.dirname(os.path.abspath(__file__))
REQUEST = {"model": "gpt-5-2025-08-07", "reasoning_effort": "low"}   # the rest of the request body (curl_openai)
# In-context examples, in the order os.listdir() gives for vlm_example_imgs on the run machine: pull first on the local
# machine; the server runs sent insert first (--example reads the order from the log).
ICL = [("pull", "sample_images/icl/pull.png"), ("insert", "sample_images/icl/insert.png")]


def read(name):
    """A committed prompt file: LearningBaseline/rl_rope/vlm_utils/<name>, copied unchanged into prompts/."""
    with open(os.path.join(HERE, "prompts", name)) as f:
        return f.read()


# [1] system message (HLVLMHandler.initialize_prompt)
def system_message(sysaddon, color_clip, num_clips):
    lines = read("dev_default_prompt_flat.txt").split("\n")
    if color_clip:
        lines[-2:] = ["You need to provide whether the task is completed? what is the next action? What color is the target clip?",
                      "The last three words of your response must be in the format yes/no Insert/Pull/Flatten Red/Green/Blue/Yellow'"]
    elif num_clips == 4:
        lines[-2:] = ["You need to provide whether the task is completed? what is the next action? What height is the target clip located in the scene?",
                      "The last three words of your response must be in the format yes/no Insert/Pull/Flatten Top-most/Top/Bottom/Bottom-most'"]
    return sysaddon + "\n".join(lines)


# [2] order verification prompt (HLVLMHandler.clipOrderVerification)
def order_prompt(orderaddon, color_clip, num_clips):
    if color_clip:
        return orderaddon + read("order_prompt_color.txt")
    if num_clips == 4:   # the committed handler opens vlm_utils/order_prompt_4clip.txt here, which is not committed
        raise FileNotFoundError("order_prompt_4clip.txt is not committed")
    return orderaddon + read("order_prompt.txt")


# [3] step prompt (HLVLMHandler.query). last_action, target: the last two words of the previous reply (first step:
# target = clip 1 as named in the order reply). insert_trial: the env's counter AffordanceEnv.insert_trial.
def step_prompt(last_action, target, insert_trial=0, first_step=False):
    body = read("use_default_prompt_flat.txt")
    if first_step:
        return (f"You can now predict action for the image. The zoomed-in view is centered at the {target} clip.\n"
                + ".".join(body.split(".")[1:]))
    text = f"Remind that the last action is {last_action} and the target clip is {target} clip."
    if insert_trial:
        text += f"You have prompted insert for {insert_trial} times."
    return text + f" The zoomed-in view is centered at the {target} clip.\n" + body


# Message format (gpt4_interface.py): an image goes as a 400x300 PNG in base64, labelled image/jpeg.
def image_url(image):
    with Image.fromarray(image) as img:
        buf = io.BytesIO()
        img.resize((400, 300), Image.Resampling.LANCZOS).save(buf, format="PNG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def message(role, text=None, images=()):
    content = [{"type": "text", "text": text}] if text is not None else []
    return {"role": role, "content": content + [{"type": "image_url", "image_url": {"url": image_url(i)}} for i in images]}


def load(path):
    with Image.open(path) as img:
        return np.array(img)


def start_messages(system_text, icl=None):
    """What every episode starts with: [1] and the in-context examples (a text message, then an image message).
    icl: the examples in the order sent (default ICL)."""
    msgs = [message("developer", system_text)]
    for action, path in icl or ICL:
        msgs += [message("user", f"If you see the following, you should command {action}"),
                 message("user", images=[load(os.path.join(HERE, path))])]
    return msgs


def rebuild_chat(chat_path, log_paths=None):
    """Rebuild a logged episode from its saved images and replies: the messages the original sent, as in chat_<T>.json.
    log_paths: the episode's log_<t>.json files in time order (default: all in the chat's folder), i.e. the order query
    (asked once, its answer assumed correct), then the steps.
    [1] and [2] come from the committed config whose system message is the logged one. If there is none (the run used
    older or uncommitted prompt text), they are copied from the log. Everything else is always generated.
    Returns (messages, scenario or None), each message as (message, its image files, the model's logged reply or None)."""
    with open(chat_path) as f:
        chat = json.load(f)
    logged_system = chat[0]["content"][0]["text"]
    found = [(name, cfg) for name, sc in SCENARIOS.items() for cfg in sc["configs"]
             if system_message(cfg["sysaddon"], sc["color_clip"], sc["num_clips"]) == logged_system]
    name, cfg = found[0] if found else (None, None)
    sc = SCENARIOS.get(name)
    folder = os.path.dirname(chat_path)
    if log_paths is None:
        log_paths = sorted(glob.glob(os.path.join(folder, "log_*.json")), key=lambda p: float(os.path.basename(p)[4:-5]))
    icl = ICL if chat[1]["content"][0]["text"].endswith(ICL[0][0]) else ICL[::-1]     # the order sent in this run
    # image files of the start messages: none for the developer message, then each example's text and image message
    icl_files = [[]] + [files for _, p in icl for files in ([], [p])]
    system = system_message(cfg["sysaddon"], sc["color_clip"], sc["num_clips"]) if cfg else logged_system
    msgs = [(m, files, None) for m, files in zip(start_messages(system, icl), icl_files)]
    last_action, target, first, trial = "", "", True, None
    for path in log_paths:
        t = os.path.basename(path)[4:-5]
        with open(path) as f:
            log = json.load(f)
        files = [f"image_{t}.png"]
        if not log["Text"].startswith(("You can now", "Remind that")):       # [2]
            text = order_prompt(cfg["orderaddon"], sc["color_clip"], sc["num_clips"]) if cfg else log["Text"]
            target, first = log["Result"][2], True
        else:                                                                 # [3]
            if not first and trial is None:   # the env does not reset the count between episodes: read where it starts
                shown = re.search(r"You have prompted insert for (\d+) times", log["Text"])
                trial = int(shown.group(1)) if shown else 0
            text = step_prompt(last_action, target, trial, first)
            files.append(f"zoomin_{t}.png")
            last_action, target, first = log["Result"][1], log["Result"][2], False
            if last_action in ("pull", "flatten"):
                trial = 0
            elif last_action == "insert" and trial is not None:
                trial += 1
        query = message("user", text, [load(os.path.join(folder, f)) for f in files])
        msgs.append((query, files, log["Response"]))
        while len(msgs) < len(chat) and chat[len(msgs)]["role"] == "assistant":   # empty reply: the query is re-sent
            msgs[-1] = (query, files, None)
            msgs += [(chat[len(msgs)], [], None), (query, files, log["Response"])]
    return msgs, name


def chat_json(msgs):
    """The JSON text of chat_<T>.json: the original saves the history with json.dump(history, f, indent=4)."""
    return json.dumps([m for m, _, _ in msgs], indent=4)


def example(folder):
    """A logged episode as a concrete example to read: the messages rebuild_chat() gives, each image shown as its file
    name and each query followed by the model's reply as logged (the order reply is logged lowercased, without '*' and
    ':', with 'color' replaced by 'location'). The original adds no reply to the conversation, except an empty one.
    Returns (dict, whether the rebuilt chat is byte-identical to the logged chat_<T>.json), or (None, None) if the
    folder has no chat_<T>.json."""
    chats = glob.glob(os.path.join(folder, "chat_*.json"))
    if not chats:
        return None, None
    chat_path = chats[0]
    msgs, name = rebuild_chat(chat_path)
    with open(chat_path, "rb") as f:
        same = f.read() == chat_json(msgs).encode()
    turns = []
    for i, (m, files, reply) in enumerate(msgs):
        if m["role"] == "assistant":
            turns.append({"part": "empty reply (the query is sent again)", "role": "assistant", "text": ""})
            continue
        text = next((c["text"] for c in m["content"] if c["type"] == "text"), "")
        turn = {"part": "[1] system message" if i == 0 else "[1] in-context example" if i <= 2 * len(ICL) else
                "[3] step prompt" if text.startswith(("You can now", "Remind that")) else "[2] order verification prompt",
                "role": m["role"], "text": text, "images": files}
        if reply is not None:
            turn["model_reply"] = reply
        turns.append(turn)
    ex = {"scenario": name, "chat": os.path.relpath(chat_path, HERE)}
    if name is None:
        ex["note"] = ("No committed config produces this episode's system message (the run used older or uncommitted "
                      "prompt text): [1] the system message and [2] the order verification prompt are copied from the "
                      "log, everything else is generated.")
    return {**ex, "messages": turns}, same


def scenario_json(name):
    sc = SCENARIOS[name]
    return {
        "scenario": name, "committed_config_script": "LearningBaseline/rl_rope/" + sc["script"],
        "num_clips": sc["num_clips"], "color_clip": sc["color_clip"],
        "request": REQUEST,
        "configs": [{"config": i, "instruction": cfg["label"],
                     "system_message": system_message(cfg["sysaddon"], sc["color_clip"], sc["num_clips"]),
                     "order_verification_prompt": order_prompt(cfg["orderaddon"], sc["color_clip"], sc["num_clips"])}
                    for i, cfg in enumerate(sc["configs"])],
        "in_context_examples": [{"text": f"If you see the following, you should command {a}", "image": p} for a, p in ICL],
        "step_prompt_first": step_prompt("", "<target>", first_step=True),
        "step_prompt_next": step_prompt("<last action>", "<target>", "<insert_trial>"),
        "placeholders": {"<target>": "target clip word of the previous reply (first step: clip 1 of the order reply)",
                         "<last action>": "action word of the previous reply",
                         "<insert_trial>": "env insert counter; the sentence 'You have prompted insert for ... times.' "
                                           "is left out when it is 0"},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", nargs=2, metavar=("SCENARIO", "CONFIG"))
    ap.add_argument("--example", nargs="+", metavar="DIR")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"))
    args = ap.parse_args()
    if args.show:
        d = scenario_json(args.show[0])
        c = d["configs"][int(args.show[1])]
        for title, text in [("[1] SYSTEM MESSAGE", c["system_message"]),
                            ("[1] IN-CONTEXT EXAMPLES (each + image)", "\n".join(e["text"] for e in d["in_context_examples"])),
                            ("[2] ORDER VERIFICATION PROMPT (+ full image)", c["order_verification_prompt"]),
                            ("[3] FIRST STEP PROMPT (+ full image + zoom-in)", d["step_prompt_first"]),
                            ("[3] LATER STEP PROMPT (+ full image + zoom-in)", d["step_prompt_next"])]:
            print(f"===== {title}\n{text}\n")
    elif args.example:
        os.makedirs(os.path.join(args.out, "examples"), exist_ok=True)
        for folder in args.example:
            ex, same = example(folder)
            if ex is None:
                print(f"{folder}: skipped, no chat_*.json in it")
                continue
            if "note" in ex:
                print(f"NOTE: {ex['note']}\n")
            for m in ex["messages"]:
                images = f"  + {', '.join(m['images'])}" if m.get("images") else ""
                print(f"===== {m['part'].upper()} ({m['role']}){images}\n{m['text']}\n")
                if "model_reply" in m:
                    print(f"----- MODEL REPLY (from the log; not added to the conversation)\n{m['model_reply']}\n")
            out = os.path.join(args.out, "examples", os.path.basename(os.path.normpath(folder)) + ".json")
            with open(out, "w") as f:
                json.dump(ex, f, indent=4)
            print(f"wrote {out}; the rebuilt conversation is {'byte-identical to' if same else 'DIFFERENT from'} "
                  f"{ex['chat']}")
    else:
        os.makedirs(args.out, exist_ok=True)
        for name in SCENARIOS:
            path = os.path.join(args.out, name + ".json")
            with open(path, "w") as f:
                json.dump(scenario_json(name), f, indent=4)
            print("wrote", path)


if __name__ == "__main__":
    main()
