# VLM prompts of the rope-routing planner

`make_prompts.py` reproduces, byte for byte, the prompts that the VLM high-level planner of our rope-routing system
sends to GPT-5:
1. the **system message**, followed by two in-context examples;
2. the **order verification prompt**, which asks for the order of the clips;
3. the **step prompt**, which asks for the next skill (Insert, Pull or Flatten) and its parameter (the target clip).

Needs Python 3 with `numpy` and `Pillow` ≥ 9.1. Nothing here calls an API.

| path | what |
|---|---|
| `make_prompts.py` | Builds the three prompts and the chat messages; `--show` and `--example`. |
| `scenarios.py` | The instruction strings of the 4 scenarios × 10 configs. |
| `prompts/` | The prompt text files of the planner. |
| `sample_images/icl/` | The two in-context example images. |
| `sample_images/<sample>/` | One logged episode per setting (`implicit`, `fix_spatial`, `fix_color`, `4_5_clip`): the top-down images, the zoom-ins, one log per query, and the conversation as sent (`chat_<T>.json`). |
| `outputs/` | Written by `make_prompts.py`. |

## Use

```bash
python3 make_prompts.py                                    # outputs/<scenario>.json: the prompts of every config
python3 make_prompts.py --show spatial 2                   # print the prompts of one config, with placeholders
python3 make_prompts.py --example sample_images/fix_color  # one logged episode with its concrete prompts
```

`--example` prints the episode as it was sent: every message, its images by file name, and after each query the
model's reply from the log. It writes the same to `outputs/examples/fix_color.json` and checks that the rebuilt
conversation, images included, is byte-identical to the logged `chat_<T>.json`. It accepts several folders, e.g.
`sample_images/*`.

## Sample episodes

| sample | instruction in the prompt | clips | queries | scenario in `scenarios.py` |
|---|---|---|---|---|
| `implicit` | "in an natural order" (no order given) | 3 white | order + 6 steps | `implicit` |
| `fix_spatial` | "from right, to top, then left" | 3 white | order + 9 steps | `spatial` |
| `fix_color` | "from red, to green, then blue" | 3 coloured | order + 9 steps | `color` |
| `4_5_clip` | "from red, to green, to blue, to yellow, then pink" | 5 coloured | order + 10 steps | none (5 clips) |

All four are rebuilt byte for byte. `fix_spatial` and `fix_color` are generated entirely from `scenarios.py` and
`prompts/`. The other two were recorded with slightly different prompt text:
- `implicit`: an earlier `dev_default_prompt_flat.txt`, whose last line ends `left/center/right/top/bottom` instead of
  `left/center/right/top/bottom/top-right/bottom-left`;
- `4_5_clip`: a 5-clip config, and an order prompt whose "Exit direction" line adds "Exit direction must be
  perpendicular to the long axis of the current clip".

For these two, `--example` copies the system message and the order prompt from the log and says so in a `note`; the
in-context examples, the step prompts and all images are still generated.

## The three prompts

A conversation is a JSON list of OpenAI chat messages; the request uses `gpt-5-2025-08-07` with `reasoning_effort`
`low`. Only the queries are added to the list, not the model's replies. The exception is an empty reply, which is kept
as an empty assistant message before the query is sent again. Images are sent as 400×300 PNG in base64, labelled
`data:image/jpeg`.

1. **System message** (role `developer`).
   - Text: the config's `sysaddon` + `prompts/dev_default_prompt_flat.txt`. In the colour scenarios, the last two lines
     are replaced by the colour answer format.
   - Then the two in-context examples, each a text message ("If you see the following, you should command pull", then
     "... insert") and an image message. Their order follows `os.listdir()` on the machine that ran the planner, so some
     logs send insert first; `--example` reads the order from the log.
2. **Order verification prompt** (role `user`, with the full top-down image).
   - Text: the config's `orderaddon` + `prompts/order_prompt.txt` (colour scenarios: `order_prompt_color.txt`).
   - Asked once per episode, and the answer is assumed correct. Its first clip is the target of the first step prompt.
3. **Step prompt** (role `user`, with the full image and a zoom-in at the target clip).
   - First step: "You can now predict action for the image. The zoomed-in view is centered at the `<clip 1>` clip.",
     a newline, then `prompts/use_default_prompt_flat.txt` without its first sentence.
   - Later steps: "Remind that the last action is `<action>` and the target clip is `<clip>` clip.", then "You have
     prompted insert for `<n>` times." (only if n > 0), then " The zoomed-in view is centered at the `<clip>` clip.", a
     newline, and the whole file. `<n>` counts the Insert commands since the last Pull or Flatten.
   - The last three words of the reply are yes/no (task done), the skill (Insert / Pull / Flatten) and its parameter
     (the target clip). The skill and the clip fill `<action>` and `<clip>` of the next step prompt.

**`outputs/<scenario>.json`** holds, per config, `system_message` [1] and `order_verification_prompt` [2]. Shared by
all configs: `in_context_examples`, `step_prompt_first` and `step_prompt_next` [3] with the placeholders `<target>`,
`<last action>` and `<insert_trial>`, and `request` (model and reasoning effort).

## Scenarios

| scenario | config script of the planner code | configs | clips | colour |
|---|---|---|---|---|
| `implicit` | `rl_eval_data_gen_implicit.py` | 8 "natural order", 2 "V shape" | 3 | no |
| `spatial` | `routing_spatial_cfg.py` | "from top to bottom", "from right, to center, then left", ... | 3 | no |
| `color` | `routing_color_cfg.py` | red → green → blue | 3 | yes |
| `4clip_color` | `routing_4clip_color_cfg.py` | red → green → blue → yellow | 4 | yes |

Many configs share the same prompt text; they differ in the scene.

## Note

The insert counter is not reset between episodes, so "You have prompted insert for `<n>` times" can count inserts of
the previous episode. `--example` reads where the count starts from the log.
