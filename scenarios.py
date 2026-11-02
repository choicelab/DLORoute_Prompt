"""The committed scenario configs (git HEAD 945268b), prompt fields only.

The original reads two strings per config from a pickle that the config script writes: `sysaddon` (start of the system
message) and `orderaddon` (start of the order verification prompt). Every string here is the one the committed config
script named in SCENARIOS writes.
"""
ORDER = "Now, we can start doing the high level planning first, we want to insert the rope through several clip"
SYS = "You are a rope routing planner trying to generate the next action to insert the rope through several clip"
WHITE = "The color of the clips are white (This is not relavent to the order of insertion). "


def config(label, order_part, sys_part):
    return {"label": label, "orderaddon": ORDER + order_part, "sysaddon": SYS + sys_part}


def explicit(phrase, clip_colours=WHITE):
    """Spatial and colour configs: the same phrase goes into both strings."""
    return config(phrase.split(".")[0].strip(), phrase, phrase + clip_colours)


# rl_eval_data_gen_implicit.py: scenes 0-4, then the same 5 scenes again with lousy_pull=True; scene 3 is the V shape
NATURAL = config("natural order", " in an natural order. ",
                 " in an natural order. Natural order means that you should avoid sharp turn during the routing process. " + WHITE)
V_SHAPE = config("V shape", " to form a V shape. ", " to form a V shape. " + WHITE)

# routing_spatial_cfg.py: prompt[0..9]
SPATIAL = [
    " from top to bottom. ",
    " from top to bottom. ",
    " from right, to center, then left. To prevent spatial ambiguity, you must use right, center or left to describe the spatial location of clips. ",
    " from right, to top, then left. To prevent spatial ambiguity, you must use right, top or left to describe the spatial location of clips. ",
    " from right, to left, then top. To prevent spatial ambiguity, you must use right, left or top to describe the spatial location of clips. ",
    " from right, to top, then left. To prevent spatial ambiguity, you must use right, top or left to describe the spatial location of clips. ",
    " from top, to left, then right. To prevent spatial ambiguity, you must use top, left or right to describe the spatial location of clips. ",
    " from top, to right, then left. To prevent spatial ambiguity, you must use top, right or left to describe the spatial location of clips. ",
    " from top, to right, then left. To prevent spatial ambiguity, you must use top, right or left to describe the spatial location of clips. ",
    " from right, to left, then top. To prevent spatial ambiguity, you must use right, left or top to describe the spatial location of clips. ",
]

SCENARIOS = {   # num_clips / color_clip: the NUMCLIP / COLOR switches of rl_utils/simulation_env_full.py
    "implicit": {"script": "rl_eval_data_gen_implicit.py", "num_clips": 3, "color_clip": False,
                 "configs": [V_SHAPE if i % 5 == 3 else NATURAL for i in range(10)]},
    "spatial": {"script": "routing_spatial_cfg.py", "num_clips": 3, "color_clip": False,
                "configs": [explicit(p) for p in SPATIAL]},
    "color": {"script": "routing_color_cfg.py", "num_clips": 3, "color_clip": True,
              "configs": [explicit(" from red, to green, then blue. To prevent ambiguity, you must use red, green or blue to describe the clips. ",
                                   "The color of the clips are red, green or blue. ")] * 10},
    "4clip_color": {"script": "routing_4clip_color_cfg.py", "num_clips": 4, "color_clip": True,
                    "configs": [explicit(" from red, to green, to blue, then yellow. To prevent ambiguity, you must use red, green, blue or yellow to describe the clips. ",
                                         "The color of the clips are red, green, blue or yellow. ")] * 10},
}
