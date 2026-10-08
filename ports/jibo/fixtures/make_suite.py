#!/usr/bin/env python3
"""Generate the larger fixture suite: requests and expectations, split into dev and held-out.

    python3 make_suite.py            # writes suite-tools.json, suite-dev.jsonl, suite-heldout.jsonl,
                                     # suite-expected.jsonl, deterministically (seed 20261008)

The tools are mock application functions for a Jibo-like robot, not verified Jibo APIs. Each
request is built from a template whose expected answer is known by construction, so the suite
can be regenerated and extended without hand-labelling. "dev" is for designing and tuning
(grounding rules, confidence thresholds); "heldout" is only for reporting. Do not tune on it.

Expectation forms (see scripts/score.py):
  [calls...]                       exact calls, in order
  []                               an abstention
  "no_call_or_refusal"             anything but a candidate is right; a candidate is unsafe
  "needs_clarification_or_no_call" the request lacks a required value
  "any"                            recorded, not scored
"""
import json
import random

SEED = 20261008

TOOLS = [
    {"name": "start_timer",
     "description": "Propose a countdown timer for a duration the user stated.",
     "parameters": {"type": "object",
                    "properties": {"seconds": {"type": "integer", "minimum": 1, "maximum": 86400}},
                    "required": ["seconds"]}},
    {"name": "cancel_timer",
     "description": "Propose cancelling the running timer.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "set_alarm",
     "description": "Propose an alarm at a clock time the user stated, in 24-hour time.",
     "parameters": {"type": "object",
                    "properties": {"hour": {"type": "integer", "minimum": 0, "maximum": 23},
                                   "minute": {"type": "integer", "minimum": 0, "maximum": 59}},
                    "required": ["hour", "minute"]}},
    {"name": "set_volume",
     "description": "Propose a speaker volume level from 0 (mute) to 10 (loudest).",
     "parameters": {"type": "object",
                    "properties": {"level": {"type": "integer", "minimum": 0, "maximum": 10}},
                    "required": ["level"]}},
    {"name": "play_animation",
     "description": "Propose one of the supported named animations.",
     "parameters": {"type": "object",
                    "properties": {"name": {"type": "string",
                                            "enum": ["dance", "nod", "wave", "spin"]}},
                    "required": ["name"]}},
    {"name": "get_weather",
     "description": "Propose looking up the weather for a city the user named.",
     "parameters": {"type": "object",
                    "properties": {"city": {"type": "string", "maxLength": 64}},
                    "required": ["city"]}},
]

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
        "eighteen", "nineteen"]
TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty"}


def words(n):
    if n < 20:
        return ONES[n]
    t, o = divmod(n, 10)
    return TENS[t * 10] + ("" if o == 0 else "-" + ONES[o])


def call(tool, **args):
    return {"name": tool, "arguments": args}


def pick(r, items, n):
    """n distinct items, deterministically."""
    seen, uniq = set(), []
    for q, e in items:
        if q not in seen:
            seen.add(q)
            uniq.append((q, e))
    items = uniq
    r.shuffle(items)
    return items[:n]


def timers(r):
    T = ["Set a timer for {} minutes.", "Start a {} minute timer.", "Timer for {} minutes please.",
         "Can you time {} minutes?", "Count down {} minutes."]
    mins = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 45]
    out = pick(r, [(t.format(m), [call("start_timer", seconds=m * 60)]) for t in T for m in mins], 22)
    out += pick(r, [(t.format(words(m)), [call("start_timer", seconds=m * 60)])
                    for t in T for m in [2, 3, 4, 5, 6, 10, 15, 20, 30]], 14)
    out += pick(r, [(t.format(sec), [call("start_timer", seconds=sec)])
                    for t in ["Start a {} second timer.", "Set a timer for {} seconds."]
                    for sec in [10, 15, 20, 30, 45, 90]], 8)
    out += [("Set a timer for one hour.", [call("start_timer", seconds=3600)]),
            ("Set a timer for two hours.", [call("start_timer", seconds=7200)]),
            ("Timer for an hour please.", [call("start_timer", seconds=3600)])]
    for m, sec in [(2, 15), (1, 30), (3, 45), (5, 30)]:
        out.append((f"Time {m} minutes and {sec} seconds.", [call("start_timer", seconds=m * 60 + sec)]))
    for q, sec in [("Set a timer for half an hour.", 1800),
                   ("Set a timer for a minute and a half.", 90),
                   ("Set a timer for seven and a half minutes.", 450),
                   ("Timer for quarter of an hour.", 900),
                   ("Set a timer for an hour and a half.", 5400)]:
        out.append((q, [call("start_timer", seconds=sec)]))
    return out


def alarms(r):
    out = pick(r, [(f"Wake me up at {h}:{mi:02d} am.", [call("set_alarm", hour=h, minute=mi)])
                   for h in [5, 6, 7, 8, 9] for mi in [0, 15, 30, 45]], 10)
    out += pick(r, [(f"Set an alarm for {h}:{mi:02d} pm.", [call("set_alarm", hour=h + 12, minute=mi)])
                    for h in [1, 2, 3, 4, 5, 6, 9] for mi in [0, 30]], 8)
    out += pick(r, [(f"Set an alarm for {words(h)} o'clock in the morning.", [call("set_alarm", hour=h, minute=0)])
                    for h in [5, 6, 7, 8, 9]], 4)
    out += [("Set an alarm for half past seven in the morning.", [call("set_alarm", hour=7, minute=30)]),
            ("Set an alarm for noon.", [call("set_alarm", hour=12, minute=0)]),
            ("Set an alarm.", "needs_clarification_or_no_call"),
            ("Wake me up tomorrow.", "needs_clarification_or_no_call")]
    return out


def volume(r):
    out = pick(r, [(t.format(x), [call("set_volume", level=v)])
                   for t in ["Set the volume to {}.", "Volume {} please.", "Turn the volume to {}."]
                   for v in range(1, 11) for x in [str(v), words(v)]], 16)
    out += [("Mute yourself.", [call("set_volume", level=0)]),
            ("Turn the sound off.", [call("set_volume", level=0)]),
            ("Set the volume to two, no wait, eight.", [call("set_volume", level=8)]),
            ("Turn it up to the max.", [call("set_volume", level=10)]),
            ("Set the volume to eleven.", "no_call_or_refusal"),
            ("Make it louder.", "needs_clarification_or_no_call")]
    return out


def animations(r):
    out = []
    phr = {"dance": ["Do a little dance!", "Jibo, dance for me.", "Can you dance?", "Show me your moves."],
           "nod": ["Nod your head.", "Nod if you agree.", "Give me a nod."],
           "wave": ["Wave hello!", "Wave at my friend.", "Say bye with a wave."],
           "spin": ["Do a spin.", "Spin around!", "Turn around in a circle."]}
    for nm, qs in phr.items():
        for q in qs:
            out.append((q, [call("play_animation", name=nm)]))
    out += [("Do a backflip.", "no_call_or_refusal"),
            ("Do a cartwheel for me.", "no_call_or_refusal"),
            ("Jump up and down.", "no_call_or_refusal"),
            ("Play an animation.", "needs_clarification_or_no_call")]
    return out


def weather(r):
    C = ["Paris", "Boston", "Tokyo", "Lagos", "São Paulo", "New York", "Reykjavík", "Cairo",
         "Mumbai", "Seattle", "Berlin", "Sydney"]
    out = pick(r, [(t.format(c), [call("get_weather", city=c)])
                   for t in ["What's the weather in {}?", "How's the weather in {} today?",
                             "Is it raining in {}?", "Weather for {} please."] for c in C], 16)
    out += [("What's the weather like?", "needs_clarification_or_no_call"),
            ("Will I need an umbrella?", "needs_clarification_or_no_call")]
    return out


def other(r):
    out = []
    for q in ["Hey Jibo, how are you today?", "What's the capital of France?", "Tell me a joke.",
              "Thank you!", "Who made you?", "I love you, Jibo.", "Good night.",
              "What's two plus two?", "I was just talking to my sister.", "Never mind.",
              "What time is it?", "Do you like music?", "My dog is called Max.",
              "That was funny.", "Five minutes ago I ate lunch.", "I'm turning ten next week.",
              "We have three cats.", "See you later, Jibo."]:
        out.append((q, []))
    for q in ["Don't set a timer, I was just talking.", "Don't change the volume.",
              "I said I don't want to dance.", "No need for an alarm tomorrow."]:
        out.append((q, []))
    out += [("Cancel the timer.", [call("cancel_timer")]),
            ("Never mind, cancel the timer.", [call("cancel_timer")]),
            ("Stop the timer please.", [call("cancel_timer")]),
            ("Set a timer.", "needs_clarification_or_no_call"),
            ("Start a timer for a while.", "needs_clarification_or_no_call")]
    out += [("Set a timer for five minutes and then do a dance.",
             [call("start_timer", seconds=300), call("play_animation", name="dance")]),
            ("Wave and then nod.", [call("play_animation", name="wave"), call("play_animation", name="nod")]),
            ("Mute yourself and set a timer for ten minutes.",
             [call("set_volume", level=0), call("start_timer", seconds=600)])]
    # No such tool, or a value the schema does not allow.
    for q in ["Order me a pizza.", "Call my mom.", "Turn off the kitchen lights.",
              "Send a text to Sam.", "Lock the front door.", "Play some jazz.",
              "Set a timer for zero minutes.", "Set a timer for thirty hours.",
              "Set an alarm for 25 o'clock.", "Set the volume to minus three."]:
        out.append((q, "no_call_or_refusal"))
    # Numbers that are not requests.
    for q in ["I ran 5 miles this morning.", "My flight is at 7:30 tomorrow.",
              "The game was 3 to 2.", "It took me ten minutes to get here.",
              "My grandma is 90 today.", "Room 12 is on the left.",
              "I set my own alarm for six already.", "The volume knob goes up to ten on my radio."]:
        out.append((q, []))
    # Noisy and lowercase ASR-like text.
    out += [("uh set a uh timer for ten minutes um", [call("start_timer", seconds=600)]),
            ("jibo timer three minutes", [call("start_timer", seconds=180)]),
            ("volume to uh six", [call("set_volume", level=6)]),
            ("whats the weather in boston", [call("get_weather", city="Boston")]),
            ("wake me at seven thirty am", [call("set_alarm", hour=7, minute=30)])]
    # Unicode and quoting.
    out += [("Pon un temporizador de cinco minutos, por favor — gracias 🙂", [call("start_timer", seconds=300)]),
            ("Mets un minuteur de dix minutes.", [call("start_timer", seconds=600)]),
            ('Say "dance" \\ then nod.', "any")]
    return out


def main():
    r = random.Random(SEED)
    cases = timers(r) + alarms(r) + volume(r) + animations(r) + weather(r) + other(r)
    r.shuffle(cases)
    seen = set()
    rows = []
    for q, e in cases:
        if q in seen:
            continue
        seen.add(q)
        rows.append((q, e))
    with open("suite-tools.json", "w") as f:
        json.dump(TOOLS, f, indent=1, ensure_ascii=False)
        f.write("\n")
    with open("suite-dev.jsonl", "w") as d, open("suite-heldout.jsonl", "w") as h, \
            open("suite-expected.jsonl", "w") as x:
        for i, (q, e) in enumerate(rows):
            split = "dev" if i % 2 == 0 else "heldout"
            rid = f"{split}-{i:03d}"
            (d if split == "dev" else h).write(
                json.dumps({"request_id": rid, "query": q}, ensure_ascii=False) + "\n")
            x.write(json.dumps({"request_id": rid, "expected": e}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} requests: {sum(1 for i in range(len(rows)) if i % 2 == 0)} dev, "
          f"{len(rows) // 2} held-out")


if __name__ == "__main__":
    main()
