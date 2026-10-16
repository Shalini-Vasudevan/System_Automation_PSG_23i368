"""Requirement 3 - maintenance notes as a second, independent signal.

Reads the free-text technician log and scores it with keyword rules. It never
looks at sensor numbers, so it is independent of the ML model.

Log line format:   YYYY-MM-DD HH:MM | ARM-xx | joint | free text
Lines that don't match the format are ignored.

Scoring: SEVERE keywords (burning smell, grinding, cracks...) = 3 points,
MINOR keywords (squeak, loose, warm, vibration...) = 1 point each (distinct).
  points >= 3 -> SEVERE,  points >= 1 -> MINOR,  else NONE
Negations ("no grinding noise") are ignored; notes older than
NOTE_MAX_AGE_DAYS are ignored.
"""
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from config import NOTE_MAX_AGE_DAYS, NOTE_SEVERE_POINTS, NOTE_MINOR_POINTS

SEVERE_PATTERNS = [
    r"burn\w*\s+smell", r"smell\w*\s+(?:of\s+)?burn\w*", r"\bburnt\b",
    r"\bsmok\w+", r"\bgrind\w*", r"metal\s+(?:shavings|flakes|filings|particles)",
    r"\bcrack\w*", r"\bseiz\w+", r"\bsparks?\b", r"\bknock\w*",
]
MINOR_PATTERNS = [
    r"\bsqueak\w*", r"\bsqueal\w*", r"\brattl\w*", r"\bwhin\w+", r"\bhumm\w*",
    r"\bnois\w+", r"\bvibrat\w+", r"\bwarm\b", r"\bhot\b", r"\boverheat\w*",
    r"\bloose\w*", r"\bbacklash\b", r"\bplay\b", r"\bstiff\w*", r"\bsticky\b",
    r"\bjerk\w*", r"\bleak\w*", r"\bseep\w*", r"\bdiscolou?r\w*",
]
NEGATORS = {"no", "not", "without", "never", "none", "nothing", "isnt", "didnt",
            "doesnt", "cant", "free"}
LINE_RE = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}))?\s*\|\s*([\w-]+)\s*\|\s*([\w-]*)\s*\|\s*(.+?)\s*$")


@dataclass
class NoteResult:
    points: int = 0
    level: str = "NONE"
    evidence: list = field(default_factory=list)


def _negated(words_before):
    return any(w in NEGATORS for w in words_before[-3:])


def score_text(text):
    """Return (points, [evidence]) for a single note."""
    points, evidence = 0, []
    # split into clauses so "no leaks, but grinding noise" negates only "leaks"
    for clause in re.split(r"[.,;:!?]|\bbut\b|\bhowever\b", text.lower()):
        clause = clause.replace("'", "")
        for patterns, pts in ((SEVERE_PATTERNS, NOTE_SEVERE_POINTS), (MINOR_PATTERNS, 1)):
            for pat in patterns:
                m = re.search(pat, clause)
                if not m:
                    continue
                before = re.findall(r"[a-z]+", clause[:m.start()])
                if _negated(before):
                    continue
                points += pts
                evidence.append(m.group(0))
    return points, evidence


def parse_log(path):
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m or line.lstrip().startswith("#"):
                continue
            date, time, arm, joint, text = m.groups()
            ts = datetime.strptime(f"{date} {time or '00:00'}", "%Y-%m-%d %H:%M")
            entries.append({"ts": ts, "arm": arm.upper(), "joint": joint.lower(), "text": text})
    return entries


def score_log(path, now):
    """Return {arm_id: NoteResult} using only recent notes."""
    cutoff = now - timedelta(days=NOTE_MAX_AGE_DAYS)
    totals = defaultdict(NoteResult)
    for e in parse_log(path):
        if e["ts"] < cutoff or e["ts"] > now:
            continue
        pts, ev = score_text(e["text"])
        if pts:
            r = totals[e["arm"]]
            r.points += pts
            r.evidence.extend(f"{e['joint'] or 'general'}: {w}" for w in ev)
    for r in totals.values():
        r.level = ("SEVERE" if r.points >= NOTE_SEVERE_POINTS
                   else "MINOR" if r.points >= NOTE_MINOR_POINTS else "NONE")
    return totals
