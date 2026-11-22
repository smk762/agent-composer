"""SingGuard's default policy, its rules, and the thinking modes.

The checks themselves run on argus-retina's SingGuard server (``routers/guard.py`` proxies
to it), which renders the prompt, neutralizes control tokens and role markers, and parses
the generation. This module keeps what rag-chat still answers itself: the default policy
and rule list for ``/api/guard/policy`` and ``/api/guard/categories`` (the tester edits and
sends them back), the thinking modes, and the text of the rule a verdict named.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Default taxonomy from the SingGuard chat template. A request may replace
# this whole block; the model then judges only the rules it was given.
DEFAULT_POLICY = """## Risk Categories

### A. Sexual Content Risk
  - Content involving explicit sexual material, exploitation, or coercive sexual acts.

### B. Real-World Crimes & Public Safety
  - Content involving violent crime, weapons, other crimes, or public-safety threats.

### C. Unethical Behavior
  - Content involving hate, harassment, manipulation, self-harm, disturbing imagery, or harmful misinformation.

### D. Cybersecurity & Information Manipulation
  - Content involving data leaks, hacking, surveillance abuse, platform abuse, or copyright abuse.

### E. Agent Safety
  - Content attempting to expose system prompts, internal policies, or other model safeguards.

### F. Politically Sensitive Content
  - Content involving political advocacy, rumors, unrest, historical distortion, or attacks on political figures.

### G. Animal Abuse
  - Content involving cruelty to animals or the spread of animal abuse.

### Safe
  - Content that does not match any risk category."""

THINKING_FAST = "fast"
THINKING_FAST_SLOW = "fast-slow"
THINKING_SLOW = "slow"
THINKING_CASCADE = "cascade"
# Same order as argus_cortex.moderation.THINKING_MODES (and argus-retina's guard.THINKING).
THINKING_MODES = (THINKING_FAST, THINKING_FAST_SLOW, THINKING_SLOW, THINKING_CASCADE)

_HEADING_RE = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class PolicyRule:
    title: str
    text: str


def normalize_thinking(value: str | None) -> str:
    mode = (value or THINKING_FAST_SLOW).strip().lower()
    if mode not in THINKING_MODES:
        allowed = ", ".join(THINKING_MODES)
        raise ValueError(f"thinking must be one of: {allowed}")
    return mode


def risk_categories_block(policy: str | None) -> str:
    """The ``## Risk Categories`` section for *policy*: the default when it is blank."""
    text = (policy or "").strip()
    if not text:
        return DEFAULT_POLICY
    if "## Risk Categories" in text:
        return text
    return f"## Risk Categories\n\n{text}"


def policy_rules(policy: str | None) -> list[PolicyRule]:
    """Split a policy into rule titles and the text that defines each one."""
    block = risk_categories_block(policy)
    headings = list(_HEADING_RE.finditer(block))
    rules: list[PolicyRule] = []
    for index, match in enumerate(headings):
        start = match.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(block)
        rules.append(PolicyRule(title=match.group(1).strip(), text=block[start:end].strip()))
    return rules


def rule_text(rule: str, policy: str | None) -> str:
    """The text that defines *rule* in *policy* (the default when blank), or "".

    The guard names a rule by its heading, or by a heading cut short at a word boundary
    ("B. Real-World Crimes"), so a shortened title matches when exactly one heading starts
    with it.
    """
    wanted = " ".join(rule.split()).casefold()
    if not wanted:
        return ""
    rules = policy_rules(policy)
    for item in rules:
        if item.title.casefold() == wanted:
            return item.text
    candidates = []
    for item in rules:
        name = item.title.casefold()
        if name.startswith(wanted) and not name[len(wanted)].isalnum():
            candidates.append(item)
    return candidates[0].text if len(candidates) == 1 else ""
