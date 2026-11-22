"""SingGuard's default policy, rule lookup and thinking modes. The checks run on argus-retina."""

import unittest

from app.guard_policy import (
    DEFAULT_POLICY,
    THINKING_MODES,
    normalize_thinking,
    policy_rules,
    risk_categories_block,
    rule_text,
)


class PageTests(unittest.TestCase):
    def test_guard_page_explains_the_rule(self) -> None:
        from pathlib import Path

        ui = Path(__file__).resolve().parents[1] / "app" / "routers" / "ui.py"
        html = ui.read_text(encoding="utf-8")
        self.assertIn("SingGuard Safety Tester", html)
        self.assertIn('id="guardPolicy"', html)
        self.assertIn('id="whyDisplay"', html)
        self.assertIn("Explain why (fast-slow)", html)
        self.assertIn("Reason first (slow)", html)
        self.assertIn("Fast, then slow if flagged (cascade)", html)
        self.assertNotIn("llama-guard3", html)


class PolicyTests(unittest.TestCase):
    def test_default_rules_cover_the_template_headings(self) -> None:
        titles = [rule.title for rule in policy_rules(None)]
        self.assertEqual(
            titles,
            [
                "A. Sexual Content Risk",
                "B. Real-World Crimes & Public Safety",
                "C. Unethical Behavior",
                "D. Cybersecurity & Information Manipulation",
                "E. Agent Safety",
                "F. Politically Sensitive Content",
                "G. Animal Abuse",
                "Safe",
            ],
        )

    def test_policy_header_is_not_wrapped_twice(self) -> None:
        block = risk_categories_block(DEFAULT_POLICY)
        self.assertEqual(block, DEFAULT_POLICY)
        self.assertEqual(block.count("## Risk Categories"), 1)

    def test_custom_policy_replaces_the_default_rules(self) -> None:
        policy = "### B. Real-World Crimes\n  - weapons or violent crime\n\n### Safe\n  - anything else"
        self.assertEqual([r.title for r in policy_rules(policy)], ["B. Real-World Crimes", "Safe"])
        self.assertEqual(risk_categories_block(policy).count("## Risk Categories"), 1)

    def test_thinking_modes_match_cortex(self) -> None:
        self.assertEqual(THINKING_MODES, ("fast", "fast-slow", "slow", "cascade"))
        self.assertEqual(normalize_thinking("SLOW"), "slow")
        self.assertEqual(normalize_thinking("cascade"), "cascade")
        self.assertEqual(normalize_thinking(""), "fast-slow")
        with self.assertRaises(ValueError):
            normalize_thinking("hybrid")
        from app.config import GUARD_THINKING

        self.assertIn(GUARD_THINKING, THINKING_MODES)


class RuleTextTests(unittest.TestCase):
    def test_a_default_rule_has_its_policy_text(self) -> None:
        text = rule_text("B. Real-World Crimes & Public Safety", None)
        self.assertIn("violent crime, weapons", text)
        self.assertEqual(rule_text("Safe", ""), "- Content that does not match any risk category.")

    def test_a_custom_policy_rule(self) -> None:
        policy = "### B. Weapons\n  - guns and knives\n\n### Safe\n  - anything else"
        self.assertEqual(rule_text("B. Weapons", policy), "- guns and knives")
        self.assertEqual(rule_text("B. Real-World Crimes & Public Safety", policy), "")

    def test_a_shortened_title_matches_one_heading_only(self) -> None:
        self.assertIn("violent crime", rule_text("B. Real-World Crimes", None))
        policy = "### B. Weapons\n  - guns\n\n### B. Weapons trade\n  - selling guns"
        self.assertEqual(rule_text("B. Weap", policy), "")  # cut mid-word
        self.assertEqual(rule_text("B.", policy), "")  # ambiguous

    def test_no_rule_has_no_text(self) -> None:
        self.assertEqual(rule_text("", None), "")
        self.assertEqual(rule_text("Z. Unknown", None), "")


if __name__ == "__main__":
    unittest.main()
