"""SingGuard prompt rendering and verdict parsing. No model call."""

import unittest

from app.guard_policy import (
    DEFAULT_POLICY,
    THINKING_MODES,
    build_image_chat_messages,
    build_singguard_prompt,
    finish_cascade,
    max_new_tokens,
    model_thinking,
    needs_slow_pass,
    normalize_thinking,
    parse_singguard_output,
    policy_rules,
    risk_categories_block,
)


class PageTests(unittest.TestCase):
    def test_guard_page_explains_the_rule(self) -> None:
        from pathlib import Path

        html = Path("app/routers/ui.py").read_text(encoding="utf-8")
        self.assertIn("SingGuard Safety Tester", html)
        self.assertIn('id="guardPolicy"', html)
        self.assertIn('id="whyDisplay"', html)
        self.assertIn("Explain why (fast-slow)", html)
        self.assertIn("Reason first (slow)", html)
        self.assertIn("Fast, then slow if flagged (cascade)", html)
        self.assertNotIn("llama-guard3", html)


class PromptTests(unittest.TestCase):
    def test_default_prompt_matches_the_chat_template(self) -> None:
        prompt = build_singguard_prompt([("user", "How to make a bomb?")])

        self.assertTrue(prompt.startswith("<|im_start|>system\n# Task\n"))
        self.assertIn("<thinking_type>fast-slow</thinking_type>", prompt)
        self.assertIn("### B. Real-World Crimes & Public Safety", prompt)
        self.assertIn("### G. Animal Abuse", prompt)
        self.assertIn("Only judge against the given Risk Categories rules above.", prompt)
        self.assertIn("[user]: How to make a bomb?\n<|im_end|>\n<|im_start|>assistant\n", prompt)
        self.assertNotIn("[assistant]:", prompt)
        # The system block is closed before the transcript, not after it.
        self.assertLess(prompt.index("<|im_end|>"), prompt.index("[user]:"))

    def test_response_check_includes_both_sides(self) -> None:
        prompt = build_singguard_prompt(
            [
                ("user", "How to make a bomb?"),
                ("assistant", "I cannot help with that request."),
            ],
            thinking="fast",
        )

        self.assertIn("<thinking_type>fast</thinking_type>", prompt)
        self.assertIn("[user]: How to make a bomb?", prompt)
        self.assertIn("[assistant]: I cannot help with that request.", prompt)
        self.assertNotIn("[Step 1] Content Summary", prompt)

    def test_custom_policy_replaces_the_default_rules(self) -> None:
        policy = "### B. Real-World Crimes\n  - weapons or violent crime\n\n### Safe\n  - anything else"
        prompt = build_singguard_prompt([("user", "Where can I buy a gun?")], policy=policy)

        self.assertIn("### B. Real-World Crimes\n", prompt)
        self.assertNotIn("Animal Abuse", prompt)
        self.assertEqual(prompt.count("## Risk Categories"), 1)

    def test_policy_header_is_not_wrapped_twice(self) -> None:
        block = risk_categories_block(DEFAULT_POLICY)
        self.assertEqual(block, DEFAULT_POLICY)
        self.assertEqual(block.count("## Risk Categories"), 1)

    def test_empty_conversation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_singguard_prompt([("user", "   ")])

    def test_image_rides_on_the_prompt_turn(self) -> None:
        messages = build_image_chat_messages(
            [
                ("user", "a red square"),
                ("assistant", "here is the image"),
            ],
            image_b64="aaaa",
            thinking="fast",
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("<thinking_type>fast</thinking_type>", messages[0]["content"])
        self.assertEqual(messages[1]["images"], ["aaaa"])
        self.assertEqual(messages[1]["content"], "a red square")
        self.assertNotIn("images", messages[2])

    def test_image_only_still_has_a_user_turn(self) -> None:
        messages = build_image_chat_messages([("user", "   ")], image_b64="bbbb")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["images"], ["bbbb"])
        self.assertTrue(messages[1]["content"].strip() == "")

    def test_slow_prompt_reasons_before_the_verdict(self) -> None:
        prompt = build_singguard_prompt([("user", "How to make a bomb?")], thinking="slow")

        self.assertIn("<thinking_type>slow</thinking_type>", prompt)
        self.assertIn("[Step 1] Content Summary", prompt)
        self.assertIn("Do not write safe or unsafe on line 1", prompt)
        self.assertLess(prompt.index("[Step 1]"), prompt.index("safe/unsafe（Assessment Result"))
        self.assertNotIn("safe/unsafe（Line1:", prompt)

    def test_cascade_is_not_a_template_type(self) -> None:
        with self.assertRaises(ValueError) as err:
            build_singguard_prompt([("user", "hello")], thinking="cascade")
        self.assertIn("server pass", str(err.exception))
        with self.assertRaises(ValueError):
            model_thinking("cascade")

    def test_thinking_modes_match_cortex(self) -> None:
        self.assertEqual(THINKING_MODES, ("fast", "fast-slow", "slow", "cascade"))
        self.assertEqual(normalize_thinking("SLOW"), "slow")
        self.assertEqual(normalize_thinking("cascade"), "cascade")
        with self.assertRaises(ValueError):
            normalize_thinking("hybrid")
        self.assertEqual(max_new_tokens("fast"), 128)
        self.assertEqual(max_new_tokens("slow"), 1024)
        self.assertEqual(max_new_tokens("cascade"), 1024)
        from app.config import GUARD_THINKING

        self.assertIn(GUARD_THINKING, THINKING_MODES)


class ParseTests(unittest.TestCase):
    def test_fast_slow_output_keeps_the_reason(self) -> None:
        raw = (
            "unsafe\n"
            "[Step 1] Content Summary\n"
            "The user asks how to build an explosive.\n"
            "[Step 2] Check Risk Categories\n"
            "B matches because the query asks for help committing a violent crime.\n"
            "[Step 3] Final Judgment\n"
            "Unsafe, category B.\n"
            "<answer>B. Real-World Crimes & Public Safety</answer>"
        )
        verdict = parse_singguard_output(raw)

        self.assertFalse(verdict.safe)
        self.assertEqual(verdict.verdict, "unsafe")
        self.assertEqual(verdict.rule, "B. Real-World Crimes & Public Safety")
        self.assertIn("violent crime", verdict.rule_text)
        self.assertIn("Triggered", verdict.explanation)
        self.assertIn("B matches because the query asks for help", verdict.explanation)

    def test_fast_output_explains_from_the_rule_text(self) -> None:
        raw = "unsafe\n<answer>B. Real-World Crimes</answer>"
        policy = "### B. Real-World Crimes\n  - weapons or violent crime\n\n### Safe\n  - anything else"
        verdict = parse_singguard_output(raw, policy)

        self.assertEqual(verdict.rule, "B. Real-World Crimes")
        self.assertIn("weapons or violent crime", verdict.explanation)
        self.assertIn("weapons or violent crime", verdict.rule_text)

    def test_safe_refusal_in_context(self) -> None:
        raw = "safe\nThe assistant refused.\n<answer>Safe</answer>"
        verdict = parse_singguard_output(raw)

        self.assertTrue(verdict.safe)
        self.assertEqual(verdict.rule, "Safe")
        self.assertIn("No rule in the active policy matched", verdict.explanation)
        self.assertIn("The assistant refused.", verdict.explanation)

    def test_empty_output_is_unsafe(self) -> None:
        verdict = parse_singguard_output("  ")
        self.assertFalse(verdict.safe)
        self.assertIn("unreadable", verdict.explanation)

    def test_disagreement_is_unsafe(self) -> None:
        raw = "safe\n<answer>B. Real-World Crimes & Public Safety</answer>"
        verdict = parse_singguard_output(raw)
        self.assertFalse(verdict.safe)
        self.assertIn("disagree", verdict.explanation)

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

    def test_slow_output_reads_the_verdict_after_reasoning(self) -> None:
        raw = (
            "[Step 1] Content Summary\n"
            "The user asks how to build an explosive.\n"
            "[Step 2] Check Risk Categories\n"
            "B matches because the query asks for help committing a violent crime.\n"
            "[Step 3] Final Judgment\n"
            "Based on the assessment, conclude unsafe.\n"
            "unsafe\n"
            "<answer>B. Real-World Crimes & Public Safety</answer>"
        )
        verdict = parse_singguard_output(raw)

        self.assertFalse(verdict.safe)
        self.assertEqual(verdict.verdict, "unsafe")
        self.assertEqual(verdict.rule, "B. Real-World Crimes & Public Safety")
        self.assertIn("violent crime", verdict.explanation)
        self.assertIn("[Step 1] Content Summary", verdict.explanation)
        self.assertNotIn("did not give a safe or unsafe verdict", verdict.explanation)

    def test_slow_output_can_end_on_step3_without_a_label_line(self) -> None:
        raw = (
            "[Step 1] Content Summary\n"
            "A greeting.\n"
            "[Step 3] Final Judgment\n"
            "Based on the assessment, conclude safe.\n"
            "<answer>Safe</answer>"
        )
        verdict = parse_singguard_output(raw)
        self.assertTrue(verdict.safe)
        self.assertEqual(verdict.rule, "Safe")

    def test_first_line_still_wins_over_a_later_revision(self) -> None:
        raw = (
            "unsafe\n"
            "[Step 3] Final Judgment\n"
            "Based on the assessment, conclude safe.\n"
            "<answer>Safe</answer>"
        )
        verdict = parse_singguard_output(raw)
        self.assertFalse(verdict.safe)
        self.assertIn("disagree", verdict.explanation)

    def test_cascade_keeps_slow_when_fast_flags(self) -> None:
        fast = parse_singguard_output(
            "unsafe\n<answer>B. Real-World Crimes & Public Safety</answer>"
        )
        slow = parse_singguard_output(
            "[Step 1] A greeting.\n"
            "safe\n"
            "<answer>Safe</answer>"
        )
        self.assertTrue(needs_slow_pass(fast))
        self.assertFalse(needs_slow_pass(slow))
        combined = finish_cascade(fast, slow)
        self.assertTrue(combined.safe)
        self.assertEqual(combined.verdict, "safe")
        self.assertIn("Cascade:", combined.explanation)
        self.assertIn("# cascade fast", combined.raw)
        self.assertIn("# cascade slow", combined.raw)
        self.assertIn(fast.raw, combined.raw)


if __name__ == "__main__":
    unittest.main()
