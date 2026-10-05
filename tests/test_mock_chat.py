import unittest

from sugumi.inference import SugumiEngine


class SugumiEngineTests(unittest.TestCase):
    def test_chat_echoes_latest_user_message_as_mock(self) -> None:
        engine = SugumiEngine()

        reply = engine.chat(
            [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Previous reply"},
                {"role": "user", "content": "Latest"},
            ]
        )

        self.assertEqual(reply, "[Mock response] You said: Latest")

    def test_chat_handles_missing_user_message(self) -> None:
        engine = SugumiEngine()

        self.assertEqual(
            engine.chat([{"role": "system", "content": "Instructions"}]),
            "[Mock response] Please provide a non-empty user message.",
        )

    def test_real_inference_is_not_silently_mocked(self) -> None:
        with self.assertRaises(NotImplementedError):
            SugumiEngine(mock=False)


if __name__ == "__main__":
    unittest.main()
