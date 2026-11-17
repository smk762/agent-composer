"""Provider-key decoding never hands back an undecodable stored value as the key."""

import os
import unittest
from unittest import mock

from app.providers._crypto import decrypt_value


class UndecodableStoredKeyTests(unittest.TestCase):
    def test_undecodable_legacy_value_is_none_not_the_raw_value(self) -> None:
        with mock.patch.dict(os.environ, {"PROVIDER_ENCRYPTION_KEY": ""}):
            self.assertIsNone(decrypt_value("not*base64!"))

    def test_undecodable_plain_value_is_none(self) -> None:
        with mock.patch.dict(os.environ, {"PROVIDER_ENCRYPTION_KEY": ""}):
            self.assertIsNone(decrypt_value("plain:abc"))

    def test_value_decoding_to_nothing_is_none(self) -> None:
        with mock.patch.dict(os.environ, {"PROVIDER_ENCRYPTION_KEY": ""}):
            self.assertIsNone(decrypt_value("plain:%%%"))

    def test_fernet_token_without_a_key_is_none(self) -> None:
        with mock.patch.dict(os.environ, {"PROVIDER_ENCRYPTION_KEY": ""}):
            self.assertIsNone(decrypt_value("gAAAAABexampletoken=="))


if __name__ == "__main__":
    unittest.main()
