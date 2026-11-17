"""Provider API keys at rest fail closed: never silently stored as reversible base64. No network."""

import base64
import builtins
import os
import unittest
from unittest import mock

from app.providers import _crypto
from app.providers._crypto import decrypt_value, encrypt_value


def _env(**values):
    clean = {k: v for k, v in os.environ.items()
             if k not in ("PROVIDER_ENCRYPTION_KEY", "ALLOW_PLAINTEXT_PROVIDER_KEYS")}
    clean.update(values)
    return mock.patch.dict(os.environ, clean, clear=True)


class ProviderKeyStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        # No cached Fernet or key from another test.
        for name, value in (("_FERNET", None), ("_FERNET_FOR", None)):
            if hasattr(_crypto, name):
                p = mock.patch.object(_crypto, name, value)
                p.start()
                self.addCleanup(p.stop)

    def test_no_key_refuses_to_store_plaintext(self) -> None:
        with _env():
            with self.assertRaises(RuntimeError):
                token = encrypt_value("sk-secret")
                # Old behaviour: the "encrypted" token was the key in base64.
                self.assertNotEqual(base64.urlsafe_b64decode(token), b"sk-secret")

    def test_explicit_dev_opt_in_stores_a_marked_value(self) -> None:
        with _env(ALLOW_PLAINTEXT_PROVIDER_KEYS="1"):
            token = encrypt_value("sk-secret")
            self.assertTrue(token.startswith("plain:"))
            self.assertEqual(decrypt_value(token), "sk-secret")

    def test_key_set_but_cryptography_missing_is_an_error(self) -> None:
        real_import = builtins.__import__

        def no_cryptography(name, *args, **kwargs):
            if name.startswith("cryptography"):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        with _env(PROVIDER_ENCRYPTION_KEY="passphrase"), mock.patch.object(builtins, "__import__", no_cryptography):
            with self.assertRaises(RuntimeError):
                token = encrypt_value("sk-secret")
                self.assertNotEqual(base64.urlsafe_b64decode(token), b"sk-secret")

    def test_encrypted_round_trip(self) -> None:
        with _env(PROVIDER_ENCRYPTION_KEY="passphrase"):
            token = encrypt_value("sk-secret")
            self.assertNotIn("sk-secret", token)
            self.assertEqual(decrypt_value(token), "sk-secret")

    def test_keys_stored_in_the_old_format_stay_readable(self) -> None:
        legacy = base64.urlsafe_b64encode(b"sk-old").decode()
        with _env():
            self.assertEqual(decrypt_value(legacy), "sk-old")
        with _env(PROVIDER_ENCRYPTION_KEY="passphrase"):
            # Setting a key later must not lock out keys saved before it.
            self.assertEqual(decrypt_value(legacy), "sk-old")


if __name__ == "__main__":
    unittest.main()
