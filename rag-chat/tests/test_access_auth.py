"""Cloudflare Access identity comes only from a verified Access JWT. No network."""

import time
import unittest
from unittest.mock import patch

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app import auth
from app.db import get_db

TEAM = "team.cloudflareaccess.com"
AUD = "aud-tag-123"
_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class _SigningKey:
    key = _KEY.public_key()


class _FakeJwks:
    def get_signing_key_from_jwt(self, token):
        return _SigningKey()


def _token(key=_KEY, **overrides) -> str:
    now = int(time.time())
    claims = {"email": "alice@example.com", "aud": [AUD], "iss": f"https://{TEAM}",
              "iat": now, "exp": now + 300}
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "k1"})


class AccessAuthTest(unittest.TestCase):
    def setUp(self) -> None:
        async def no_db():
            yield None

        app = FastAPI()

        @app.get("/me")
        async def me(ctx=Depends(auth.require_user)):
            return {"user": ctx.user_id, "via": ctx.via}

        app.dependency_overrides[get_db] = no_db
        self.client = TestClient(app)
        for name, value in (("DEV_AUTH_BYPASS", False), ("CF_ACCESS_TEAM_DOMAIN", TEAM),
                            ("CF_ACCESS_AUD", AUD)):
            p = patch.object(auth, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(auth, "_jwks_client", lambda team: _FakeJwks())
        p.start()
        self.addCleanup(p.stop)

    def _get(self, **headers):
        return self.client.get("/me", headers=headers)

    def test_email_header_alone_is_rejected(self) -> None:
        r = self._get(**{"Cf-Access-Authenticated-User-Email": "victim@example.com"})
        self.assertEqual(r.status_code, 401)

    def test_valid_token_sets_user_from_claim_not_header(self) -> None:
        r = self._get(**{"Cf-Access-Jwt-Assertion": _token(),
                         "Cf-Access-Authenticated-User-Email": "victim@example.com"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"user": "alice@example.com", "via": "access"})

    def test_cookie_token_is_accepted(self) -> None:
        self.client.cookies.set("CF_Authorization", _token())
        self.assertEqual(self._get().json()["user"], "alice@example.com")

    def test_bad_tokens_are_rejected(self) -> None:
        bad = {
            "wrong signer": _token(key=_OTHER_KEY),
            "wrong aud": _token(aud=["other"]),
            "wrong iss": _token(iss="https://evil.cloudflareaccess.com"),
            "expired": _token(exp=int(time.time()) - 3600),
            "no email": _token(email=None),
            "garbage": "not.a.jwt",
        }
        for label, token in bad.items():
            with self.subTest(label):
                self.assertEqual(self._get(**{"Cf-Access-Jwt-Assertion": token}).status_code, 401)

    def test_unconfigured_access_refuses_tokens(self) -> None:
        with patch.object(auth, "CF_ACCESS_AUD", ""):
            self.assertEqual(self._get(**{"Cf-Access-Jwt-Assertion": _token()}).status_code, 401)

    def test_no_credentials_is_401(self) -> None:
        self.assertEqual(self._get().status_code, 401)

    def test_dev_bypass_ignores_unverifiable_access_credentials(self) -> None:
        with patch.object(auth, "DEV_AUTH_BYPASS", True), patch.object(auth, "CF_ACCESS_AUD", ""):
            self.client.cookies.set("CF_Authorization", _token())
            r = self._get()
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["via"], "dev")
        with patch.object(auth, "DEV_AUTH_BYPASS", True):
            r = self._get(**{"Cf-Access-Jwt-Assertion": _token(key=_OTHER_KEY)})
            self.assertEqual(r.json()["via"], "dev")
            # A valid token still wins over the dev user.
            r = self._get(**{"Cf-Access-Jwt-Assertion": _token()})
            self.assertEqual(r.json(), {"user": "alice@example.com", "via": "access"})


if __name__ == "__main__":
    unittest.main()
