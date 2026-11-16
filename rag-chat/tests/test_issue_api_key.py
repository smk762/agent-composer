"""A key issued by issue_api_key (the CLI's and POST /api-keys' path) authenticates as its user."""

import unittest

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.auth import _api_key_from_token, issue_api_key
from app.models.orm import ApiKey


class IssueApiKeyTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with self.engine.begin() as conn:
            await conn.run_sync(ApiKey.__table__.create)
        self.Session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    async def asyncTearDown(self) -> None:
        await self.engine.dispose()

    async def test_issued_key_resolves_to_its_user(self) -> None:
        async with self.Session() as db:
            rec, secret = await issue_api_key(db, "argus-iris", "iris broker")
            self.assertNotIn(secret, rec.hashed_key)
            self.assertTrue(secret.startswith(rec.prefix + "."))
            found = await _api_key_from_token(db, secret)
            self.assertIsNotNone(found)
            self.assertEqual((found.user_id, found.name), ("argus-iris", "iris broker"))
            self.assertIsNone(await _api_key_from_token(db, secret + "x"))


if __name__ == "__main__":
    unittest.main()
