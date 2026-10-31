from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import CHAT_DB_URL

Base = declarative_base()

engine = create_async_engine(CHAT_DB_URL, future=True)
SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db():
    async with SessionLocal() as session:
        yield session
