from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config.settings import settings

engine = create_async_engine(
    settings.postgres_dsn,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=1800,
)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncSession:
    """Yield a database session, closing it on exit (FastAPI dependency-style)."""
    async with async_session_factory() as session:
        yield session
