"""
Pytest fixtures for assets API tests.
"""

import io
import pytest
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from aiohttp import web

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="session")
def in_memory_engine():
    """Create an in-memory SQLite engine with all asset tables."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    
    from app.database.models import Base
    from app.assets.database.models import (
        Asset,
        AssetInfo,
        AssetCacheState,
        AssetInfoMeta,
        AssetInfoTag,
        Tag,
    )
    
    Base.metadata.create_all(engine)
    
    yield engine
    
    engine.dispose()


@pytest.fixture
def db_session(in_memory_engine) -> Session:
    """Create a fresh database session for each test."""
    SessionLocal = sessionmaker(bind=in_memory_engine)
    session = SessionLocal()
    
    yield session
    
    session.rollback()
    session.close()


@pytest.fixture
def mock_user_manager():
    """Create a mock UserManager that returns a predictable owner_id."""
    mock = MagicMock()
    mock.get_request_user_id = MagicMock(return_value="test-user-123")
    return mock


@pytest.fixture
def app(mock_user_manager) -> web.Application:
    """Create an aiohttp Application with assets routes registered."""
    from app.assets.api.routes import register_assets_system
    
    application = web.Application()
    register_assets_system(application, mock_user_manager)
    return application


@pytest.fixture
def test_image_bytes() -> bytes:
    """Generate a minimal valid PNG image (10x10 red pixels)."""
    from PIL import Image
    
    img = Image.new("RGB", (10, 10), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def tmp_upload_dir(tmp_path):
    """Create a temporary directory for uploads and patch folder_paths."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    
    with patch("folder_paths.get_temp_directory", return_value=str(tmp_path)):
        yield tmp_path


@pytest.fixture(autouse=True)
def patch_create_session(in_memory_engine):
    """Patch create_session to use our in-memory database."""
    SessionLocal = sessionmaker(bind=in_memory_engine)
    
    with patch("app.database.db.Session", SessionLocal):
        with patch("app.database.db.create_session", lambda: SessionLocal()):
            with patch("app.database.db.can_create_session", return_value=True):
                yield


async def test_fixtures_work(db_session, mock_user_manager):
    """Smoke test to verify fixtures are working."""
    assert db_session is not None
    assert mock_user_manager.get_request_user_id(None) == "test-user-123"
