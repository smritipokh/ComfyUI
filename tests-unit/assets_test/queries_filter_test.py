"""
Tests for filtering and pagination query functions in app.assets.database.queries.
"""

import hashlib
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session, sessionmaker

from app.assets.database.models import Asset, AssetInfo, AssetInfoTag, AssetInfoMeta, AssetCacheState, Tag
from app.assets.database.queries import (
    apply_metadata_filter,
    apply_tag_filters,
    ingest_fs_asset,
    list_asset_infos_page,
    replace_asset_info_metadata_projection,
    visible_owner_clause,
)
from app.assets.helpers import utcnow
from sqlalchemy import select
from app.database.models import Base


@pytest.fixture
def clean_db_session():
    """Create a fresh in-memory database for each test."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()
    engine.dispose()


def seed_assets(
    session: Session,
    tmp_path: Path,
    count: int = 10,
    owner_id: str = "",
    tag_sets: list[list[str]] | None = None,
) -> list[str]:
    """
    Create test assets with varied tags.
    Returns list of asset_info_ids.
    """
    asset_info_ids = []
    for i in range(count):
        f = tmp_path / f"test_{i}.png"
        f.write_bytes(b"x" * (100 + i))
        asset_hash = hashlib.sha256(f"unique-{uuid.uuid4()}".encode()).hexdigest()

        if tag_sets is not None:
            tags = tag_sets[i % len(tag_sets)]
        else:
            tags = ["input"] if i % 2 == 0 else ["models", "loras"]

        result = ingest_fs_asset(
            session,
            asset_hash=asset_hash,
            abs_path=str(f),
            size_bytes=100 + i,
            mtime_ns=1000000000 + i,
            mime_type="image/png",
            info_name=f"test_asset_{i}.png",
            owner_id=owner_id,
            tags=tags,
        )
        if result.get("asset_info_id"):
            asset_info_ids.append(result["asset_info_id"])

    session.commit()
    return asset_info_ids


class TestListAssetInfosPage:
    @pytest.fixture
    def seeded_db(self, clean_db_session, tmp_path):
        seed_assets(clean_db_session, tmp_path, 15, owner_id="")
        return clean_db_session

    def test_pagination_limit(self, seeded_db):
        infos, _, total = list_asset_infos_page(
            seeded_db, owner_id="", limit=5, offset=0
        )
        assert len(infos) <= 5
        assert total >= 5

    def test_pagination_offset(self, seeded_db):
        first_page, _, total = list_asset_infos_page(
            seeded_db, owner_id="", limit=5, offset=0
        )
        second_page, _, _ = list_asset_infos_page(
            seeded_db, owner_id="", limit=5, offset=5
        )

        first_ids = {i.id for i in first_page}
        second_ids = {i.id for i in second_page}
        assert first_ids.isdisjoint(second_ids)

    def test_returns_tuple_with_tag_map(self, seeded_db):
        infos, tag_map, total = list_asset_infos_page(
            seeded_db, owner_id="", limit=10, offset=0
        )
        assert isinstance(infos, list)
        assert isinstance(tag_map, dict)
        assert isinstance(total, int)

        for info in infos:
            if info.id in tag_map:
                assert isinstance(tag_map[info.id], list)

    def test_total_count_matches(self, seeded_db):
        _, _, total = list_asset_infos_page(seeded_db, owner_id="", limit=100, offset=0)
        assert total == 15


class TestApplyTagFilters:
    @pytest.fixture
    def tagged_db(self, clean_db_session, tmp_path):
        tag_sets = [
            ["alpha", "beta"],
            ["alpha", "gamma"],
            ["beta", "gamma"],
            ["alpha", "beta", "gamma"],
            ["delta"],
        ]
        seed_assets(clean_db_session, tmp_path, 5, owner_id="", tag_sets=tag_sets)
        return clean_db_session

    def test_include_tags_requires_all(self, tagged_db):
        infos, tag_map, _ = list_asset_infos_page(
            tagged_db,
            owner_id="",
            include_tags=["alpha", "beta"],
            limit=100,
        )
        for info in infos:
            tags = tag_map.get(info.id, [])
            assert "alpha" in tags and "beta" in tags

    def test_include_single_tag(self, tagged_db):
        infos, tag_map, total = list_asset_infos_page(
            tagged_db,
            owner_id="",
            include_tags=["alpha"],
            limit=100,
        )
        assert total >= 1
        for info in infos:
            tags = tag_map.get(info.id, [])
            assert "alpha" in tags

    def test_exclude_tags_excludes_any(self, tagged_db):
        infos, tag_map, _ = list_asset_infos_page(
            tagged_db,
            owner_id="",
            exclude_tags=["delta"],
            limit=100,
        )
        for info in infos:
            tags = tag_map.get(info.id, [])
            assert "delta" not in tags

    def test_exclude_multiple_tags(self, tagged_db):
        infos, tag_map, _ = list_asset_infos_page(
            tagged_db,
            owner_id="",
            exclude_tags=["alpha", "delta"],
            limit=100,
        )
        for info in infos:
            tags = tag_map.get(info.id, [])
            assert "alpha" not in tags
            assert "delta" not in tags

    def test_combine_include_and_exclude(self, tagged_db):
        infos, tag_map, _ = list_asset_infos_page(
            tagged_db,
            owner_id="",
            include_tags=["alpha"],
            exclude_tags=["gamma"],
            limit=100,
        )
        for info in infos:
            tags = tag_map.get(info.id, [])
            assert "alpha" in tags
            assert "gamma" not in tags


class TestApplyMetadataFilter:
    @pytest.fixture
    def metadata_db(self, clean_db_session, tmp_path):
        ids = seed_assets(clean_db_session, tmp_path, 5, owner_id="")
        metadata_sets = [
            {"author": "alice", "version": 1},
            {"author": "bob", "version": 2},
            {"author": "alice", "version": 2},
            {"author": "charlie", "active": True},
            {"author": "alice", "active": False},
        ]
        for i, info_id in enumerate(ids):
            replace_asset_info_metadata_projection(
                clean_db_session,
                asset_info_id=info_id,
                user_metadata=metadata_sets[i],
            )
        clean_db_session.commit()
        return clean_db_session

    def test_filter_by_string_value(self, metadata_db):
        infos, _, total = list_asset_infos_page(
            metadata_db,
            owner_id="",
            metadata_filter={"author": "alice"},
            limit=100,
        )
        assert total == 3
        for info in infos:
            assert info.user_metadata.get("author") == "alice"

    def test_filter_by_numeric_value(self, metadata_db):
        infos, _, total = list_asset_infos_page(
            metadata_db,
            owner_id="",
            metadata_filter={"version": 2},
            limit=100,
        )
        assert total == 2

    def test_filter_by_boolean_value(self, metadata_db):
        infos, _, total = list_asset_infos_page(
            metadata_db,
            owner_id="",
            metadata_filter={"active": True},
            limit=100,
        )
        assert total == 1

    def test_filter_by_multiple_keys(self, metadata_db):
        infos, _, total = list_asset_infos_page(
            metadata_db,
            owner_id="",
            metadata_filter={"author": "alice", "version": 2},
            limit=100,
        )
        assert total == 1

    def test_filter_with_list_values(self, metadata_db):
        infos, _, total = list_asset_infos_page(
            metadata_db,
            owner_id="",
            metadata_filter={"author": ["alice", "bob"]},
            limit=100,
        )
        assert total == 4


class TestVisibleOwnerClause:
    @pytest.fixture
    def multi_owner_db(self, clean_db_session, tmp_path):
        seed_assets(clean_db_session, tmp_path, 3, owner_id="user1")
        seed_assets(clean_db_session, tmp_path, 2, owner_id="user2")
        seed_assets(clean_db_session, tmp_path, 4, owner_id="")
        return clean_db_session

    def test_empty_owner_sees_only_public(self, multi_owner_db):
        infos, _, total = list_asset_infos_page(
            multi_owner_db,
            owner_id="",
            limit=100,
        )
        assert total == 4
        for info in infos:
            assert info.owner_id == ""

    def test_owner_sees_own_plus_public(self, multi_owner_db):
        infos, _, total = list_asset_infos_page(
            multi_owner_db,
            owner_id="user1",
            limit=100,
        )
        assert total == 7
        owner_ids = {info.owner_id for info in infos}
        assert owner_ids == {"user1", ""}

    def test_owner_sees_only_own_and_public(self, multi_owner_db):
        infos, _, total = list_asset_infos_page(
            multi_owner_db,
            owner_id="user2",
            limit=100,
        )
        assert total == 6
        owner_ids = {info.owner_id for info in infos}
        assert owner_ids == {"user2", ""}
        assert all(info.owner_id in ("user2", "") for info in infos)

    def test_nonexistent_owner_sees_public(self, multi_owner_db):
        infos, _, total = list_asset_infos_page(
            multi_owner_db,
            owner_id="unknown-user",
            limit=100,
        )
        assert total == 4
        for info in infos:
            assert info.owner_id == ""


class TestSorting:
    @pytest.fixture
    def sortable_db(self, clean_db_session, tmp_path):
        import time

        ids = []
        names = ["zebra.png", "alpha.png", "mango.png"]
        sizes = [500, 100, 300]

        for i, name in enumerate(names):
            f = tmp_path / f"sort_{i}.png"
            f.write_bytes(b"x" * sizes[i])
            asset_hash = hashlib.sha256(f"sort-{uuid.uuid4()}".encode()).hexdigest()

            result = ingest_fs_asset(
                clean_db_session,
                asset_hash=asset_hash,
                abs_path=str(f),
                size_bytes=sizes[i],
                mtime_ns=1000000000 + i,
                mime_type="image/png",
                info_name=name,
                owner_id="",
                tags=["test"],
            )
            ids.append(result["asset_info_id"])
            time.sleep(0.01)

        clean_db_session.commit()
        return clean_db_session

    def test_sort_by_name_asc(self, sortable_db):
        infos, _, _ = list_asset_infos_page(
            sortable_db,
            owner_id="",
            sort="name",
            order="asc",
            limit=100,
        )
        names = [i.name for i in infos]
        assert names == sorted(names)

    def test_sort_by_name_desc(self, sortable_db):
        infos, _, _ = list_asset_infos_page(
            sortable_db,
            owner_id="",
            sort="name",
            order="desc",
            limit=100,
        )
        names = [i.name for i in infos]
        assert names == sorted(names, reverse=True)

    def test_sort_by_size(self, sortable_db):
        infos, _, _ = list_asset_infos_page(
            sortable_db,
            owner_id="",
            sort="size",
            order="asc",
            limit=100,
        )
        sizes = [i.asset.size_bytes for i in infos]
        assert sizes == sorted(sizes)

    def test_sort_by_created_at_desc(self, sortable_db):
        infos, _, _ = list_asset_infos_page(
            sortable_db,
            owner_id="",
            sort="created_at",
            order="desc",
            limit=100,
        )
        dates = [i.created_at for i in infos]
        assert dates == sorted(dates, reverse=True)

    def test_sort_by_updated_at(self, sortable_db):
        infos, _, _ = list_asset_infos_page(
            sortable_db,
            owner_id="",
            sort="updated_at",
            order="desc",
            limit=100,
        )
        dates = [i.updated_at for i in infos]
        assert dates == sorted(dates, reverse=True)

    def test_sort_by_last_access_time(self, sortable_db):
        infos, _, _ = list_asset_infos_page(
            sortable_db,
            owner_id="",
            sort="last_access_time",
            order="asc",
            limit=100,
        )
        times = [i.last_access_time for i in infos]
        assert times == sorted(times)

    def test_invalid_sort_defaults_to_created_at(self, sortable_db):
        infos, _, _ = list_asset_infos_page(
            sortable_db,
            owner_id="",
            sort="invalid_column",
            order="desc",
            limit=100,
        )
        dates = [i.created_at for i in infos]
        assert dates == sorted(dates, reverse=True)


class TestNameContainsFilter:
    @pytest.fixture
    def named_db(self, clean_db_session, tmp_path):
        names = ["report_2023.pdf", "report_2024.pdf", "image.png", "data.csv"]
        for i, name in enumerate(names):
            f = tmp_path / f"file_{i}.bin"
            f.write_bytes(b"x" * 100)
            asset_hash = hashlib.sha256(f"named-{uuid.uuid4()}".encode()).hexdigest()
            ingest_fs_asset(
                clean_db_session,
                asset_hash=asset_hash,
                abs_path=str(f),
                size_bytes=100,
                mtime_ns=1000000000,
                mime_type="application/octet-stream",
                info_name=name,
                owner_id="",
                tags=["test"],
            )
        clean_db_session.commit()
        return clean_db_session

    def test_name_contains_filter(self, named_db):
        infos, _, total = list_asset_infos_page(
            named_db,
            owner_id="",
            name_contains="report",
            limit=100,
        )
        assert total == 2
        for info in infos:
            assert "report" in info.name.lower()

    def test_name_contains_case_insensitive(self, named_db):
        infos, _, total = list_asset_infos_page(
            named_db,
            owner_id="",
            name_contains="REPORT",
            limit=100,
        )
        assert total == 2

    def test_name_contains_partial_match(self, named_db):
        infos, _, total = list_asset_infos_page(
            named_db,
            owner_id="",
            name_contains=".p",
            limit=100,
        )
        assert total >= 1
