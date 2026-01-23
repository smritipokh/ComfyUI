"""
Tests for core CRUD database query functions in app.assets.database.queries.
"""

import pytest
import uuid
from datetime import datetime, timedelta, timezone

from app.assets.database.queries import (
    asset_exists_by_hash,
    get_asset_by_hash,
    get_asset_info_by_id,
    create_asset_info_for_existing_asset,
    ingest_fs_asset,
    delete_asset_info_by_id,
    touch_asset_info_by_id,
    update_asset_info_full,
    fetch_asset_info_and_asset,
    fetch_asset_info_asset_and_tags,
    ensure_tags_exist,
)
from app.assets.database.models import Asset, AssetInfo, AssetCacheState


def make_hash(seed: str = "a") -> str:
    return "blake3:" + seed * 64


def make_unique_hash() -> str:
    return "blake3:" + uuid.uuid4().hex + uuid.uuid4().hex


class TestAssetExistsByHash:
    def test_returns_true_when_exists(self, db_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"fake png data")
        asset_hash = make_unique_hash()

        ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=len(b"fake png data"),
            mtime_ns=1000000,
            mime_type="image/png",
        )
        db_session.flush()

        assert asset_exists_by_hash(db_session, asset_hash=asset_hash) is True

    def test_returns_false_when_missing(self, db_session):
        assert asset_exists_by_hash(db_session, asset_hash=make_unique_hash()) is False


class TestGetAssetByHash:
    def test_returns_asset_when_exists(self, db_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"test data")
        asset_hash = make_unique_hash()

        ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=9,
            mtime_ns=1000000,
            mime_type="image/png",
        )
        db_session.flush()

        asset = get_asset_by_hash(db_session, asset_hash=asset_hash)
        assert asset is not None
        assert asset.hash == asset_hash
        assert asset.size_bytes == 9
        assert asset.mime_type == "image/png"

    def test_returns_none_when_missing(self, db_session):
        result = get_asset_by_hash(db_session, asset_hash=make_unique_hash())
        assert result is None


class TestGetAssetInfoById:
    def test_returns_asset_info_when_exists(self, db_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"test data")
        asset_hash = make_unique_hash()

        result = ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=9,
            mtime_ns=1000000,
            info_name="my-asset",
            owner_id="user1",
        )
        db_session.flush()

        info = get_asset_info_by_id(db_session, asset_info_id=result["asset_info_id"])
        assert info is not None
        assert info.name == "my-asset"
        assert info.owner_id == "user1"

    def test_returns_none_when_missing(self, db_session):
        fake_id = str(uuid.uuid4())
        result = get_asset_info_by_id(db_session, asset_info_id=fake_id)
        assert result is None


class TestCreateAssetInfoForExistingAsset:
    def test_creates_linked_asset_info(self, db_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"test data")
        asset_hash = make_unique_hash()

        ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=9,
            mtime_ns=1000000,
        )
        db_session.flush()

        info = create_asset_info_for_existing_asset(
            db_session,
            asset_hash=asset_hash,
            name="new-info",
            owner_id="owner123",
            user_metadata={"key": "value"},
        )
        db_session.flush()

        assert info is not None
        assert info.name == "new-info"
        assert info.owner_id == "owner123"

        asset = get_asset_by_hash(db_session, asset_hash=asset_hash)
        assert info.asset_id == asset.id

    def test_raises_on_unknown_hash(self, db_session):
        with pytest.raises(ValueError, match="Unknown asset hash"):
            create_asset_info_for_existing_asset(
                db_session,
                asset_hash=make_unique_hash(),
                name="test",
            )

    def test_returns_existing_on_duplicate(self, db_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"test data")
        asset_hash = make_unique_hash()

        ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=9,
            mtime_ns=1000000,
        )
        db_session.flush()

        info1 = create_asset_info_for_existing_asset(
            db_session,
            asset_hash=asset_hash,
            name="same-name",
            owner_id="owner1",
        )
        db_session.flush()

        info2 = create_asset_info_for_existing_asset(
            db_session,
            asset_hash=asset_hash,
            name="same-name",
            owner_id="owner1",
        )
        db_session.flush()

        assert info1.id == info2.id


class TestIngestFsAsset:
    def test_creates_all_records(self, db_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"fake png data")
        asset_hash = make_unique_hash()

        result = ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=len(b"fake png data"),
            mtime_ns=1000000,
            mime_type="image/png",
            info_name="test-asset",
            owner_id="user1",
        )
        db_session.flush()

        assert result["asset_created"] is True
        assert result["state_created"] is True
        assert result["asset_info_id"] is not None

        asset = get_asset_by_hash(db_session, asset_hash=asset_hash)
        assert asset is not None
        assert asset.size_bytes == len(b"fake png data")

        info = get_asset_info_by_id(db_session, asset_info_id=result["asset_info_id"])
        assert info is not None
        assert info.name == "test-asset"

        cache_states = db_session.query(AssetCacheState).filter_by(asset_id=asset.id).all()
        assert len(cache_states) == 1
        assert cache_states[0].file_path == str(test_file)

    def test_idempotent_on_same_file(self, db_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"data")
        asset_hash = make_unique_hash()

        result1 = ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=4,
            mtime_ns=1000000,
            info_name="test",
        )
        db_session.flush()

        result2 = ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=4,
            mtime_ns=1000000,
            info_name="test",
        )
        db_session.flush()

        assert result1["asset_info_id"] == result2["asset_info_id"]
        assert result2["asset_created"] is False

    def test_creates_with_tags(self, db_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"data")
        asset_hash = make_unique_hash()

        result = ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=4,
            mtime_ns=1000000,
            info_name="test",
            tags=["tag1", "tag2"],
        )
        db_session.flush()

        info, asset, tags = fetch_asset_info_asset_and_tags(
            db_session,
            asset_info_id=result["asset_info_id"],
        )
        assert set(tags) == {"tag1", "tag2"}


class TestDeleteAssetInfoById:
    def test_deletes_existing_record(self, db_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"data")
        asset_hash = make_unique_hash()

        result = ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=4,
            mtime_ns=1000000,
            info_name="to-delete",
            owner_id="user1",
        )
        db_session.flush()

        deleted = delete_asset_info_by_id(
            db_session,
            asset_info_id=result["asset_info_id"],
            owner_id="user1",
        )
        db_session.flush()

        assert deleted is True
        assert get_asset_info_by_id(db_session, asset_info_id=result["asset_info_id"]) is None

    def test_returns_false_for_nonexistent(self, db_session):
        result = delete_asset_info_by_id(
            db_session,
            asset_info_id=str(uuid.uuid4()),
            owner_id="user1",
        )
        assert result is False

    def test_respects_owner_visibility(self, db_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"data")
        asset_hash = make_unique_hash()

        result = ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=4,
            mtime_ns=1000000,
            info_name="owned-asset",
            owner_id="user1",
        )
        db_session.flush()

        deleted = delete_asset_info_by_id(
            db_session,
            asset_info_id=result["asset_info_id"],
            owner_id="different-user",
        )
        assert deleted is False

        assert get_asset_info_by_id(db_session, asset_info_id=result["asset_info_id"]) is not None


class TestTouchAssetInfoById:
    def test_updates_last_access_time(self, db_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"data")
        asset_hash = make_unique_hash()

        result = ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=4,
            mtime_ns=1000000,
            info_name="test",
        )
        db_session.flush()

        info_before = get_asset_info_by_id(db_session, asset_info_id=result["asset_info_id"])
        original_time = info_before.last_access_time

        new_time = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1)
        touch_asset_info_by_id(
            db_session,
            asset_info_id=result["asset_info_id"],
            ts=new_time,
        )
        db_session.flush()
        db_session.expire_all()

        info_after = get_asset_info_by_id(db_session, asset_info_id=result["asset_info_id"])
        assert info_after.last_access_time == new_time
        assert info_after.last_access_time > original_time

    def test_only_if_newer_respects_flag(self, db_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"data")
        asset_hash = make_unique_hash()

        result = ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=4,
            mtime_ns=1000000,
            info_name="test",
        )
        db_session.flush()

        info = get_asset_info_by_id(db_session, asset_info_id=result["asset_info_id"])
        original_time = info.last_access_time

        older_time = original_time - timedelta(hours=1)
        touch_asset_info_by_id(
            db_session,
            asset_info_id=result["asset_info_id"],
            ts=older_time,
            only_if_newer=True,
        )
        db_session.flush()
        db_session.expire_all()

        info_after = get_asset_info_by_id(db_session, asset_info_id=result["asset_info_id"])
        assert info_after.last_access_time == original_time


class TestUpdateAssetInfoFull:
    def test_updates_name(self, db_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"data")
        asset_hash = make_unique_hash()

        result = ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=4,
            mtime_ns=1000000,
            info_name="original-name",
        )
        db_session.flush()

        updated = update_asset_info_full(
            db_session,
            asset_info_id=result["asset_info_id"],
            name="new-name",
        )
        db_session.flush()

        assert updated.name == "new-name"

    def test_updates_tags(self, db_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"data")
        asset_hash = make_unique_hash()

        result = ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=4,
            mtime_ns=1000000,
            info_name="test",
        )
        db_session.flush()

        update_asset_info_full(
            db_session,
            asset_info_id=result["asset_info_id"],
            tags=["newtag1", "newtag2"],
        )
        db_session.flush()

        _, _, tags = fetch_asset_info_asset_and_tags(
            db_session,
            asset_info_id=result["asset_info_id"],
        )
        assert set(tags) == {"newtag1", "newtag2"}

    def test_updates_metadata(self, db_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"data")
        asset_hash = make_unique_hash()

        result = ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=4,
            mtime_ns=1000000,
            info_name="test",
        )
        db_session.flush()

        update_asset_info_full(
            db_session,
            asset_info_id=result["asset_info_id"],
            user_metadata={"custom_key": "custom_value"},
        )
        db_session.flush()
        db_session.expire_all()

        info = get_asset_info_by_id(db_session, asset_info_id=result["asset_info_id"])
        assert "custom_key" in info.user_metadata
        assert info.user_metadata["custom_key"] == "custom_value"

    def test_raises_on_invalid_id(self, db_session):
        with pytest.raises(ValueError, match="not found"):
            update_asset_info_full(
                db_session,
                asset_info_id=str(uuid.uuid4()),
                name="test",
            )


class TestFetchAssetInfoAndAsset:
    def test_returns_tuple_when_exists(self, db_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"data")
        asset_hash = make_unique_hash()

        result = ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=4,
            mtime_ns=1000000,
            info_name="test",
            mime_type="image/png",
        )
        db_session.flush()

        fetched = fetch_asset_info_and_asset(
            db_session,
            asset_info_id=result["asset_info_id"],
        )

        assert fetched is not None
        info, asset = fetched
        assert info.name == "test"
        assert asset.hash == asset_hash
        assert asset.mime_type == "image/png"

    def test_returns_none_when_missing(self, db_session):
        result = fetch_asset_info_and_asset(
            db_session,
            asset_info_id=str(uuid.uuid4()),
        )
        assert result is None

    def test_respects_owner_visibility(self, db_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"data")
        asset_hash = make_unique_hash()

        result = ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=4,
            mtime_ns=1000000,
            info_name="test",
            owner_id="user1",
        )
        db_session.flush()

        fetched = fetch_asset_info_and_asset(
            db_session,
            asset_info_id=result["asset_info_id"],
            owner_id="different-user",
        )
        assert fetched is None


class TestFetchAssetInfoAssetAndTags:
    def test_returns_tuple_with_tags(self, db_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"data")
        asset_hash = make_unique_hash()

        result = ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=4,
            mtime_ns=1000000,
            info_name="test",
            tags=["alpha", "beta"],
        )
        db_session.flush()

        fetched = fetch_asset_info_asset_and_tags(
            db_session,
            asset_info_id=result["asset_info_id"],
        )

        assert fetched is not None
        info, asset, tags = fetched
        assert info.name == "test"
        assert asset.hash == asset_hash
        assert set(tags) == {"alpha", "beta"}

    def test_returns_empty_tags_when_none(self, db_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"data")
        asset_hash = make_unique_hash()

        result = ingest_fs_asset(
            db_session,
            asset_hash=asset_hash,
            abs_path=str(test_file),
            size_bytes=4,
            mtime_ns=1000000,
            info_name="test",
        )
        db_session.flush()

        fetched = fetch_asset_info_asset_and_tags(
            db_session,
            asset_info_id=result["asset_info_id"],
        )

        assert fetched is not None
        info, asset, tags = fetched
        assert tags == []

    def test_returns_none_when_missing(self, db_session):
        result = fetch_asset_info_asset_and_tags(
            db_session,
            asset_info_id=str(uuid.uuid4()),
        )
        assert result is None
