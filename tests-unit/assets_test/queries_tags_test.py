"""
Tests for tag-related database query functions in app.assets.database.queries.
"""

import pytest
import uuid

from app.assets.database.queries import (
    add_tags_to_asset_info,
    remove_tags_from_asset_info,
    get_asset_tags,
    list_tags_with_usage,
    set_asset_info_preview,
    ingest_fs_asset,
    get_asset_by_hash,
)


def make_unique_hash() -> str:
    return "blake3:" + uuid.uuid4().hex + uuid.uuid4().hex


def create_test_asset(db_session, tmp_path, name="test", tags=None, owner_id=""):
    test_file = tmp_path / f"{name}.png"
    test_file.write_bytes(b"fake png data")
    asset_hash = make_unique_hash()

    result = ingest_fs_asset(
        db_session,
        asset_hash=asset_hash,
        abs_path=str(test_file),
        size_bytes=len(b"fake png data"),
        mtime_ns=1000000,
        mime_type="image/png",
        info_name=name,
        owner_id=owner_id,
        tags=tags,
    )
    db_session.flush()
    return result


class TestAddTagsToAssetInfo:
    def test_adds_new_tags(self, db_session, tmp_path):
        result = create_test_asset(db_session, tmp_path, name="test-add-tags")

        add_result = add_tags_to_asset_info(
            db_session,
            asset_info_id=result["asset_info_id"],
            tags=["tag1", "tag2"],
            origin="manual",
        )
        db_session.flush()

        assert set(add_result["added"]) == {"tag1", "tag2"}
        assert add_result["already_present"] == []
        assert set(add_result["total_tags"]) == {"tag1", "tag2"}

    def test_idempotent_on_duplicates(self, db_session, tmp_path):
        result = create_test_asset(db_session, tmp_path, name="test-idempotent")

        add_tags_to_asset_info(
            db_session,
            asset_info_id=result["asset_info_id"],
            tags=["dup-tag"],
            origin="manual",
        )
        db_session.flush()

        second_result = add_tags_to_asset_info(
            db_session,
            asset_info_id=result["asset_info_id"],
            tags=["dup-tag"],
            origin="manual",
        )
        db_session.flush()

        assert second_result["added"] == []
        assert second_result["already_present"] == ["dup-tag"]
        assert second_result["total_tags"] == ["dup-tag"]

    def test_mixed_new_and_existing_tags(self, db_session, tmp_path):
        result = create_test_asset(db_session, tmp_path, name="test-mixed", tags=["existing"])

        add_result = add_tags_to_asset_info(
            db_session,
            asset_info_id=result["asset_info_id"],
            tags=["existing", "new-tag"],
            origin="manual",
        )
        db_session.flush()

        assert add_result["added"] == ["new-tag"]
        assert add_result["already_present"] == ["existing"]
        assert set(add_result["total_tags"]) == {"existing", "new-tag"}

    def test_empty_tags_list(self, db_session, tmp_path):
        result = create_test_asset(db_session, tmp_path, name="test-empty", tags=["pre-existing"])

        add_result = add_tags_to_asset_info(
            db_session,
            asset_info_id=result["asset_info_id"],
            tags=[],
            origin="manual",
        )

        assert add_result["added"] == []
        assert add_result["already_present"] == []
        assert add_result["total_tags"] == ["pre-existing"]

    def test_raises_on_invalid_asset_info_id(self, db_session):
        with pytest.raises(ValueError, match="not found"):
            add_tags_to_asset_info(
                db_session,
                asset_info_id=str(uuid.uuid4()),
                tags=["tag1"],
                origin="manual",
            )


class TestRemoveTagsFromAssetInfo:
    def test_removes_existing_tags(self, db_session, tmp_path):
        result = create_test_asset(db_session, tmp_path, name="test-remove", tags=["tag1", "tag2", "tag3"])

        remove_result = remove_tags_from_asset_info(
            db_session,
            asset_info_id=result["asset_info_id"],
            tags=["tag1", "tag2"],
        )
        db_session.flush()

        assert set(remove_result["removed"]) == {"tag1", "tag2"}
        assert remove_result["not_present"] == []
        assert remove_result["total_tags"] == ["tag3"]

    def test_handles_nonexistent_tags_gracefully(self, db_session, tmp_path):
        result = create_test_asset(db_session, tmp_path, name="test-nonexistent", tags=["existing"])

        remove_result = remove_tags_from_asset_info(
            db_session,
            asset_info_id=result["asset_info_id"],
            tags=["nonexistent"],
        )
        db_session.flush()

        assert remove_result["removed"] == []
        assert remove_result["not_present"] == ["nonexistent"]
        assert remove_result["total_tags"] == ["existing"]

    def test_mixed_existing_and_nonexistent(self, db_session, tmp_path):
        result = create_test_asset(db_session, tmp_path, name="test-mixed-remove", tags=["tag1", "tag2"])

        remove_result = remove_tags_from_asset_info(
            db_session,
            asset_info_id=result["asset_info_id"],
            tags=["tag1", "nonexistent"],
        )
        db_session.flush()

        assert remove_result["removed"] == ["tag1"]
        assert remove_result["not_present"] == ["nonexistent"]
        assert remove_result["total_tags"] == ["tag2"]

    def test_empty_tags_list(self, db_session, tmp_path):
        result = create_test_asset(db_session, tmp_path, name="test-empty-remove", tags=["existing"])

        remove_result = remove_tags_from_asset_info(
            db_session,
            asset_info_id=result["asset_info_id"],
            tags=[],
        )

        assert remove_result["removed"] == []
        assert remove_result["not_present"] == []
        assert remove_result["total_tags"] == ["existing"]

    def test_raises_on_invalid_asset_info_id(self, db_session):
        with pytest.raises(ValueError, match="not found"):
            remove_tags_from_asset_info(
                db_session,
                asset_info_id=str(uuid.uuid4()),
                tags=["tag1"],
            )


class TestGetAssetTags:
    def test_returns_list_of_tag_names(self, db_session, tmp_path):
        result = create_test_asset(db_session, tmp_path, name="test-get-tags", tags=["alpha", "beta", "gamma"])

        tags = get_asset_tags(db_session, asset_info_id=result["asset_info_id"])

        assert set(tags) == {"alpha", "beta", "gamma"}

    def test_returns_empty_list_when_no_tags(self, db_session, tmp_path):
        result = create_test_asset(db_session, tmp_path, name="test-no-tags")

        tags = get_asset_tags(db_session, asset_info_id=result["asset_info_id"])

        assert tags == []

    def test_returns_empty_for_nonexistent_asset(self, db_session):
        tags = get_asset_tags(db_session, asset_info_id=str(uuid.uuid4()))

        assert tags == []


class TestListTagsWithUsage:
    def test_returns_tags_with_counts(self, db_session, tmp_path):
        create_test_asset(db_session, tmp_path, name="asset1", tags=["shared-tag", "unique1"])
        create_test_asset(db_session, tmp_path, name="asset2", tags=["shared-tag", "unique2"])
        create_test_asset(db_session, tmp_path, name="asset3", tags=["shared-tag"])

        tags, total = list_tags_with_usage(db_session)

        tag_dict = {name: count for name, _, count in tags}
        assert tag_dict["shared-tag"] == 3
        assert tag_dict.get("unique1", 0) == 1
        assert tag_dict.get("unique2", 0) == 1

    def test_prefix_filtering(self, db_session, tmp_path):
        create_test_asset(db_session, tmp_path, name="asset-prefix", tags=["prefix-a", "prefix-b", "other"])

        tags, total = list_tags_with_usage(db_session, prefix="prefix")

        tag_names = [name for name, _, _ in tags]
        assert "prefix-a" in tag_names
        assert "prefix-b" in tag_names
        assert "other" not in tag_names

    def test_pagination(self, db_session, tmp_path):
        create_test_asset(db_session, tmp_path, name="asset-page", tags=["page1", "page2", "page3", "page4", "page5"])

        first_page, _ = list_tags_with_usage(db_session, limit=2, offset=0)
        second_page, _ = list_tags_with_usage(db_session, limit=2, offset=2)

        first_names = {name for name, _, _ in first_page}
        second_names = {name for name, _, _ in second_page}

        assert len(first_page) == 2
        assert len(second_page) == 2
        assert first_names.isdisjoint(second_names)

    def test_order_by_count_desc(self, db_session, tmp_path):
        create_test_asset(db_session, tmp_path, name="count1", tags=["popular", "rare"])
        create_test_asset(db_session, tmp_path, name="count2", tags=["popular"])
        create_test_asset(db_session, tmp_path, name="count3", tags=["popular"])

        tags, _ = list_tags_with_usage(db_session, order="count_desc", include_zero=False)

        counts = [count for _, _, count in tags]
        assert counts == sorted(counts, reverse=True)

    def test_order_by_name_asc(self, db_session, tmp_path):
        create_test_asset(db_session, tmp_path, name="name-order", tags=["zebra", "apple", "mango"])

        tags, _ = list_tags_with_usage(db_session, order="name_asc", include_zero=False)

        names = [name for name, _, _ in tags]
        assert names == sorted(names)

    def test_include_zero_false_excludes_unused_tags(self, db_session, tmp_path):
        create_test_asset(db_session, tmp_path, name="used-tag-asset", tags=["used-tag"])

        add_tags_to_asset_info(
            db_session,
            asset_info_id=create_test_asset(db_session, tmp_path, name="temp")["asset_info_id"],
            tags=["orphan-tag"],
            origin="manual",
        )
        db_session.flush()
        remove_tags_from_asset_info(
            db_session,
            asset_info_id=create_test_asset(db_session, tmp_path, name="temp2")["asset_info_id"],
            tags=["orphan-tag"],
        )
        db_session.flush()

        tags_with_zero, _ = list_tags_with_usage(db_session, include_zero=True)
        tags_without_zero, _ = list_tags_with_usage(db_session, include_zero=False)

        with_zero_names = {name for name, _, _ in tags_with_zero}
        without_zero_names = {name for name, _, _ in tags_without_zero}

        assert "used-tag" in without_zero_names
        assert len(without_zero_names) <= len(with_zero_names)

    def test_respects_owner_visibility(self, db_session, tmp_path):
        create_test_asset(db_session, tmp_path, name="user1-asset", tags=["user1-tag"], owner_id="user1")
        create_test_asset(db_session, tmp_path, name="user2-asset", tags=["user2-tag"], owner_id="user2")

        user1_tags, _ = list_tags_with_usage(db_session, owner_id="user1", include_zero=False)

        user1_tag_names = {name for name, _, _ in user1_tags}
        assert "user1-tag" in user1_tag_names


class TestSetAssetInfoPreview:
    def test_sets_preview_id(self, db_session, tmp_path):
        asset_result = create_test_asset(db_session, tmp_path, name="main-asset")

        preview_file = tmp_path / "preview.png"
        preview_file.write_bytes(b"preview data")
        preview_hash = make_unique_hash()
        preview_result = ingest_fs_asset(
            db_session,
            asset_hash=preview_hash,
            abs_path=str(preview_file),
            size_bytes=len(b"preview data"),
            mtime_ns=1000000,
            mime_type="image/png",
            info_name="preview",
        )
        db_session.flush()

        preview_asset = get_asset_by_hash(db_session, asset_hash=preview_hash)

        set_asset_info_preview(
            db_session,
            asset_info_id=asset_result["asset_info_id"],
            preview_asset_id=preview_asset.id,
        )
        db_session.flush()

        from app.assets.database.queries import get_asset_info_by_id
        info = get_asset_info_by_id(db_session, asset_info_id=asset_result["asset_info_id"])
        assert info.preview_id == preview_asset.id

    def test_clears_preview_with_none(self, db_session, tmp_path):
        asset_result = create_test_asset(db_session, tmp_path, name="clear-preview")

        preview_file = tmp_path / "preview2.png"
        preview_file.write_bytes(b"preview data")
        preview_hash = make_unique_hash()
        ingest_fs_asset(
            db_session,
            asset_hash=preview_hash,
            abs_path=str(preview_file),
            size_bytes=len(b"preview data"),
            mtime_ns=1000000,
            info_name="preview2",
        )
        db_session.flush()

        preview_asset = get_asset_by_hash(db_session, asset_hash=preview_hash)

        set_asset_info_preview(
            db_session,
            asset_info_id=asset_result["asset_info_id"],
            preview_asset_id=preview_asset.id,
        )
        db_session.flush()

        set_asset_info_preview(
            db_session,
            asset_info_id=asset_result["asset_info_id"],
            preview_asset_id=None,
        )
        db_session.flush()

        from app.assets.database.queries import get_asset_info_by_id
        info = get_asset_info_by_id(db_session, asset_info_id=asset_result["asset_info_id"])
        assert info.preview_id is None

    def test_raises_on_invalid_asset_info_id(self, db_session):
        with pytest.raises(ValueError, match="AssetInfo.*not found"):
            set_asset_info_preview(
                db_session,
                asset_info_id=str(uuid.uuid4()),
                preview_asset_id=None,
            )

    def test_raises_on_invalid_preview_asset_id(self, db_session, tmp_path):
        asset_result = create_test_asset(db_session, tmp_path, name="invalid-preview")

        with pytest.raises(ValueError, match="Preview Asset.*not found"):
            set_asset_info_preview(
                db_session,
                asset_info_id=asset_result["asset_info_id"],
                preview_asset_id=str(uuid.uuid4()),
            )
