"""
Comprehensive tests for Pydantic schemas in the assets API.
"""

import pytest
from pydantic import ValidationError

from app.assets.api.schemas_in import (
    ListAssetsQuery,
    UpdateAssetBody,
    CreateFromHashBody,
    UploadAssetSpec,
    SetPreviewBody,
    TagsAdd,
    TagsRemove,
    TagsListQuery,
    ScheduleAssetScanBody,
)


class TestListAssetsQuery:
    def test_defaults(self):
        q = ListAssetsQuery()
        assert q.limit == 20
        assert q.offset == 0
        assert q.sort == "created_at"
        assert q.order == "desc"
        assert q.include_tags == []
        assert q.exclude_tags == []
        assert q.name_contains is None
        assert q.metadata_filter is None

    def test_csv_tags_parsing_string(self):
        q = ListAssetsQuery.model_validate({"include_tags": "a,b,c"})
        assert q.include_tags == ["a", "b", "c"]

    def test_csv_tags_parsing_with_whitespace(self):
        q = ListAssetsQuery.model_validate({"include_tags": " a , b , c "})
        assert q.include_tags == ["a", "b", "c"]

    def test_csv_tags_parsing_list(self):
        q = ListAssetsQuery.model_validate({"include_tags": ["a", "b", "c"]})
        assert q.include_tags == ["a", "b", "c"]

    def test_csv_tags_parsing_list_with_csv(self):
        q = ListAssetsQuery.model_validate({"include_tags": ["a,b", "c"]})
        assert q.include_tags == ["a", "b", "c"]

    def test_csv_tags_exclude_tags(self):
        q = ListAssetsQuery.model_validate({"exclude_tags": "x,y,z"})
        assert q.exclude_tags == ["x", "y", "z"]

    def test_csv_tags_empty_string(self):
        q = ListAssetsQuery.model_validate({"include_tags": ""})
        assert q.include_tags == []

    def test_csv_tags_none(self):
        q = ListAssetsQuery.model_validate({"include_tags": None})
        assert q.include_tags == []

    def test_metadata_filter_json_string(self):
        q = ListAssetsQuery.model_validate({"metadata_filter": '{"key": "value"}'})
        assert q.metadata_filter == {"key": "value"}

    def test_metadata_filter_dict(self):
        q = ListAssetsQuery.model_validate({"metadata_filter": {"key": "value"}})
        assert q.metadata_filter == {"key": "value"}

    def test_metadata_filter_none(self):
        q = ListAssetsQuery.model_validate({"metadata_filter": None})
        assert q.metadata_filter is None

    def test_metadata_filter_empty_string(self):
        q = ListAssetsQuery.model_validate({"metadata_filter": ""})
        assert q.metadata_filter is None

    def test_metadata_filter_invalid_json(self):
        with pytest.raises(ValidationError) as exc_info:
            ListAssetsQuery.model_validate({"metadata_filter": "not json"})
        assert "must be JSON" in str(exc_info.value)

    def test_metadata_filter_non_object_json(self):
        with pytest.raises(ValidationError) as exc_info:
            ListAssetsQuery.model_validate({"metadata_filter": "[1, 2, 3]"})
        assert "must be a JSON object" in str(exc_info.value)

    def test_limit_bounds_min(self):
        with pytest.raises(ValidationError):
            ListAssetsQuery.model_validate({"limit": 0})

    def test_limit_bounds_max(self):
        with pytest.raises(ValidationError):
            ListAssetsQuery.model_validate({"limit": 501})

    def test_limit_bounds_valid(self):
        q = ListAssetsQuery.model_validate({"limit": 500})
        assert q.limit == 500

    def test_offset_bounds_min(self):
        with pytest.raises(ValidationError):
            ListAssetsQuery.model_validate({"offset": -1})

    def test_sort_enum_valid(self):
        for sort_val in ["name", "created_at", "updated_at", "size", "last_access_time"]:
            q = ListAssetsQuery.model_validate({"sort": sort_val})
            assert q.sort == sort_val

    def test_sort_enum_invalid(self):
        with pytest.raises(ValidationError):
            ListAssetsQuery.model_validate({"sort": "invalid"})

    def test_order_enum_valid(self):
        for order_val in ["asc", "desc"]:
            q = ListAssetsQuery.model_validate({"order": order_val})
            assert q.order == order_val

    def test_order_enum_invalid(self):
        with pytest.raises(ValidationError):
            ListAssetsQuery.model_validate({"order": "invalid"})


class TestUpdateAssetBody:
    def test_requires_at_least_one_field(self):
        with pytest.raises(ValidationError) as exc_info:
            UpdateAssetBody.model_validate({})
        assert "at least one of" in str(exc_info.value)

    def test_name_only(self):
        body = UpdateAssetBody.model_validate({"name": "new_name"})
        assert body.name == "new_name"
        assert body.tags is None
        assert body.user_metadata is None

    def test_tags_only(self):
        body = UpdateAssetBody.model_validate({"tags": ["tag1", "tag2"]})
        assert body.tags == ["tag1", "tag2"]

    def test_user_metadata_only(self):
        body = UpdateAssetBody.model_validate({"user_metadata": {"key": "value"}})
        assert body.user_metadata == {"key": "value"}

    def test_tags_must_be_list_of_strings(self):
        with pytest.raises(ValidationError) as exc_info:
            UpdateAssetBody.model_validate({"tags": "not_a_list"})
        assert "list" in str(exc_info.value).lower()

    def test_tags_must_contain_strings(self):
        with pytest.raises(ValidationError) as exc_info:
            UpdateAssetBody.model_validate({"tags": [1, 2, 3]})
        assert "string" in str(exc_info.value).lower()

    def test_multiple_fields(self):
        body = UpdateAssetBody.model_validate({
            "name": "new_name",
            "tags": ["tag1"],
            "user_metadata": {"foo": "bar"}
        })
        assert body.name == "new_name"
        assert body.tags == ["tag1"]
        assert body.user_metadata == {"foo": "bar"}


class TestCreateFromHashBody:
    def test_valid_blake3(self):
        body = CreateFromHashBody(
            hash="blake3:" + "a" * 64,
            name="test"
        )
        assert body.hash.startswith("blake3:")
        assert body.name == "test"

    def test_valid_blake3_lowercase(self):
        body = CreateFromHashBody(
            hash="BLAKE3:" + "A" * 64,
            name="test"
        )
        assert body.hash == "blake3:" + "a" * 64

    def test_rejects_sha256(self):
        with pytest.raises(ValidationError) as exc_info:
            CreateFromHashBody(hash="sha256:" + "a" * 64, name="test")
        assert "blake3" in str(exc_info.value).lower()

    def test_rejects_no_colon(self):
        with pytest.raises(ValidationError) as exc_info:
            CreateFromHashBody(hash="a" * 64, name="test")
        assert "blake3:<hex>" in str(exc_info.value)

    def test_rejects_invalid_hex(self):
        with pytest.raises(ValidationError) as exc_info:
            CreateFromHashBody(hash="blake3:" + "g" * 64, name="test")
        assert "hex" in str(exc_info.value).lower()

    def test_rejects_empty_digest(self):
        with pytest.raises(ValidationError) as exc_info:
            CreateFromHashBody(hash="blake3:", name="test")
        assert "hex" in str(exc_info.value).lower()

    def test_default_tags_empty(self):
        body = CreateFromHashBody(hash="blake3:" + "a" * 64, name="test")
        assert body.tags == []

    def test_default_user_metadata_empty(self):
        body = CreateFromHashBody(hash="blake3:" + "a" * 64, name="test")
        assert body.user_metadata == {}

    def test_tags_normalized_lowercase(self):
        body = CreateFromHashBody(
            hash="blake3:" + "a" * 64,
            name="test",
            tags=["TAG1", "Tag2"]
        )
        assert body.tags == ["tag1", "tag2"]

    def test_tags_deduplicated(self):
        body = CreateFromHashBody(
            hash="blake3:" + "a" * 64,
            name="test",
            tags=["tag", "TAG", "tag"]
        )
        assert body.tags == ["tag"]

    def test_tags_csv_parsing(self):
        body = CreateFromHashBody(
            hash="blake3:" + "a" * 64,
            name="test",
            tags="a,b,c"
        )
        assert body.tags == ["a", "b", "c"]

    def test_whitespace_stripping(self):
        body = CreateFromHashBody(
            hash="  blake3:" + "a" * 64 + "  ",
            name="  test  "
        )
        assert body.hash == "blake3:" + "a" * 64
        assert body.name == "test"


class TestUploadAssetSpec:
    def test_first_tag_must_be_root_type_models(self):
        spec = UploadAssetSpec.model_validate({"tags": ["models", "loras"]})
        assert spec.tags[0] == "models"

    def test_first_tag_must_be_root_type_input(self):
        spec = UploadAssetSpec.model_validate({"tags": ["input"]})
        assert spec.tags[0] == "input"

    def test_first_tag_must_be_root_type_output(self):
        spec = UploadAssetSpec.model_validate({"tags": ["output"]})
        assert spec.tags[0] == "output"

    def test_rejects_invalid_first_tag(self):
        with pytest.raises(ValidationError) as exc_info:
            UploadAssetSpec.model_validate({"tags": ["invalid"]})
        assert "models, input, output" in str(exc_info.value)

    def test_models_requires_category_tag(self):
        with pytest.raises(ValidationError) as exc_info:
            UploadAssetSpec.model_validate({"tags": ["models"]})
        assert "category tag" in str(exc_info.value)

    def test_input_does_not_require_second_tag(self):
        spec = UploadAssetSpec.model_validate({"tags": ["input"]})
        assert spec.tags == ["input"]

    def test_output_does_not_require_second_tag(self):
        spec = UploadAssetSpec.model_validate({"tags": ["output"]})
        assert spec.tags == ["output"]

    def test_tags_empty_rejected(self):
        with pytest.raises(ValidationError):
            UploadAssetSpec.model_validate({"tags": []})

    def test_tags_csv_parsing(self):
        spec = UploadAssetSpec.model_validate({"tags": "models,loras"})
        assert spec.tags == ["models", "loras"]

    def test_tags_json_array_parsing(self):
        spec = UploadAssetSpec.model_validate({"tags": '["models", "loras"]'})
        assert spec.tags == ["models", "loras"]

    def test_tags_normalized_lowercase(self):
        spec = UploadAssetSpec.model_validate({"tags": ["MODELS", "LORAS"]})
        assert spec.tags == ["models", "loras"]

    def test_tags_deduplicated(self):
        spec = UploadAssetSpec.model_validate({"tags": ["models", "loras", "models"]})
        assert spec.tags == ["models", "loras"]

    def test_hash_validation_valid_blake3(self):
        spec = UploadAssetSpec.model_validate({
            "tags": ["input"],
            "hash": "blake3:" + "a" * 64
        })
        assert spec.hash == "blake3:" + "a" * 64

    def test_hash_validation_rejects_sha256(self):
        with pytest.raises(ValidationError):
            UploadAssetSpec.model_validate({
                "tags": ["input"],
                "hash": "sha256:" + "a" * 64
            })

    def test_hash_none_allowed(self):
        spec = UploadAssetSpec.model_validate({"tags": ["input"], "hash": None})
        assert spec.hash is None

    def test_hash_empty_string_becomes_none(self):
        spec = UploadAssetSpec.model_validate({"tags": ["input"], "hash": ""})
        assert spec.hash is None

    def test_name_optional(self):
        spec = UploadAssetSpec.model_validate({"tags": ["input"]})
        assert spec.name is None

    def test_name_max_length(self):
        with pytest.raises(ValidationError):
            UploadAssetSpec.model_validate({
                "tags": ["input"],
                "name": "x" * 513
            })

    def test_user_metadata_json_string(self):
        spec = UploadAssetSpec.model_validate({
            "tags": ["input"],
            "user_metadata": '{"key": "value"}'
        })
        assert spec.user_metadata == {"key": "value"}

    def test_user_metadata_dict(self):
        spec = UploadAssetSpec.model_validate({
            "tags": ["input"],
            "user_metadata": {"key": "value"}
        })
        assert spec.user_metadata == {"key": "value"}

    def test_user_metadata_empty_string(self):
        spec = UploadAssetSpec.model_validate({
            "tags": ["input"],
            "user_metadata": ""
        })
        assert spec.user_metadata == {}

    def test_user_metadata_invalid_json(self):
        with pytest.raises(ValidationError) as exc_info:
            UploadAssetSpec.model_validate({
                "tags": ["input"],
                "user_metadata": "not json"
            })
        assert "must be JSON" in str(exc_info.value)


class TestSetPreviewBody:
    def test_valid_uuid(self):
        body = SetPreviewBody.model_validate({"preview_id": "550e8400-e29b-41d4-a716-446655440000"})
        assert body.preview_id == "550e8400-e29b-41d4-a716-446655440000"

    def test_none_allowed(self):
        body = SetPreviewBody.model_validate({"preview_id": None})
        assert body.preview_id is None

    def test_empty_string_becomes_none(self):
        body = SetPreviewBody.model_validate({"preview_id": ""})
        assert body.preview_id is None

    def test_whitespace_only_becomes_none(self):
        body = SetPreviewBody.model_validate({"preview_id": "   "})
        assert body.preview_id is None

    def test_invalid_uuid(self):
        with pytest.raises(ValidationError) as exc_info:
            SetPreviewBody.model_validate({"preview_id": "not-a-uuid"})
        assert "UUID" in str(exc_info.value)

    def test_default_is_none(self):
        body = SetPreviewBody.model_validate({})
        assert body.preview_id is None


class TestTagsAdd:
    def test_non_empty_required(self):
        with pytest.raises(ValidationError):
            TagsAdd.model_validate({"tags": []})

    def test_valid_tags(self):
        body = TagsAdd.model_validate({"tags": ["tag1", "tag2"]})
        assert body.tags == ["tag1", "tag2"]

    def test_tags_normalized_lowercase(self):
        body = TagsAdd.model_validate({"tags": ["TAG1", "Tag2"]})
        assert body.tags == ["tag1", "tag2"]

    def test_tags_whitespace_stripped(self):
        body = TagsAdd.model_validate({"tags": ["  tag1  ", "  tag2  "]})
        assert body.tags == ["tag1", "tag2"]

    def test_tags_deduplicated(self):
        body = TagsAdd.model_validate({"tags": ["tag", "TAG", "tag"]})
        assert body.tags == ["tag"]

    def test_empty_strings_filtered(self):
        body = TagsAdd.model_validate({"tags": ["tag1", "", "  ", "tag2"]})
        assert body.tags == ["tag1", "tag2"]

    def test_missing_tags_field_fails(self):
        with pytest.raises(ValidationError):
            TagsAdd.model_validate({})


class TestTagsRemove:
    def test_non_empty_required(self):
        with pytest.raises(ValidationError):
            TagsRemove.model_validate({"tags": []})

    def test_valid_tags(self):
        body = TagsRemove.model_validate({"tags": ["tag1", "tag2"]})
        assert body.tags == ["tag1", "tag2"]

    def test_inherits_normalization(self):
        body = TagsRemove.model_validate({"tags": ["TAG1", "Tag2"]})
        assert body.tags == ["tag1", "tag2"]


class TestTagsListQuery:
    def test_defaults(self):
        q = TagsListQuery()
        assert q.prefix is None
        assert q.limit == 100
        assert q.offset == 0
        assert q.order == "count_desc"
        assert q.include_zero is True

    def test_prefix_normalized_lowercase(self):
        q = TagsListQuery.model_validate({"prefix": "PREFIX"})
        assert q.prefix == "prefix"

    def test_prefix_whitespace_stripped(self):
        q = TagsListQuery.model_validate({"prefix": "  prefix  "})
        assert q.prefix == "prefix"

    def test_prefix_whitespace_only_fails_min_length(self):
        # After stripping, whitespace-only prefix becomes empty, which fails min_length=1
        # The min_length check happens before the normalizer can return None
        with pytest.raises(ValidationError):
            TagsListQuery.model_validate({"prefix": "   "})

    def test_prefix_min_length(self):
        with pytest.raises(ValidationError):
            TagsListQuery.model_validate({"prefix": ""})

    def test_prefix_max_length(self):
        with pytest.raises(ValidationError):
            TagsListQuery.model_validate({"prefix": "x" * 257})

    def test_limit_bounds_min(self):
        with pytest.raises(ValidationError):
            TagsListQuery.model_validate({"limit": 0})

    def test_limit_bounds_max(self):
        with pytest.raises(ValidationError):
            TagsListQuery.model_validate({"limit": 1001})

    def test_limit_bounds_valid(self):
        q = TagsListQuery.model_validate({"limit": 1000})
        assert q.limit == 1000

    def test_offset_bounds_min(self):
        with pytest.raises(ValidationError):
            TagsListQuery.model_validate({"offset": -1})

    def test_offset_bounds_max(self):
        with pytest.raises(ValidationError):
            TagsListQuery.model_validate({"offset": 10_000_001})

    def test_order_valid_values(self):
        for order_val in ["count_desc", "name_asc"]:
            q = TagsListQuery.model_validate({"order": order_val})
            assert q.order == order_val

    def test_order_invalid(self):
        with pytest.raises(ValidationError):
            TagsListQuery.model_validate({"order": "invalid"})

    def test_include_zero_bool(self):
        q = TagsListQuery.model_validate({"include_zero": False})
        assert q.include_zero is False


class TestScheduleAssetScanBody:
    def test_valid_roots(self):
        body = ScheduleAssetScanBody.model_validate({"roots": ["models"]})
        assert body.roots == ["models"]

    def test_multiple_roots(self):
        body = ScheduleAssetScanBody.model_validate({"roots": ["models", "input", "output"]})
        assert body.roots == ["models", "input", "output"]

    def test_empty_roots_rejected(self):
        with pytest.raises(ValidationError):
            ScheduleAssetScanBody.model_validate({"roots": []})

    def test_invalid_root_rejected(self):
        with pytest.raises(ValidationError):
            ScheduleAssetScanBody.model_validate({"roots": ["invalid"]})

    def test_missing_roots_rejected(self):
        with pytest.raises(ValidationError):
            ScheduleAssetScanBody.model_validate({})
