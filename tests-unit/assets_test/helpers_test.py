"""Tests for app.assets.helpers utility functions."""

import os
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

from app.assets.helpers import (
    normalize_tags,
    escape_like_prefix,
    ensure_within_base,
    get_query_dict,
    utcnow,
    project_kv,
    is_scalar,
    fast_asset_file_check,
    list_tree,
    RootType,
    ALLOWED_ROOTS,
)


class TestNormalizeTags:
    def test_lowercases(self):
        assert normalize_tags(["FOO", "Bar"]) == ["foo", "bar"]

    def test_strips_whitespace(self):
        assert normalize_tags(["  hello  ", "world "]) == ["hello", "world"]

    def test_does_not_deduplicate(self):
        result = normalize_tags(["a", "A", "a"])
        assert result == ["a", "a", "a"]

    def test_none_returns_empty(self):
        assert normalize_tags(None) == []

    def test_empty_list_returns_empty(self):
        assert normalize_tags([]) == []

    def test_filters_empty_strings(self):
        assert normalize_tags(["a", "", "  ", "b"]) == ["a", "b"]

    def test_preserves_order(self):
        result = normalize_tags(["Z", "A", "z", "B"])
        assert result == ["z", "a", "z", "b"]


class TestEscapeLikePrefix:
    def test_escapes_percent(self):
        result, esc = escape_like_prefix("50%")
        assert result == "50!%"
        assert esc == "!"

    def test_escapes_underscore(self):
        result, esc = escape_like_prefix("file_name")
        assert result == "file!_name"
        assert esc == "!"

    def test_escapes_escape_char(self):
        result, esc = escape_like_prefix("a!b")
        assert result == "a!!b"
        assert esc == "!"

    def test_normal_string_unchanged(self):
        result, esc = escape_like_prefix("hello")
        assert result == "hello"
        assert esc == "!"

    def test_complex_string(self):
        result, esc = escape_like_prefix("50%_!x")
        assert result == "50!%!_!!x"

    def test_custom_escape_char(self):
        result, esc = escape_like_prefix("50%", escape="\\")
        assert result == "50\\%"
        assert esc == "\\"


class TestEnsureWithinBase:
    def test_valid_path_within_base(self, tmp_path):
        base = str(tmp_path)
        candidate = str(tmp_path / "subdir" / "file.txt")
        ensure_within_base(candidate, base)

    def test_path_traversal_rejected(self, tmp_path):
        base = str(tmp_path / "safe")
        candidate = str(tmp_path / "safe" / ".." / "unsafe")
        with pytest.raises(ValueError, match="escapes base directory|invalid destination"):
            ensure_within_base(candidate, base)

    def test_completely_outside_path_rejected(self, tmp_path):
        base = str(tmp_path / "safe")
        candidate = "/etc/passwd"
        with pytest.raises(ValueError):
            ensure_within_base(candidate, base)

    def test_same_path_is_valid(self, tmp_path):
        base = str(tmp_path)
        ensure_within_base(base, base)


class TestGetQueryDict:
    def test_single_values(self):
        request = MagicMock()
        request.query.keys.return_value = ["a", "b"]
        request.query.get.side_effect = lambda k: {"a": "1", "b": "2"}[k]
        request.query.getall.side_effect = lambda k: [{"a": "1", "b": "2"}[k]]

        result = get_query_dict(request)
        assert result == {"a": "1", "b": "2"}

    def test_multiple_values_same_key(self):
        request = MagicMock()
        request.query.keys.return_value = ["tags"]
        request.query.get.return_value = "tag1"
        request.query.getall.return_value = ["tag1", "tag2", "tag3"]

        result = get_query_dict(request)
        assert result == {"tags": ["tag1", "tag2", "tag3"]}

    def test_empty_query(self):
        request = MagicMock()
        request.query.keys.return_value = []

        result = get_query_dict(request)
        assert result == {}


class TestUtcnow:
    def test_returns_datetime(self):
        result = utcnow()
        assert isinstance(result, datetime)

    def test_no_tzinfo(self):
        result = utcnow()
        assert result.tzinfo is None

    def test_is_approximately_now(self):
        before = datetime.now(timezone.utc).replace(tzinfo=None)
        result = utcnow()
        after = datetime.now(timezone.utc).replace(tzinfo=None)
        assert before <= result <= after


class TestIsScalar:
    def test_none_is_scalar(self):
        assert is_scalar(None) is True

    def test_bool_is_scalar(self):
        assert is_scalar(True) is True
        assert is_scalar(False) is True

    def test_int_is_scalar(self):
        assert is_scalar(42) is True

    def test_float_is_scalar(self):
        assert is_scalar(3.14) is True

    def test_decimal_is_scalar(self):
        assert is_scalar(Decimal("10.5")) is True

    def test_str_is_scalar(self):
        assert is_scalar("hello") is True

    def test_list_is_not_scalar(self):
        assert is_scalar([1, 2, 3]) is False

    def test_dict_is_not_scalar(self):
        assert is_scalar({"a": 1}) is False


class TestProjectKv:
    def test_none_value(self):
        result = project_kv("key", None)
        assert len(result) == 1
        assert result[0]["key"] == "key"
        assert result[0]["ordinal"] == 0
        assert result[0]["val_str"] is None
        assert result[0]["val_num"] is None

    def test_string_value(self):
        result = project_kv("name", "test")
        assert len(result) == 1
        assert result[0]["val_str"] == "test"

    def test_int_value(self):
        result = project_kv("count", 42)
        assert len(result) == 1
        assert result[0]["val_num"] == Decimal("42")

    def test_float_value(self):
        result = project_kv("ratio", 3.14)
        assert len(result) == 1
        assert result[0]["val_num"] == Decimal("3.14")

    def test_bool_value(self):
        result = project_kv("enabled", True)
        assert len(result) == 1
        assert result[0]["val_bool"] is True

    def test_list_of_strings(self):
        result = project_kv("tags", ["a", "b", "c"])
        assert len(result) == 3
        assert result[0]["ordinal"] == 0
        assert result[0]["val_str"] == "a"
        assert result[1]["ordinal"] == 1
        assert result[1]["val_str"] == "b"
        assert result[2]["ordinal"] == 2
        assert result[2]["val_str"] == "c"

    def test_list_of_mixed_scalars(self):
        result = project_kv("mixed", [1, "two", True])
        assert len(result) == 3
        assert result[0]["val_num"] == Decimal("1")
        assert result[1]["val_str"] == "two"
        assert result[2]["val_bool"] is True

    def test_list_with_none(self):
        result = project_kv("items", ["a", None, "b"])
        assert len(result) == 3
        assert result[1]["val_str"] is None
        assert result[1]["val_num"] is None

    def test_dict_value_stored_as_json(self):
        result = project_kv("meta", {"nested": "value"})
        assert len(result) == 1
        assert result[0]["val_json"] == {"nested": "value"}

    def test_list_of_dicts_stored_as_json(self):
        result = project_kv("items", [{"a": 1}, {"b": 2}])
        assert len(result) == 2
        assert result[0]["val_json"] == {"a": 1}
        assert result[1]["val_json"] == {"b": 2}


class TestFastAssetFileCheck:
    def test_none_mtime_returns_false(self):
        stat = MagicMock()
        assert fast_asset_file_check(mtime_db=None, size_db=100, stat_result=stat) is False

    def test_matching_mtime_and_size(self):
        stat = MagicMock()
        stat.st_mtime_ns = 1234567890123456789
        stat.st_size = 100

        result = fast_asset_file_check(
            mtime_db=1234567890123456789,
            size_db=100,
            stat_result=stat
        )
        assert result is True

    def test_mismatched_mtime(self):
        stat = MagicMock()
        stat.st_mtime_ns = 9999999999999999999
        stat.st_size = 100

        result = fast_asset_file_check(
            mtime_db=1234567890123456789,
            size_db=100,
            stat_result=stat
        )
        assert result is False

    def test_mismatched_size(self):
        stat = MagicMock()
        stat.st_mtime_ns = 1234567890123456789
        stat.st_size = 200

        result = fast_asset_file_check(
            mtime_db=1234567890123456789,
            size_db=100,
            stat_result=stat
        )
        assert result is False

    def test_zero_size_skips_size_check(self):
        stat = MagicMock()
        stat.st_mtime_ns = 1234567890123456789
        stat.st_size = 999

        result = fast_asset_file_check(
            mtime_db=1234567890123456789,
            size_db=0,
            stat_result=stat
        )
        assert result is True


class TestListTree:
    def test_lists_files_in_directory(self, tmp_path):
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").touch()

        result = list_tree(str(tmp_path))
        assert len(result) == 3
        assert all(os.path.isabs(p) for p in result)
        assert str(tmp_path / "file1.txt") in result
        assert str(tmp_path / "subdir" / "file3.txt") in result

    def test_nonexistent_directory_returns_empty(self):
        result = list_tree("/nonexistent/path/that/does/not/exist")
        assert result == []


class TestRootType:
    def test_allowed_roots_contains_expected_values(self):
        assert "models" in ALLOWED_ROOTS
        assert "input" in ALLOWED_ROOTS
        assert "output" in ALLOWED_ROOTS

    def test_allowed_roots_is_tuple(self):
        assert isinstance(ALLOWED_ROOTS, tuple)
