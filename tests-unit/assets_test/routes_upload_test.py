"""
Tests for upload and create endpoints in assets API routes.
"""

import pytest
from aiohttp import FormData
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.asyncio


class TestUploadAsset:
    """Tests for POST /api/assets (multipart upload)."""

    async def test_upload_success(self, aiohttp_client, app, test_image_bytes, tmp_upload_dir):
        with patch("app.assets.manager.upload_asset_from_temp_path") as mock_upload:
            mock_result = MagicMock()
            mock_result.created_new = True
            mock_result.model_dump.return_value = {
                "id": "11111111-1111-1111-1111-111111111111",
                "name": "Test Asset",
                "tags": ["input"],
            }
            mock_upload.return_value = mock_result

            client = await aiohttp_client(app)

            data = FormData()
            data.add_field("file", test_image_bytes, filename="test.png", content_type="image/png")
            data.add_field("tags", "input")
            data.add_field("name", "Test Asset")

            resp = await client.post("/api/assets", data=data)
            assert resp.status == 201
            body = await resp.json()
            assert "id" in body
            assert body["name"] == "Test Asset"

    async def test_upload_existing_hash_returns_200(self, aiohttp_client, app, test_image_bytes, tmp_upload_dir):
        with patch("app.assets.manager.asset_exists", return_value=True):
            with patch("app.assets.manager.create_asset_from_hash") as mock_create:
                mock_result = MagicMock()
                mock_result.created_new = False
                mock_result.model_dump.return_value = {
                    "id": "22222222-2222-2222-2222-222222222222",
                    "name": "Existing Asset",
                    "tags": ["input"],
                }
                mock_create.return_value = mock_result

                client = await aiohttp_client(app)

                data = FormData()
                data.add_field("hash", "blake3:" + "a" * 64)
                data.add_field("file", test_image_bytes, filename="test.png", content_type="image/png")
                data.add_field("tags", "input")
                data.add_field("name", "Existing Asset")

                resp = await client.post("/api/assets", data=data)
                assert resp.status == 200
                body = await resp.json()
                assert "id" in body

    async def test_upload_missing_file_returns_400(self, aiohttp_client, app):
        client = await aiohttp_client(app)

        data = FormData()
        data.add_field("tags", "input")
        data.add_field("name", "No File Asset")

        resp = await client.post("/api/assets", data=data)
        assert resp.status in (400, 415)

    async def test_upload_empty_file_returns_400(self, aiohttp_client, app, tmp_upload_dir):
        client = await aiohttp_client(app)

        data = FormData()
        data.add_field("file", b"", filename="empty.png", content_type="image/png")
        data.add_field("tags", "input")
        data.add_field("name", "Empty File Asset")

        resp = await client.post("/api/assets", data=data)
        assert resp.status == 400
        body = await resp.json()
        assert body["error"]["code"] == "EMPTY_UPLOAD"

    async def test_upload_invalid_tags_missing_root_returns_400(self, aiohttp_client, app, test_image_bytes, tmp_upload_dir):
        client = await aiohttp_client(app)

        data = FormData()
        data.add_field("file", test_image_bytes, filename="test.png", content_type="image/png")
        data.add_field("tags", "invalid_root_tag")
        data.add_field("name", "Invalid Tags Asset")

        resp = await client.post("/api/assets", data=data)
        assert resp.status == 400
        body = await resp.json()
        assert body["error"]["code"] == "INVALID_BODY"

    async def test_upload_hash_mismatch_returns_400(self, aiohttp_client, app, test_image_bytes, tmp_upload_dir):
        with patch("app.assets.manager.asset_exists", return_value=False):
            with patch("app.assets.manager.upload_asset_from_temp_path") as mock_upload:
                mock_upload.side_effect = ValueError("HASH_MISMATCH")

                client = await aiohttp_client(app)

                data = FormData()
                data.add_field("hash", "blake3:" + "b" * 64)
                data.add_field("file", test_image_bytes, filename="test.png", content_type="image/png")
                data.add_field("tags", "input")
                data.add_field("name", "Hash Mismatch Asset")

                resp = await client.post("/api/assets", data=data)
                assert resp.status == 400
                body = await resp.json()
                assert body["error"]["code"] == "HASH_MISMATCH"

    async def test_upload_non_multipart_returns_415(self, aiohttp_client, app):
        client = await aiohttp_client(app)

        resp = await client.post("/api/assets", json={"name": "test"})
        assert resp.status == 415
        body = await resp.json()
        assert body["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


class TestCreateFromHash:
    """Tests for POST /api/assets/from-hash."""

    async def test_create_from_hash_success(self, aiohttp_client, app):
        with patch("app.assets.manager.create_asset_from_hash") as mock_create:
            mock_result = MagicMock()
            mock_result.model_dump.return_value = {
                "id": "33333333-3333-3333-3333-333333333333",
                "name": "Created From Hash",
                "tags": ["input"],
            }
            mock_create.return_value = mock_result

            client = await aiohttp_client(app)

            resp = await client.post("/api/assets/from-hash", json={
                "hash": "blake3:" + "c" * 64,
                "name": "Created From Hash",
                "tags": ["input"],
            })
            assert resp.status == 201
            body = await resp.json()
            assert body["id"] == "33333333-3333-3333-3333-333333333333"
            assert body["name"] == "Created From Hash"

    async def test_create_from_hash_unknown_hash_returns_404(self, aiohttp_client, app):
        with patch("app.assets.manager.create_asset_from_hash", return_value=None):
            client = await aiohttp_client(app)

            resp = await client.post("/api/assets/from-hash", json={
                "hash": "blake3:" + "d" * 64,
                "name": "Unknown Hash",
                "tags": ["input"],
            })
            assert resp.status == 404
            body = await resp.json()
            assert body["error"]["code"] == "ASSET_NOT_FOUND"

    async def test_create_from_hash_invalid_hash_format_returns_400(self, aiohttp_client, app):
        client = await aiohttp_client(app)

        resp = await client.post("/api/assets/from-hash", json={
            "hash": "invalid_hash_no_colon",
            "name": "Invalid Hash",
            "tags": ["input"],
        })
        assert resp.status == 400
        body = await resp.json()
        assert body["error"]["code"] == "INVALID_BODY"

    async def test_create_from_hash_missing_name_returns_400(self, aiohttp_client, app):
        client = await aiohttp_client(app)

        resp = await client.post("/api/assets/from-hash", json={
            "hash": "blake3:" + "e" * 64,
            "tags": ["input"],
        })
        assert resp.status == 400
        body = await resp.json()
        assert body["error"]["code"] == "INVALID_BODY"

    async def test_create_from_hash_invalid_json_returns_400(self, aiohttp_client, app):
        client = await aiohttp_client(app)

        resp = await client.post(
            "/api/assets/from-hash",
            data="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400
        body = await resp.json()
        assert body["error"]["code"] == "INVALID_JSON"


class TestHeadAssetByHash:
    """Tests for HEAD /api/assets/hash/{hash}."""

    async def test_head_existing_hash_returns_200(self, aiohttp_client, app):
        with patch("app.assets.manager.asset_exists", return_value=True):
            client = await aiohttp_client(app)

            resp = await client.head("/api/assets/hash/blake3:" + "f" * 64)
            assert resp.status == 200

    async def test_head_missing_hash_returns_404(self, aiohttp_client, app):
        with patch("app.assets.manager.asset_exists", return_value=False):
            client = await aiohttp_client(app)

            resp = await client.head("/api/assets/hash/blake3:" + "0" * 64)
            assert resp.status == 404

    async def test_head_invalid_hash_no_colon_returns_400(self, aiohttp_client, app):
        client = await aiohttp_client(app)

        resp = await client.head("/api/assets/hash/invalidhashwithoutcolon")
        assert resp.status == 400

    async def test_head_invalid_hash_wrong_algo_returns_400(self, aiohttp_client, app):
        client = await aiohttp_client(app)

        resp = await client.head("/api/assets/hash/sha256:" + "a" * 64)
        assert resp.status == 400

    async def test_head_invalid_hash_non_hex_returns_400(self, aiohttp_client, app):
        client = await aiohttp_client(app)

        resp = await client.head("/api/assets/hash/blake3:zzzz")
        assert resp.status == 400

    async def test_head_empty_hash_returns_400(self, aiohttp_client, app):
        client = await aiohttp_client(app)

        resp = await client.head("/api/assets/hash/blake3:")
        assert resp.status == 400
