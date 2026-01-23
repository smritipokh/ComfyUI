"""
Tests for read and update endpoints in the assets API.
"""

import pytest
import uuid
from aiohttp import FormData
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.asyncio


def make_mock_asset(asset_id=None, name="Test Asset", tags=None, user_metadata=None, preview_id=None):
    """Helper to create a mock asset result."""
    if asset_id is None:
        asset_id = str(uuid.uuid4())
    if tags is None:
        tags = ["input"]
    if user_metadata is None:
        user_metadata = {}

    mock = MagicMock()
    mock.model_dump.return_value = {
        "id": asset_id,
        "name": name,
        "tags": tags,
        "user_metadata": user_metadata,
        "preview_id": preview_id,
    }
    return mock


def make_mock_list_result(assets, total=None):
    """Helper to create a mock list result."""
    if total is None:
        total = len(assets)
    mock = MagicMock()
    mock.model_dump.return_value = {
        "assets": [a.model_dump() if hasattr(a, 'model_dump') else a for a in assets],
        "total": total,
    }
    return mock


class TestListAssets:
    async def test_returns_list(self, aiohttp_client, app):
        with patch("app.assets.manager.list_assets") as mock_list:
            mock_list.return_value = make_mock_list_result([
                {"id": str(uuid.uuid4()), "name": "Asset 1", "tags": ["input"]},
            ], total=1)

            client = await aiohttp_client(app)
            resp = await client.get('/api/assets')
            assert resp.status == 200
            body = await resp.json()
            assert 'assets' in body
            assert 'total' in body
            assert body['total'] == 1

    async def test_returns_list_with_pagination(self, aiohttp_client, app):
        with patch("app.assets.manager.list_assets") as mock_list:
            mock_list.return_value = make_mock_list_result([
                {"id": str(uuid.uuid4()), "name": "Asset 1", "tags": ["input"]},
                {"id": str(uuid.uuid4()), "name": "Asset 2", "tags": ["input"]},
            ], total=5)

            client = await aiohttp_client(app)
            resp = await client.get('/api/assets?limit=2&offset=0')
            assert resp.status == 200
            body = await resp.json()
            assert len(body['assets']) == 2
            assert body['total'] == 5
            mock_list.assert_called_once()
            call_kwargs = mock_list.call_args.kwargs
            assert call_kwargs['limit'] == 2
            assert call_kwargs['offset'] == 0

    async def test_filter_by_include_tags(self, aiohttp_client, app):
        with patch("app.assets.manager.list_assets") as mock_list:
            mock_list.return_value = make_mock_list_result([
                {"id": str(uuid.uuid4()), "name": "Special Asset", "tags": ["special"]},
            ], total=1)

            client = await aiohttp_client(app)
            resp = await client.get('/api/assets?include_tags=special')
            assert resp.status == 200
            body = await resp.json()
            for asset in body['assets']:
                assert 'special' in asset.get('tags', [])
            mock_list.assert_called_once()
            call_kwargs = mock_list.call_args.kwargs
            assert 'special' in call_kwargs['include_tags']

    async def test_filter_by_exclude_tags(self, aiohttp_client, app):
        with patch("app.assets.manager.list_assets") as mock_list:
            mock_list.return_value = make_mock_list_result([
                {"id": str(uuid.uuid4()), "name": "Kept Asset", "tags": ["keep"]},
            ], total=1)

            client = await aiohttp_client(app)
            resp = await client.get('/api/assets?exclude_tags=exclude_me')
            assert resp.status == 200
            body = await resp.json()
            for asset in body['assets']:
                assert 'exclude_me' not in asset.get('tags', [])
            mock_list.assert_called_once()
            call_kwargs = mock_list.call_args.kwargs
            assert 'exclude_me' in call_kwargs['exclude_tags']

    async def test_filter_by_name_contains(self, aiohttp_client, app):
        with patch("app.assets.manager.list_assets") as mock_list:
            mock_list.return_value = make_mock_list_result([
                {"id": str(uuid.uuid4()), "name": "UniqueSearchName", "tags": ["input"]},
            ], total=1)

            client = await aiohttp_client(app)
            resp = await client.get('/api/assets?name_contains=UniqueSearch')
            assert resp.status == 200
            body = await resp.json()
            for asset in body['assets']:
                assert 'UniqueSearch' in asset.get('name', '')
            mock_list.assert_called_once()
            call_kwargs = mock_list.call_args.kwargs
            assert call_kwargs['name_contains'] == 'UniqueSearch'

    async def test_sort_and_order(self, aiohttp_client, app):
        with patch("app.assets.manager.list_assets") as mock_list:
            mock_list.return_value = make_mock_list_result([
                {"id": str(uuid.uuid4()), "name": "Alpha", "tags": ["input"]},
                {"id": str(uuid.uuid4()), "name": "Zeta", "tags": ["input"]},
            ], total=2)

            client = await aiohttp_client(app)
            resp = await client.get('/api/assets?sort=name&order=asc')
            assert resp.status == 200
            mock_list.assert_called_once()
            call_kwargs = mock_list.call_args.kwargs
            assert call_kwargs['sort'] == 'name'
            assert call_kwargs['order'] == 'asc'


class TestGetAssetById:
    async def test_returns_asset(self, aiohttp_client, app):
        asset_id = str(uuid.uuid4())
        with patch("app.assets.manager.get_asset") as mock_get:
            mock_get.return_value = make_mock_asset(asset_id=asset_id, name="Test Asset")

            client = await aiohttp_client(app)
            resp = await client.get(f'/api/assets/{asset_id}')
            assert resp.status == 200
            body = await resp.json()
            assert body['id'] == asset_id

    async def test_returns_404_for_missing_id(self, aiohttp_client, app):
        fake_id = str(uuid.uuid4())
        with patch("app.assets.manager.get_asset") as mock_get:
            mock_get.side_effect = ValueError("Asset not found")

            client = await aiohttp_client(app)
            resp = await client.get(f'/api/assets/{fake_id}')
            assert resp.status == 404
            body = await resp.json()
            assert body['error']['code'] == 'ASSET_NOT_FOUND'

    async def test_returns_404_for_wrong_owner(self, aiohttp_client, app):
        asset_id = str(uuid.uuid4())
        with patch("app.assets.manager.get_asset") as mock_get:
            mock_get.side_effect = ValueError("Asset not found for this owner")

            client = await aiohttp_client(app)
            resp = await client.get(f'/api/assets/{asset_id}')
            assert resp.status == 404
            body = await resp.json()
            assert body['error']['code'] == 'ASSET_NOT_FOUND'


class TestDownloadAssetContent:
    async def test_returns_file_content(self, aiohttp_client, app, test_image_bytes, tmp_path):
        asset_id = str(uuid.uuid4())
        test_file = tmp_path / "test_image.png"
        test_file.write_bytes(test_image_bytes)

        with patch("app.assets.manager.resolve_asset_content_for_download") as mock_resolve:
            mock_resolve.return_value = (str(test_file), "image/png", "test_image.png")

            client = await aiohttp_client(app)
            resp = await client.get(f'/api/assets/{asset_id}/content')
            assert resp.status == 200
            assert 'image' in resp.content_type

    async def test_sets_content_disposition_header(self, aiohttp_client, app, test_image_bytes, tmp_path):
        asset_id = str(uuid.uuid4())
        test_file = tmp_path / "test_image.png"
        test_file.write_bytes(test_image_bytes)

        with patch("app.assets.manager.resolve_asset_content_for_download") as mock_resolve:
            mock_resolve.return_value = (str(test_file), "image/png", "test_image.png")

            client = await aiohttp_client(app)
            resp = await client.get(f'/api/assets/{asset_id}/content')
            assert resp.status == 200
            assert 'Content-Disposition' in resp.headers
            assert 'test_image.png' in resp.headers['Content-Disposition']

    async def test_returns_404_for_missing_asset(self, aiohttp_client, app):
        fake_id = str(uuid.uuid4())
        with patch("app.assets.manager.resolve_asset_content_for_download") as mock_resolve:
            mock_resolve.side_effect = ValueError("Asset not found")

            client = await aiohttp_client(app)
            resp = await client.get(f'/api/assets/{fake_id}/content')
            assert resp.status == 404
            body = await resp.json()
            assert body['error']['code'] == 'ASSET_NOT_FOUND'

    async def test_returns_404_for_missing_file(self, aiohttp_client, app):
        asset_id = str(uuid.uuid4())
        with patch("app.assets.manager.resolve_asset_content_for_download") as mock_resolve:
            mock_resolve.side_effect = FileNotFoundError("File not found on disk")

            client = await aiohttp_client(app)
            resp = await client.get(f'/api/assets/{asset_id}/content')
            assert resp.status == 404
            body = await resp.json()
            assert body['error']['code'] == 'FILE_NOT_FOUND'


class TestUpdateAsset:
    async def test_update_name(self, aiohttp_client, app):
        asset_id = str(uuid.uuid4())
        with patch("app.assets.manager.update_asset") as mock_update:
            mock_update.return_value = make_mock_asset(asset_id=asset_id, name="New Name")

            client = await aiohttp_client(app)
            resp = await client.put(f'/api/assets/{asset_id}', json={'name': 'New Name'})
            assert resp.status == 200
            body = await resp.json()
            assert body['name'] == 'New Name'
            mock_update.assert_called_once()
            call_kwargs = mock_update.call_args.kwargs
            assert call_kwargs['name'] == 'New Name'

    async def test_update_tags(self, aiohttp_client, app):
        asset_id = str(uuid.uuid4())
        with patch("app.assets.manager.update_asset") as mock_update:
            mock_update.return_value = make_mock_asset(
                asset_id=asset_id, tags=['new_tag', 'another_tag']
            )

            client = await aiohttp_client(app)
            resp = await client.put(f'/api/assets/{asset_id}', json={'tags': ['new_tag', 'another_tag']})
            assert resp.status == 200
            body = await resp.json()
            assert 'new_tag' in body.get('tags', [])
            assert 'another_tag' in body.get('tags', [])
            mock_update.assert_called_once()
            call_kwargs = mock_update.call_args.kwargs
            assert call_kwargs['tags'] == ['new_tag', 'another_tag']

    async def test_update_user_metadata(self, aiohttp_client, app):
        asset_id = str(uuid.uuid4())
        with patch("app.assets.manager.update_asset") as mock_update:
            mock_update.return_value = make_mock_asset(
                asset_id=asset_id, user_metadata={'key': 'value'}
            )

            client = await aiohttp_client(app)
            resp = await client.put(f'/api/assets/{asset_id}', json={'user_metadata': {'key': 'value'}})
            assert resp.status == 200
            body = await resp.json()
            assert body.get('user_metadata', {}).get('key') == 'value'
            mock_update.assert_called_once()
            call_kwargs = mock_update.call_args.kwargs
            assert call_kwargs['user_metadata'] == {'key': 'value'}

    async def test_returns_400_on_empty_body(self, aiohttp_client, app):
        asset_id = str(uuid.uuid4())

        client = await aiohttp_client(app)
        resp = await client.put(f'/api/assets/{asset_id}', data=b'')
        assert resp.status == 400
        body = await resp.json()
        assert body['error']['code'] == 'INVALID_JSON'

    async def test_returns_404_for_missing_asset(self, aiohttp_client, app):
        fake_id = str(uuid.uuid4())
        with patch("app.assets.manager.update_asset") as mock_update:
            mock_update.side_effect = ValueError("Asset not found")

            client = await aiohttp_client(app)
            resp = await client.put(f'/api/assets/{fake_id}', json={'name': 'New Name'})
            assert resp.status == 404
            body = await resp.json()
            assert body['error']['code'] == 'ASSET_NOT_FOUND'


class TestSetAssetPreview:
    async def test_sets_preview_id(self, aiohttp_client, app):
        asset_id = str(uuid.uuid4())
        preview_id = str(uuid.uuid4())
        with patch("app.assets.manager.set_asset_preview") as mock_set_preview:
            mock_set_preview.return_value = make_mock_asset(
                asset_id=asset_id, preview_id=preview_id
            )

            client = await aiohttp_client(app)
            resp = await client.put(f'/api/assets/{asset_id}/preview', json={'preview_id': preview_id})
            assert resp.status == 200
            body = await resp.json()
            assert body.get('preview_id') == preview_id
            mock_set_preview.assert_called_once()
            call_kwargs = mock_set_preview.call_args.kwargs
            assert call_kwargs['preview_asset_id'] == preview_id

    async def test_clears_preview_with_null(self, aiohttp_client, app):
        asset_id = str(uuid.uuid4())
        with patch("app.assets.manager.set_asset_preview") as mock_set_preview:
            mock_set_preview.return_value = make_mock_asset(
                asset_id=asset_id, preview_id=None
            )

            client = await aiohttp_client(app)
            resp = await client.put(f'/api/assets/{asset_id}/preview', json={'preview_id': None})
            assert resp.status == 200
            body = await resp.json()
            assert body.get('preview_id') is None
            mock_set_preview.assert_called_once()
            call_kwargs = mock_set_preview.call_args.kwargs
            assert call_kwargs['preview_asset_id'] is None

    async def test_returns_404_for_missing_asset(self, aiohttp_client, app):
        fake_id = str(uuid.uuid4())
        with patch("app.assets.manager.set_asset_preview") as mock_set_preview:
            mock_set_preview.side_effect = ValueError("Asset not found")

            client = await aiohttp_client(app)
            resp = await client.put(f'/api/assets/{fake_id}/preview', json={'preview_id': None})
            assert resp.status == 404
            body = await resp.json()
            assert body['error']['code'] == 'ASSET_NOT_FOUND'
