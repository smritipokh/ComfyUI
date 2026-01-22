from __future__ import annotations

from aiohttp import web

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from comfy_api.latest._node_replace import NodeReplace

REGISTERED_NODE_REPLACEMENTS: dict[str, list[NodeReplace]] = {}

def register_node_replacement(node_replace: NodeReplace):
    REGISTERED_NODE_REPLACEMENTS.setdefault(node_replace.old_node_id, []).append(node_replace)

def registered_as_dict():
    return {
        k: [v.as_dict() for v in v_list] for k, v_list in REGISTERED_NODE_REPLACEMENTS.items()
    }

class NodeReplaceManager:
    def add_routes(self, routes):
        @routes.get("/node_replacements")
        async def get_node_replacements(request):
            return web.json_response(registered_as_dict())
