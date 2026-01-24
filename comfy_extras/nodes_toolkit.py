from __future__ import annotations
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_api.latest import _io


class ConvertToList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template_matchtype = io.MatchType.Template("type")
        template_autogrow = _io.Autogrow.TemplatePrefix(
            input=io.MatchType.Input("input", template=template_matchtype),
            prefix="input",
        )
        return io.Schema(
            node_id="ConvertToList",
            display_name="Convert to List",
            category="logic",
            is_input_list=True,
            inputs=[_io.Autogrow.Input("inputs", template=template_autogrow)],
            outputs=[
                io.MatchType.Output(template=template_matchtype, is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, inputs: _io.Autogrow.Type) -> io.NodeOutput:
        output_list = []
        for input in inputs.values():
            output_list += input
        return io.NodeOutput(output_list)


class ToolkitExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ConvertToList,
        ]


async def comfy_entrypoint() -> ToolkitExtension:
    return ToolkitExtension()
