from __future__ import annotations

import os
import av
import torch
import folder_paths
import json
from typing import Optional
from typing_extensions import override
from fractions import Fraction
from comfy_api.latest import ComfyExtension, io, ui, Input, InputImpl, Types
from comfy.cli_args import args

class SaveWEBM(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveWEBM",
            category="image/video",
            is_experimental=True,
            inputs=[
                io.Image.Input("images"),
                io.String.Input("filename_prefix", default="ComfyUI"),
                io.Combo.Input("codec", options=["vp9", "av1"]),
                io.Float.Input("fps", default=24.0, min=0.01, max=1000.0, step=0.01),
                io.Float.Input("crf", default=32.0, min=0, max=63.0, step=1, tooltip="Higher crf means lower quality with a smaller file size, lower crf means higher quality higher filesize."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, codec, fps, filename_prefix, crf) -> io.NodeOutput:
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), images[0].shape[1], images[0].shape[0]
        )

        file = f"{filename}_{counter:05}_.webm"
        container = av.open(os.path.join(full_output_folder, file), mode="w")

        if cls.hidden.prompt is not None:
            container.metadata["prompt"] = json.dumps(cls.hidden.prompt)

        if cls.hidden.extra_pnginfo is not None:
            for x in cls.hidden.extra_pnginfo:
                container.metadata[x] = json.dumps(cls.hidden.extra_pnginfo[x])

        codec_map = {"vp9": "libvpx-vp9", "av1": "libsvtav1"}
        stream = container.add_stream(codec_map[codec], rate=Fraction(round(fps * 1000), 1000))
        stream.width = images.shape[-2]
        stream.height = images.shape[-3]
        stream.pix_fmt = "yuv420p10le" if codec == "av1" else "yuv420p"
        stream.bit_rate = 0
        stream.options = {'crf': str(crf)}
        if codec == "av1":
            stream.options["preset"] = "6"

        for frame in images:
            frame = av.VideoFrame.from_ndarray(torch.clamp(frame[..., :3] * 255, min=0, max=255).to(device=torch.device("cpu"), dtype=torch.uint8).numpy(), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        container.mux(stream.encode())
        container.close()

        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))

class SaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        # H264-specific inputs
        h264_quality = io.Int.Input(
            "quality",
            default=80,
            min=0,
            max=100,
            step=1,
            display_name="Quality",
            tooltip="Output quality (0-100). Higher = better quality, larger files. "
                    "Internally maps to CRF: 100→CRF 12, 50→CRF 23, 0→CRF 40.",
        )
        h264_speed = io.Combo.Input(
            "speed",
            options=Types.VideoSpeedPreset.as_input(),
            default="auto",
            display_name="Encoding Speed",
            tooltip="Encoding speed preset. Slower = better compression at same quality. "
                    "Maps to FFmpeg presets: Fastest=ultrafast, Balanced=medium, Best=veryslow.",
        )
        h264_profile = io.Combo.Input(
            "profile",
            options=["auto", "baseline", "main", "high"],
            default="auto",
            display_name="Profile",
            tooltip="H.264 profile. 'baseline' for max compatibility (older devices), "
                    "'main' for standard use, 'high' for best quality/compression.",
            advanced=True,
        )
        h264_tune = io.Combo.Input(
            "tune",
            options=["auto", "film", "animation", "grain", "stillimage", "fastdecode", "zerolatency"],
            default="auto",
            display_name="Tune",
            tooltip="Optimize encoding for specific content types. "
                    "'film' for live action, 'animation' for cartoons/anime, 'grain' to preserve film grain.",
            advanced=True,
        )

        # VP9-specific inputs
        vp9_quality = io.Int.Input(
            "quality",
            default=80,
            min=0,
            max=100,
            step=1,
            display_name="Quality",
            tooltip="Output quality (0-100). Higher = better quality, larger files. "
                    "Internally maps to CRF: 100→CRF 15, 50→CRF 33, 0→CRF 50.",
        )
        vp9_speed = io.Combo.Input(
            "speed",
            options=Types.VideoSpeedPreset.as_input(),
            default="auto",
            display_name="Encoding Speed",
            tooltip="Encoding speed. Slower = better compression. "
                    "Maps to VP9 cpu-used: Fastest=0, Balanced=2, Best=4.",
        )
        vp9_row_mt = io.Boolean.Input(
            "row_mt",
            default=True,
            display_name="Row Multi-threading",
            tooltip="Enable row-based multi-threading for faster encoding on multi-core CPUs.",
            advanced=True,
        )
        vp9_tile_columns = io.Combo.Input(
            "tile_columns",
            options=["auto", "0", "1", "2", "3", "4"],
            default="auto",
            display_name="Tile Columns",
            tooltip="Number of tile columns (as power of 2). More tiles = faster encoding "
                    "but slightly worse compression. 'auto' picks based on resolution.",
            advanced=True,
        )

        return io.Schema(
            node_id="SaveVideo",
            display_name="Save Video",
            category="image/video",
            description="Saves video to the output directory. "
                        "When format/codec/quality differ from source, the video is re-encoded.",
            inputs=[
                io.Video.Input("video", tooltip="The video to save."),
                io.String.Input(
                    "filename_prefix",
                    default="video/ComfyUI",
                    tooltip="The prefix for the file to save. "
                            "Supports formatting like %date:yyyy-MM-dd%.",
                ),
                io.DynamicCombo.Input("codec", options=[
                    io.DynamicCombo.Option("auto", []),
                    io.DynamicCombo.Option("h264", [h264_quality, h264_speed, h264_profile, h264_tune]),
                    io.DynamicCombo.Option("vp9", [vp9_quality, vp9_speed, vp9_row_mt, vp9_tile_columns]),
                ], tooltip="Video codec. 'auto' preserves source when possible. "
                           "h264 outputs MP4, vp9 outputs WebM."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video: Input.Video, filename_prefix: str, codec: dict) -> io.NodeOutput:
        selected_codec = codec.get("codec", "auto")
        quality = codec.get("quality")
        speed_str = codec.get("speed", "auto")

        # H264-specific options
        profile = codec.get("profile", "auto")
        tune = codec.get("tune", "auto")

        # VP9-specific options
        row_mt = codec.get("row_mt", True)
        tile_columns = codec.get("tile_columns", "auto")

        if selected_codec == "auto":
            resolved_format = Types.VideoContainer.AUTO
            resolved_codec = Types.VideoCodec.AUTO
        elif selected_codec == "h264":
            resolved_format = Types.VideoContainer.MP4
            resolved_codec = Types.VideoCodec.H264
        elif selected_codec == "vp9":
            resolved_format = Types.VideoContainer.WEBM
            resolved_codec = Types.VideoCodec.VP9
        else:
            resolved_format = Types.VideoContainer.AUTO
            resolved_codec = Types.VideoCodec.AUTO

        speed = None
        if speed_str:
            try:
                speed = Types.VideoSpeedPreset(speed_str)
            except (ValueError, TypeError):
                logging.warning(f"Invalid speed preset '{speed_str}', using default")

        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height
        )

        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
            if len(metadata) > 0:
                saved_metadata = metadata

        extension = Types.VideoContainer.get_extension(resolved_format)
        file = f"{filename}_{counter:05}_.{extension}"
        video.save_to(
            os.path.join(full_output_folder, file),
            format=resolved_format,
            codec=resolved_codec,
            metadata=saved_metadata,
            quality=quality,
            speed=speed,
            profile=profile if profile != "auto" else None,
            tune=tune if tune != "auto" else None,
            row_mt=row_mt,
            tile_columns=int(tile_columns) if tile_columns != "auto" else None,
        )

        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))


class CreateVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CreateVideo",
            display_name="Create Video",
            category="image/video",
            description="Create a video from images.",
            inputs=[
                io.Image.Input("images", tooltip="The images to create a video from."),
                io.Float.Input("fps", default=30.0, min=1.0, max=120.0, step=1.0),
                io.Audio.Input("audio", optional=True, tooltip="The audio to add to the video."),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, images: Input.Image, fps: float, audio: Optional[Input.Audio] = None) -> io.NodeOutput:
        return io.NodeOutput(
            InputImpl.VideoFromComponents(Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps)))
        )

class GetVideoComponents(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GetVideoComponents",
            display_name="Get Video Components",
            category="image/video",
            description="Extracts all components from a video: frames, audio, and framerate.",
            inputs=[
                io.Video.Input("video", tooltip="The video to extract components from."),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
                io.Float.Output(display_name="fps"),
            ],
        )

    @classmethod
    def execute(cls, video: Input.Video) -> io.NodeOutput:
        components = video.get_components()
        return io.NodeOutput(components.images, components.audio, float(components.frame_rate))


class LoadVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return io.Schema(
            node_id="LoadVideo",
            display_name="Load Video",
            category="image/video",
            inputs=[
                io.Combo.Input("file", options=sorted(files), upload=io.UploadType.video),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(cls, file) -> io.NodeOutput:
        video_path = folder_paths.get_annotated_filepath(file)
        return io.NodeOutput(InputImpl.VideoFromFile(video_path))

    @classmethod
    def fingerprint_inputs(s, file):
        video_path = folder_paths.get_annotated_filepath(file)
        mod_time = os.path.getmtime(video_path)
        # Instead of hashing the file, we can just use the modification time to avoid
        # rehashing large files.
        return mod_time

    @classmethod
    def validate_inputs(s, file):
        if not folder_paths.exists_annotated_filepath(file):
            return "Invalid video file: {}".format(file)

        return True


class VideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SaveWEBM,
            SaveVideo,
            CreateVideo,
            GetVideoComponents,
            LoadVideo,
        ]

async def comfy_entrypoint() -> VideoExtension:
    return VideoExtension()
