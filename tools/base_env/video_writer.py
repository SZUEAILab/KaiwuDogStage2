"""Per-environment video writer using TiledCamera frames.

每个环境独立的视频写入器，从 TiledCamera sensor 读取帧并写入 MP4 文件。
仅在评估模式且 num_envs <= 16 时启用。
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import logging


class TiledCameraVideoWriter:
    """Manages per-env video writers for TiledCamera output.

    Usage:
        writer = TiledCameraVideoWriter(video_dir="/path/to/mp4", fps=30)
        # Each step:
        writer.write_frames(rgb_tensor)   # (num_envs, H, W, 3) uint8 on GPU
        # On close:
        writer.close()
    """

    def __init__(
        self,
        video_dir: str,
        fps: float = 30.0,
        logger: logging.Logger | None = None,
        env_terrain_names: list[str] | None = None,
    ):
        """Initialize the video writer manager.

        Args:
            video_dir: Directory to save per-env MP4 files.
            fps: Video frame rate.
            logger: Optional logger instance.
            env_terrain_names: 可选，每个 env 对应的地形名（长度应等于 num_envs）。
                如提供，文件名将变为 ``env_{id:03d}_{terrain}.mp4``，便于肉眼识别每段视频
                录制的是哪种地形；若为 None 或长度不匹配，则回退到 ``env_{id:03d}.mp4``。
        """
        self._video_dir = video_dir
        self._fps = fps
        self._logger = logger
        self._env_terrain_names = env_terrain_names
        self._writers: list | None = None
        self._writer_paths: list[str] | None = None
        self._frame_counts: list[int] | None = None
        self._num_envs: int = 0
        self._frame_size: tuple[int, int] = (0, 0)

        os.makedirs(video_dir, exist_ok=True)

    @staticmethod
    def _sanitize_terrain_name(name: str) -> str:
        """把地形名清洗成适合放进文件名的形式（去空格/斜杠/奇怪字符）。"""
        if not isinstance(name, str) or not name:
            return "unknown"
        safe = []
        for ch in name:
            if ch.isalnum() or ch in ("_", "-"):
                safe.append(ch)
            else:
                safe.append("_")
        return "".join(safe)

    def _build_video_path(self, env_id: int) -> str:
        """Build the MP4 path for a single env, appending terrain/level label for readability.

        File naming convention: ``env_{id:03d}_{label}.mp4`` (e.g. ``env_004_L1_pyramid_slope.mp4``).
        Placing ``env_{id:03d}`` first ensures natural alphabetical ordering in file managers,
        while the suffix label (level + terrain) makes each clip identifiable at a glance.

        为单个 env 生成 mp4 路径，后缀带上地形/level 标签便于肉眼识别。
        命名规范：``env_{id:03d}_{label}.mp4``（如 ``env_004_L1_pyramid_slope.mp4``）。
        env 编号放在最前能保证文件管理器按字典序天然有序，后缀（level + 地形）让每段视频一眼可辨。
        """
        filename = f"env_{env_id:03d}.mp4"
        names = self._env_terrain_names
        if names is not None and 0 <= env_id < len(names):
            terrain = self._sanitize_terrain_name(str(names[env_id]))
            if terrain:
                filename = f"env_{env_id:03d}_{terrain}.mp4"
        return os.path.join(self._video_dir, filename)

    def write_frames(self, rgb_tensor: torch.Tensor, skip_mask: torch.Tensor | None = None) -> None:
        """Write one frame per environment.

        Args:
            rgb_tensor: RGB image tensor of shape (num_envs, H, W, 3) or (num_envs, H, W, 4),
                        dtype uint8, on any device (will be moved to CPU).
            skip_mask: Optional bool tensor of shape (num_envs,). If provided,
                       frames for envs where skip_mask[i]=True will NOT be written.
                       Used to skip recording when robot is out of terrain bounds.
        """
        import cv2

        # Move to CPU and convert to numpy
        if rgb_tensor.device.type != "cpu":
            rgb_np = rgb_tensor.cpu().numpy()
        else:
            rgb_np = rgb_tensor.numpy()

        # Handle RGBA → RGB
        if rgb_np.shape[-1] == 4:
            rgb_np = rgb_np[..., :3]

        num_envs, h, w, _ = rgb_np.shape

        # Lazy init writers on first frame
        if self._writers is None:
            self._num_envs = num_envs
            # 优先使用 H.264 (avc1)：VSCode 视频插件 / 企业微信 / 浏览器均原生支持。
            # 旧的 "mp4v" (MPEG-4 Part 2) 在 Chromium 内核播放器中会解码失败或黑屏。
            # H.264 要求宽高必须为偶数，这里强制对齐。
            if (w % 2) or (h % 2):
                w = w - (w % 2)
                h = h - (h % 2)
                if self._logger:
                    self._logger.warning(f"H.264 要求偶数分辨率，已对齐到 {w}x{h}")
            self._frame_size = (w, h)

            codec_candidates = ["avc1", "H264", "h264", "mp4v"]
            chosen_fourcc = None
            self._writers = []
            self._writer_paths = []
            self._frame_counts = []
            for env_id in range(num_envs):
                path = self._build_video_path(env_id)
                writer = None
                for codec in codec_candidates if chosen_fourcc is None else [chosen_fourcc]:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    w_try = cv2.VideoWriter(path, fourcc, self._fps, (w, h))
                    if w_try.isOpened():
                        writer = w_try
                        chosen_fourcc = codec
                        break
                    w_try.release()
                if writer is None or not writer.isOpened():
                    if self._logger:
                        self._logger.warning(f"VideoWriter 无法打开: {path}")
                self._writers.append(writer)
                self._writer_paths.append(path)
                self._frame_counts.append(0)
            if self._logger:
                self._logger.info(
                    f"TiledCamera 视频写入器已初始化: {num_envs} 个环境, "
                    f"分辨率={w}x{h}, fps={self._fps}, codec={chosen_fourcc}, 目录={self._video_dir}"
                )

        # Write each env's frame (skip out-of-bounds envs if mask provided)
        tw, th = self._frame_size
        for env_id in range(min(num_envs, self._num_envs)):
            if skip_mask is not None and skip_mask[env_id]:
                continue
            frame = rgb_np[env_id]
            # 若 writer 因 H.264 偶数对齐裁剪过分辨率，这里同步裁剪
            if frame.shape[0] != th or frame.shape[1] != tw:
                frame = frame[:th, :tw, :]
            # OpenCV expects BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if self._writers[env_id] is not None:
                self._writers[env_id].write(frame_bgr)
                if self._frame_counts is not None:
                    self._frame_counts[env_id] += 1

    def close(self) -> None:
        """Release all video writers.

        关闭后对每个 writer 做空帧检测：若某 env 在整段 episode 中从未写入过帧
        （例如 maze 地形首步即被 skip_mask 屏蔽），产生的 mp4 只包含容器头而无 mdat 帧数据，
        绝大多数播放器都会报 "无法播放/损坏"。

        设计原则：**出现问题时保留现场**，因此这里只打 warning 日志指出哪些文件是空壳，
        **不做删除**。用户拿到 mp4 产物目录后，可以通过文件大小（空壳约 ~258B）
        和日志中的空壳列表定位异常 env，结合 [eval term-debug] 日志分析终止原因。
        """
        if self._writers is None:
            return
        empty_files = []
        for env_id, writer in enumerate(self._writers):
            if writer is not None:
                writer.release()
            # 空帧检测（仅收集，不删除）
            if self._frame_counts is not None and self._writer_paths is not None:
                try:
                    frame_cnt = self._frame_counts[env_id]
                    path = self._writer_paths[env_id]
                    if frame_cnt == 0 and os.path.isfile(path):
                        empty_files.append(path)
                except Exception as e:
                    if self._logger:
                        self._logger.warning(f"空 mp4 检测异常 env={env_id}: {e}")
        if self._logger:
            if empty_files:
                self._logger.warning(
                    f"检测到 {len(empty_files)} 个空壳 mp4（0 帧），已保留用于问题定位。"
                    f"可能是该 env 整段 episode 都被 skip_mask 屏蔽或首步即终止: {empty_files}"
                )
            self._logger.info(f"TiledCamera 视频写入器已关闭: {self._num_envs} 个视频文件")
        self._writers = None
        self._writer_paths = None
        self._frame_counts = None
