from __future__ import annotations

import os
import subprocess
import time
from collections import deque
from pathlib import Path

import imageio
import librosa
import numpy as np
from loguru import logger

from flashhead_server.contracts import AudioEncodeMode, GenerationRequest


class FlashHeadProvider:
    """Wraps the flash_head inference pipeline as a provider.

    The model is loaded once at construction time. All subsequent calls to
    generate() reuse the same pipeline object — model weights stay in GPU RAM.

    Must be constructed after os.chdir() to the SoulX-FlashHead root so that
    flash_head/configs/infer_params.yaml is discoverable by relative path.
    """

    provider_name = "flashhead"

    def __init__(self, ckpt_dir: str, wav2vec_dir: str, model_mode: str) -> None:
        if not ckpt_dir:
            raise ValueError("FLASHHEAD_CKPT_DIR must be set")
        if not wav2vec_dir:
            raise ValueError("FLASHHEAD_WAV2VEC_DIR must be set")
        if model_mode not in ("lite", "pro"):
            raise ValueError(f"model_mode must be 'lite' or 'pro', got '{model_mode}'")

        self._model_mode = model_mode
        self._ckpt_dir = ckpt_dir
        self._wav2vec_dir = wav2vec_dir
        self._ready = False
        self._pipeline = None

        self._load_model()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        from flash_head.inference import get_pipeline
        logger.info("Loading FlashHead pipeline (mode={}, ckpt={})", self._model_mode, self._ckpt_dir)
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        self._pipeline = get_pipeline(
            world_size=world_size,
            ckpt_dir=self._ckpt_dir,
            wav2vec_dir=self._wav2vec_dir,
            model_type=self._model_mode,
        )
        self._ready = True
        logger.info("FlashHead pipeline loaded (mode={})", self._model_mode)

    # ── Protocol ─────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        return self._ready and self._pipeline is not None

    def supported_modes(self) -> list[str]:
        return [self._model_mode]

    def generate(self, request: GenerationRequest, output_path: Path) -> None:
        from flash_head.inference import (
            get_audio_embedding,
            get_base_data,
            get_infer_params,
            run_pipeline,
        )

        if not self._ready:
            raise RuntimeError("FlashHead pipeline is not loaded")

        pipeline = self._pipeline
        infer_params = get_infer_params()

        get_base_data(
            pipeline,
            cond_image_path_or_dir=request.face_image_path,
            base_seed=request.base_seed,
            use_face_crop=request.use_face_crop,
        )

        sample_rate = infer_params["sample_rate"]
        tgt_fps = infer_params["tgt_fps"]
        frame_num = infer_params["frame_num"]
        motion_frames_num = infer_params["motion_frames_num"]
        cached_audio_duration = infer_params["cached_audio_duration"]
        slice_len = frame_num - motion_frames_num

        audio_array, _ = librosa.load(request.audio_path, sr=sample_rate, mono=True)
        human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
        human_speech_array_frame_num = frame_num * sample_rate // tgt_fps

        generated_list: list = []

        if request.audio_encode_mode == AudioEncodeMode.once:
            generated_list = self._run_once(
                pipeline,
                audio_array,
                human_speech_array_frame_num,
                human_speech_array_slice_len,
                frame_num,
                motion_frames_num,
                get_audio_embedding,
                run_pipeline,
            )
        else:
            generated_list = self._run_stream(
                pipeline,
                audio_array,
                human_speech_array_slice_len,
                sample_rate,
                cached_audio_duration,
                tgt_fps,
                frame_num,
                motion_frames_num,
                get_audio_embedding,
                run_pipeline,
            )

        self._save_video(generated_list, str(output_path), request.audio_path, tgt_fps)

    # ── Inference loops ───────────────────────────────────────────────────────

    def _run_once(
        self,
        pipeline,
        audio_array,
        frame_num_samples,
        slice_len_samples,
        frame_num,
        motion_frames_num,
        get_audio_embedding,
        run_pipeline,
    ) -> list:
        remainder = (len(audio_array) - frame_num_samples) % slice_len_samples
        if remainder > 0:
            pad = slice_len_samples - remainder
            audio_array = np.concatenate([audio_array, np.zeros(pad, dtype=audio_array.dtype)])

        audio_embedding_all = get_audio_embedding(pipeline, audio_array)
        total_chunks = (audio_embedding_all.shape[1] - frame_num) // (frame_num - motion_frames_num)
        chunks = [
            audio_embedding_all[:, i * (frame_num - motion_frames_num): i * (frame_num - motion_frames_num) + frame_num].contiguous()
            for i in range(total_chunks)
        ]

        generated = []
        for idx, chunk in enumerate(chunks):
            t0 = time.time()
            video = run_pipeline(pipeline, chunk)
            if idx != 0:
                video = video[motion_frames_num:]
            logger.info("Chunk {} done in {:.2f}s", idx, time.time() - t0)
            generated.append(video.cpu())
        return generated

    def _run_stream(
        self,
        pipeline,
        audio_array,
        slice_len_samples,
        sample_rate,
        cached_audio_duration,
        tgt_fps,
        frame_num,
        motion_frames_num,
        get_audio_embedding,
        run_pipeline,
    ) -> list:
        cached_audio_length = sample_rate * cached_audio_duration
        audio_end_idx = cached_audio_duration * tgt_fps
        audio_start_idx = audio_end_idx - frame_num

        audio_dq: deque = deque([0.0] * cached_audio_length, maxlen=cached_audio_length)

        remainder = len(audio_array) % slice_len_samples
        if remainder > 0:
            pad = slice_len_samples - remainder
            audio_array = np.concatenate([audio_array, np.zeros(pad, dtype=audio_array.dtype)])

        slices = audio_array.reshape(-1, slice_len_samples)
        generated = []
        for idx, chunk_arr in enumerate(slices):
            t0 = time.time()
            audio_dq.extend(chunk_arr.tolist())
            embedding = get_audio_embedding(pipeline, np.array(audio_dq), audio_start_idx, audio_end_idx)
            video = run_pipeline(pipeline, embedding)
            video = video[motion_frames_num:]
            logger.info("Chunk {} done in {:.2f}s", idx, time.time() - t0)
            generated.append(video.cpu())
        return generated

    # ── Video save ────────────────────────────────────────────────────────────

    @staticmethod
    def _save_video(frames_list: list, video_path: str, audio_path: str, fps: int) -> None:
        tmp_path = video_path.replace(".mp4", "_tmp.mp4")
        with imageio.get_writer(
            tmp_path, format="mp4", mode="I", fps=fps, codec="h264",
            ffmpeg_params=["-bf", "0"],
        ) as writer:
            for frames in frames_list:
                frames_np = frames.numpy().astype(np.uint8)
                for i in range(frames_np.shape[0]):
                    writer.append_data(frames_np[i])

        cmd = [
            "ffmpeg", "-i", tmp_path, "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            video_path, "-y",
        ]
        result = subprocess.run(cmd, capture_output=True)
        os.remove(tmp_path)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg muxing failed: {result.stderr.decode()}")
