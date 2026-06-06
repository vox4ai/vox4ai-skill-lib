import base64
import os
import asyncio
from dataclasses import dataclass, field
from typing import Optional

from tts_plugin_bridge import ConnectorFactory
from tts_plugin_bridge.protocol import (
    TTSRequest,
    TTSConnector,
    ChunkConfig,
)


@dataclass
class TTSResult:
    """TTSSkill の戻り値型。

    旧 `dict` 戻り値の後継。`status` は文字列リテラルに近く扱われるため
    比較的安全だが、`audio_base64` 等の存在を型レベルで示せる。
    """

    status: str
    message: str = ""
    engine: Optional[str] = None
    audio_base64: Optional[str] = None
    extras: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def error(self) -> Optional[str]:
        return self.message if self.status == "error" else None

    @classmethod
    def ok_result(cls, *, engine: str, message: str = "", **extras) -> "TTSResult":
        return cls(status="ok", message=message, engine=engine, extras=extras)

    @classmethod
    def err_result(cls, message: str, *, engine: Optional[str] = None) -> "TTSResult":
        return cls(status="error", message=message, engine=engine)

    def to_legacy_dict(self) -> dict:
        """既存呼び出しコードとの後方互換用。"""
        d = {"status": self.status, "message": self.message}
        if self.engine is not None:
            d["engine"] = self.engine
        if self.audio_base64 is not None:
            d["audio_base64"] = self.audio_base64
        return d


class TTSSkill:
    def __init__(self, default_engine: str = "piperplus", **engine_kwargs):
        self.default_engine = default_engine
        self._cache: dict[str, TTSConnector] = {}
        self._engine_kwargs = engine_kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _get_connector(self, engine: str) -> TTSConnector:
        if engine not in self._cache:
            kwargs = self._engine_kwargs.copy()
            self._cache[engine] = ConnectorFactory.create(engine, **kwargs)
        return self._cache[engine]

    def _filter_extra(
        self, connector: TTSConnector, kwargs: dict
    ) -> dict:
        return {
            k: v for k, v in kwargs.items() if k in connector.get_supported_params()
        }

    def _build_request(
        self,
        text: str,
        speed: float,
        volume: Optional[float],
        model: Optional[str],
        extra: dict,
    ) -> TTSRequest:
        return TTSRequest(
            text=text,
            speed=speed,
            volume=volume,
            model=model,
            extra=extra,
        )

    async def _synthesize_chunks(
        self,
        connector: TTSConnector,
        text: str,
        speed: float,
        volume: Optional[float],
        model: Optional[str],
        extra: dict,
        chunk_config: Optional[ChunkConfig],
    ) -> TTSResult:
        from tts_plugin_bridge.chunker import (
            HybridChunker,
            ChunkConfig as ChunkerConfig,
        )

        config = chunk_config or ChunkerConfig()
        chunks = HybridChunker().chunk(text, config)

        combined = bytearray()
        for c in chunks:
            req = self._build_request(c.text, speed, volume, model, extra)
            res = await connector.synthesize(req)
            if not res.success:
                return TTSResult.err_result(
                    f"Error in chunk {c.index}: {res.error}",
                    engine=connector.name,
                )
            if res.audio_data:
                combined.extend(res.audio_data)

        if not combined:
            return TTSResult.err_result("No audio data was generated", engine=connector.name)

        return TTSResult(
            status="ok",
            message=f"TTS synthesis completed ({len(chunks)} chunks)",
            engine=connector.name,
            audio_base64=base64.b64encode(combined).decode(),
        )

    async def _synthesize_once(
        self,
        connector: TTSConnector,
        text: str,
        speed: float,
        volume: Optional[float],
        model: Optional[str],
        extra: dict,
    ) -> TTSResult:
        req = self._build_request(text, speed, volume, model, extra)
        res = await connector.synthesize(req)
        if not res.success:
            return TTSResult.err_result(res.error, engine=connector.name)
        return TTSResult(
            status="ok",
            message="TTS synthesis completed",
            engine=connector.name,
            audio_base64=base64.b64encode(res.audio_data).decode()
            if res.audio_data
            else None,
        )

    async def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        volume: Optional[float] = None,
        engine: Optional[str] = None,
        chunk: bool = False,
        chunk_config: Optional[ChunkConfig] = None,
        **kwargs,
    ) -> TTSResult:
        target = engine or self.default_engine
        connector = await self._get_connector(target)

        if not await connector.is_available():
            return TTSResult.err_result(
                f"{connector.name} server not reachable", engine=connector.name
            )

        extra = self._filter_extra(connector, kwargs)
        model = kwargs.get("model")

        if chunk:
            return await self._synthesize_chunks(
                connector, text, speed, volume, model, extra, chunk_config
            )
        return await self._synthesize_once(
            connector, text, speed, volume, model, extra
        )

    async def save(
        self,
        text: str,
        speed: float = 1.0,
        volume: Optional[float] = None,
        engine: Optional[str] = None,
        chunk: bool = False,
        chunk_config: Optional[ChunkConfig] = None,
        **kwargs,
    ) -> TTSResult:
        return await self.synthesize(
            text, speed, volume, engine, chunk, chunk_config, **kwargs
        )

    async def say(
        self,
        text: str,
        speed: float = 1.0,
        volume: Optional[float] = None,
        engine: Optional[str] = None,
        **kwargs,
    ) -> TTSResult:
        return await self.play(text, speed, volume, engine, **kwargs)

    async def _play_streaming(
        self,
        connector: TTSConnector,
        req: TTSRequest,
    ) -> TTSResult:
        proc = await asyncio.create_subprocess_exec(
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-",
            "-loglevel",
            "quiet",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        async for chunk in connector.synthesize_stream(req):
            proc.stdin.write(chunk)
        await proc.stdin.drain()
        proc.stdin.close()
        await proc.wait()
        return TTSResult.ok_result(
            engine=connector.name, message="Streaming playback completed"
        )

    async def _play_fallback(
        self,
        connector: TTSConnector,
        req: TTSRequest,
    ) -> TTSResult:
        res = await connector.synthesize(req)
        if not res.success:
            return TTSResult.err_result(res.error, engine=connector.name)
        if not res.audio_data:
            return TTSResult.err_result("No audio data generated", engine=connector.name)
        ok = await _play_audio(res.audio_data)
        if not ok:
            return TTSResult.err_result(
                "No audio player found (paplay / aplay)", engine=connector.name
            )
        return TTSResult.ok_result(
            engine=connector.name, message="Playback completed"
        )

    async def play(
        self,
        text: str,
        speed: float = 1.0,
        volume: Optional[float] = None,
        engine: Optional[str] = None,
        **kwargs,
    ) -> TTSResult:
        target = engine or self.default_engine
        connector = await self._get_connector(target)

        if not await connector.is_available():
            return TTSResult.err_result(
                f"{connector.name} server not reachable", engine=connector.name
            )

        extra = self._filter_extra(connector, kwargs)
        req = self._build_request(
            text, speed, volume, kwargs.get("model"), extra
        )

        stream_method = getattr(connector, "synthesize_stream", None)
        ffplay_enabled = os.environ.get("FFPLAY_STREAMING", "1") == "1"
        if stream_method is not None and _has_cmd("ffplay") and ffplay_enabled:
            return await self._play_streaming(connector, req)
        return await self._play_fallback(connector, req)

    async def close(self):
        for conn in self._cache.values():
            await conn.close()
        self._cache.clear()


def _has_cmd(name: str) -> bool:
    import shutil

    return shutil.which(name) is not None


async def _play_audio(data: bytes) -> bool:
    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="_vox4ai_play_")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        player = None
        if _has_cmd("paplay"):
            player = "paplay"
        elif _has_cmd("aplay"):
            player = "aplay"
        if not player:
            return False
        proc = await asyncio.create_subprocess_exec(
            player,
            tmp,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return proc.returncode == 0
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
