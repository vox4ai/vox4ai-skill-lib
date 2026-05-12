import base64
import os
import asyncio

from tts_plugin_bridge import ConnectorFactory
from tts_plugin_bridge.protocol import TTSRequest, TTSResponse, TTSConnector, ChunkConfig
from typing import Optional


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

    async def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        volume: Optional[float] = None,
        engine: Optional[str] = None,
        chunk: bool = False,
        chunk_config: Optional[ChunkConfig] = None,
        **kwargs,
    ) -> dict:
        target = engine or self.default_engine
        connector = await self._get_connector(target)

        if not await connector.is_available():
            return {"status": "error", "message": f"{connector.name} server not reachable"}

        if chunk:
            from tts_plugin_bridge.chunker import HybridChunker, ChunkConfig as ChunkerConfig

            actual_config = chunk_config or ChunkerConfig()
            chunker = HybridChunker()
            chunks_to_process = chunker.chunk(text, actual_config)

            combined_audio = bytearray()
            supported = {k: v for k, v in kwargs.items() if k in connector.get_supported_params()}
            for chunk_res in chunks_to_process:
                req = TTSRequest(
                    text=chunk_res.text,
                    speed=speed,
                    volume=volume,
                    model=kwargs.get("model"),
                    extra=supported,
                )
                res: TTSResponse = await connector.synthesize(req)
                if not res.success:
                    return {"status": "error", "message": f"Error in chunk {chunk_res.index}: {res.error}"}
                if res.audio_data:
                    combined_audio.extend(res.audio_data)

            if not combined_audio:
                return {"status": "error", "message": "No audio data was generated"}

            return {
                "status": "ok",
                "engine": connector.name,
                "audio_base64": base64.b64encode(combined_audio).decode(),
                "message": f"TTS synthesis completed ({len(chunks_to_process)} chunks)",
            }

        extra = {k: v for k, v in kwargs.items() if k in connector.get_supported_params()}
        req = TTSRequest(
            text=text,
            speed=speed,
            volume=volume,
            model=kwargs.get("model"),
            extra=extra,
        )
        res: TTSResponse = await connector.synthesize(req)

        if res.success:
            return {
                "status": "ok",
                "engine": connector.name,
                "audio_base64": base64.b64encode(res.audio_data).decode() if res.audio_data else None,
                "message": "TTS synthesis completed",
            }
        return {"status": "error", "message": res.error}

    async def save(
        self,
        text: str,
        speed: float = 1.0,
        volume: Optional[float] = None,
        engine: Optional[str] = None,
        chunk: bool = False,
        chunk_config: Optional[ChunkConfig] = None,
        **kwargs,
    ) -> dict:
        return await self.synthesize(text, speed, volume, engine, chunk, chunk_config, **kwargs)

    async def say(self, text: str, speed: float = 1.0, volume: Optional[float] = None, engine: Optional[str] = None, **kwargs) -> dict:
        return await self.play(text, speed, volume, engine, **kwargs)

    async def play(
        self,
        text: str,
        speed: float = 1.0,
        volume: Optional[float] = None,
        engine: Optional[str] = None,
        **kwargs,
    ) -> dict:
        target = engine or self.default_engine
        connector = await self._get_connector(target)

        if not await connector.is_available():
            return {"status": "error", "message": f"{connector.name} server not reachable"}

        extra = {k: v for k, v in kwargs.items() if k in connector.get_supported_params()}
        req = TTSRequest(
            text=text,
            speed=speed,
            volume=volume,
            model=kwargs.get("model"),
            extra=extra,
        )

        stream_method = getattr(connector, "synthesize_stream", None)
        _ffplay_override = os.environ.get("FFPLAY_STREAMING", "1") == "1"
        if stream_method is not None and _has_cmd("ffplay") and _ffplay_override:
            proc = await asyncio.create_subprocess_exec(
                "ffplay", "-nodisp", "-autoexit", "-", "-loglevel", "quiet",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            async for chunk in stream_method(req):
                proc.stdin.write(chunk)
            await proc.stdin.drain()
            proc.stdin.close()
            await proc.wait()
            return {"status": "ok", "engine": connector.name, "message": "Streaming playback completed"}
        else:
            res: TTSResponse = await connector.synthesize(req)
            if not res.success:
                return {"status": "error", "message": res.error}
            if not res.audio_data:
                return {"status": "error", "message": "No audio data generated"}
            ok = await _play_audio(res.audio_data)
            if not ok:
                return {"status": "error", "message": "No audio player found (paplay / aplay)"}
            return {"status": "ok", "engine": connector.name, "message": "Playback completed"}

    async def close(self):
        for conn in self._cache.values():
            await conn.close()
        self._cache.clear()


def _has_cmd(name: str) -> bool:
    import shutil
    return shutil.which(name) is not None


async def _play_audio(data: bytes) -> bool:
    import os
    tmp = "/tmp/_tts_play.wav"
    with open(tmp, "wb") as f:
        f.write(data)
    try:
        player = None
        if _has_cmd("paplay"):
            player = "paplay"
        elif _has_cmd("aplay"):
            player = "aplay"
        if not player:
            return False
        proc = await asyncio.create_subprocess_exec(
            player, tmp,
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