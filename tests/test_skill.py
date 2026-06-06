import asyncio

import pytest
from vox4ai_skill_lib import TTSSkill
from tts_plugin_bridge import ConnectorFactory
from tts_plugin_bridge.protocol import TTSConnector, TTSRequest, TTSResponse


class MockConnector(TTSConnector):
    ENGINE_NAME = "mock"
    SUPPORTED_PARAMS = ["voice", "rate"]
    last_request: TTSRequest | None = None
    fail_on_index: int = -1
    return_none_audio: bool = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def synthesize(self, req: TTSRequest) -> TTSResponse:
        MockConnector.last_request = req
        if self.fail_on_index >= 0:
            return TTSResponse.fail(f"Failed on chunk {self.fail_on_index}")
        if self.return_none_audio:
            return TTSResponse.ok(audio_data=None)
        return TTSResponse.ok(audio_data=b"mock_audio")

    async def synthesize_stream(self, req):
        MockConnector.last_request = req
        yield b"mock_stream_chunk"

    async def is_available(self) -> bool:
        return True

    async def close(self):
        pass


class UnavailableConnector(MockConnector):
    async def is_available(self) -> bool:
        return False


@pytest.fixture
def mock_factory():
    ConnectorFactory._registry["mock"] = MockConnector
    yield
    if "mock" in ConnectorFactory._registry:
        del ConnectorFactory._registry["mock"]
    MockConnector.last_request = None
    MockConnector.fail_on_index = -1
    MockConnector.return_none_audio = False


@pytest.mark.asyncio
async def test_synthesize(mock_factory):
    skill = TTSSkill(default_engine="mock")
    res = await skill.synthesize(text="test")
    assert res.status == "ok"
    assert res.engine == "mock"
    assert res.message == "TTS synthesis completed"


@pytest.mark.asyncio
async def test_synthesize_with_model(mock_factory):
    MockConnector.last_request = None
    skill = TTSSkill(default_engine="mock")
    res = await skill.synthesize(text="test", model="test-voice")
    assert res.status == "ok"
    assert MockConnector.last_request is not None
    assert MockConnector.last_request.model == "test-voice"


@pytest.mark.asyncio
async def test_synthesize_error_response(mock_factory):
    MockConnector.fail_on_index = 0
    skill = TTSSkill(default_engine="mock")
    res = await skill.synthesize(text="test")
    assert res.status == "error"
    assert "Failed" in res.message


@pytest.mark.asyncio
async def test_synthesize_filters_unsupported_params(mock_factory):
    skill = TTSSkill(default_engine="mock")
    res = await skill.synthesize(text="test", voice="ja", unsupported_key="ignored")
    assert res.status == "ok"
    assert MockConnector.last_request.extra == {"voice": "ja"}


@pytest.mark.asyncio
async def test_synthesize_with_chunking(mock_factory):
    skill = TTSSkill(default_engine="mock")
    res = await skill.synthesize(text="一文目。二文目。三文目。", chunk=True)
    assert res.status == "ok"
    assert "chunks" in res.message
    assert res.audio_base64 is not None


@pytest.mark.asyncio
async def test_synthesize_with_chunking_custom_config(mock_factory):
    from tts_plugin_bridge.protocol import ChunkConfig

    skill = TTSSkill(default_engine="mock")
    res = await skill.synthesize(
        text="一二三四五六。七八九〇。",
        chunk=True,
        chunk_config=ChunkConfig(max_chars=3),
    )
    assert res.status == "ok"


@pytest.mark.asyncio
async def test_synthesize_chunk_failure(mock_factory):
    MockConnector.fail_on_index = 1
    skill = TTSSkill(default_engine="mock")
    res = await skill.synthesize(text="一文目。二文目。三文目。", chunk=True)
    assert res.status == "error"
    assert "chunk 1" in res.message


@pytest.mark.asyncio
async def test_synthesize_chunk_no_audio(mock_factory):
    MockConnector.return_none_audio = True
    skill = TTSSkill(default_engine="mock")
    res = await skill.synthesize(text="一文目。", chunk=True)
    assert res.status == "error"
    assert "No audio data" in res.message


@pytest.mark.asyncio
async def test_synthesize_with_volume_and_pitch(mock_factory):
    skill = TTSSkill(default_engine="mock")
    res = await skill.synthesize(text="test", volume=0.8)
    assert res.status == "ok"
    assert MockConnector.last_request.volume == 0.8


@pytest.mark.asyncio
async def test_synthesize_no_audio_returns_none_base64(mock_factory):
    MockConnector.return_none_audio = True
    skill = TTSSkill(default_engine="mock")
    res = await skill.synthesize(text="test")
    assert res.status == "ok"
    assert res.audio_base64 is None


@pytest.mark.asyncio
async def test_synthesize_engine_override(mock_factory):
    skill = TTSSkill(default_engine="mock")
    res = await skill.synthesize(text="test", engine="mock")
    assert res.status == "ok"
    assert res.engine == "mock"


@pytest.mark.asyncio
async def test_play(mock_factory):
    skill = TTSSkill(default_engine="mock")
    res = await skill.play(text="test")
    assert res.status == "ok"
    assert MockConnector.last_request is not None


@pytest.mark.asyncio
async def test_play_with_model(mock_factory):
    MockConnector.last_request = None
    skill = TTSSkill(default_engine="mock")
    res = await skill.play(text="test", model="test-voice")
    assert res.status == "ok"
    assert MockConnector.last_request is not None
    assert MockConnector.last_request.model == "test-voice"


@pytest.mark.asyncio
async def test_play_error_response(mock_factory, monkeypatch):
    monkeypatch.setenv("FFPLAY_STREAMING", "0")
    MockConnector.fail_on_index = 0
    skill = TTSSkill(default_engine="mock")
    res = await skill.play(text="test")
    assert res.status == "error"


@pytest.mark.asyncio
async def test_play_unavailable_engine(mock_factory):
    ConnectorFactory._registry["mock"] = UnavailableConnector
    skill = TTSSkill(default_engine="mock")
    res = await skill.play(text="test")
    assert res.status == "error"
    assert "not reachable" in res.message


@pytest.mark.asyncio
async def test_say_delegates_to_play(mock_factory):
    skill = TTSSkill(default_engine="mock")
    res = await skill.say(text="test")
    assert res.status == "ok"


@pytest.mark.asyncio
async def test_save_delegates_to_synthesize(mock_factory):
    skill = TTSSkill(default_engine="mock")
    res = await skill.save(text="test")
    assert res.status == "ok"
    assert res.message == "TTS synthesis completed"


@pytest.mark.asyncio
async def test_save_with_chunking(mock_factory):
    skill = TTSSkill(default_engine="mock")
    res = await skill.save(text="一文目。二文目。", chunk=True)
    assert res.status == "ok"
    assert "chunks" in res.message


@pytest.mark.asyncio
async def test_context_manager(mock_factory):
    async with TTSSkill(default_engine="mock") as skill:
        res = await skill.synthesize(text="test")
        assert res.status == "ok"
    assert len(skill._cache) == 0


@pytest.mark.asyncio
async def test_unavailable_engine(mock_factory):
    ConnectorFactory._registry["mock"] = UnavailableConnector
    skill = TTSSkill(default_engine="mock")
    res = await skill.synthesize(text="test")
    assert res.status == "error"
    assert "not reachable" in res.message


@pytest.mark.asyncio
async def test_close_clears_cache(mock_factory):
    skill = TTSSkill(default_engine="mock")
    await skill._get_connector("mock")
    assert len(skill._cache) == 1
    await skill.close()
    assert len(skill._cache) == 0


@pytest.mark.asyncio
async def test_engine_kwargs_passed_to_connector(mock_factory):
    skill = TTSSkill(default_engine="mock", voice="ja", rate=1.2)
    await skill._get_connector("mock")
    connector = skill._cache["mock"]
    assert connector.kwargs == {"voice": "ja", "rate": 1.2}


@pytest.mark.asyncio
async def test_get_connector_caches(mock_factory):
    skill = TTSSkill(default_engine="mock")
    c1 = await skill._get_connector("mock")
    c2 = await skill._get_connector("mock")
    assert c1 is c2


@pytest.mark.asyncio
async def test_play_audio_concurrent_unique_files(monkeypatch, tmp_path):
    """並行呼び出し時に tempfile パスが衝突しないこと"""
    from vox4ai_skill_lib import skill as skill_mod
    import os

    seen_paths: list[str] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        seen_paths.append(args[1])
        proc = type("P", (), {"wait": lambda self: _coro_return(0), "returncode": 0})()
        await proc.wait()
        return proc

    async def _coro_return(v):
        return v

    monkeypatch.setattr(skill_mod, "_has_cmd", lambda n: True)
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", fake_create_subprocess_exec
    )

    await asyncio.gather(
        skill_mod._play_audio(b"RIFF" + b"\x00" * 100),
        skill_mod._play_audio(b"RIFF" + b"\x00" * 100),
        skill_mod._play_audio(b"RIFF" + b"\x00" * 100),
    )

    assert len(seen_paths) == 3
    assert len(set(seen_paths)) == 3, f"path collision: {seen_paths}"
    for p in seen_paths:
        assert "_vox4ai_play_" in p
        assert not os.path.exists(p), f"tmp not cleaned: {p}"
