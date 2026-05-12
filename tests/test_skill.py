import pytest
from vox4ai_skill import TTSSkill
from tts_plugin_bridge import ConnectorFactory
from tts_plugin_bridge.protocol import TTSConnector, TTSRequest, TTSResponse


class MockConnector(TTSConnector):
    ENGINE_NAME = "mock"
    SUPPORTED_PARAMS = ["voice", "rate"]
    last_request: TTSRequest | None = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def synthesize(self, req: TTSRequest) -> TTSResponse:
        MockConnector.last_request = req
        return TTSResponse.ok(audio_data=b"mock_audio")

    async def synthesize_stream(self, req):
        MockConnector.last_request = req
        yield b"mock_stream_chunk"

    async def is_available(self) -> bool:
        return True

    async def close(self):
        pass


@pytest.fixture
def mock_factory():
    ConnectorFactory._registry["mock"] = MockConnector
    yield
    if "mock" in ConnectorFactory._registry:
        del ConnectorFactory._registry["mock"]
    MockConnector.last_request = None


@pytest.mark.asyncio
async def test_synthesize(mock_factory):
    skill = TTSSkill(default_engine="mock")
    res = await skill.synthesize(text="test")
    assert res["status"] == "ok"
    assert res["engine"] == "mock"
    assert res["message"] == "TTS synthesis completed"


@pytest.mark.asyncio
async def test_synthesize_with_model(mock_factory):
    MockConnector.last_request = None
    skill = TTSSkill(default_engine="mock")
    res = await skill.synthesize(text="test", model="test-voice")
    assert res["status"] == "ok"
    assert MockConnector.last_request is not None
    assert MockConnector.last_request.model == "test-voice"


@pytest.mark.asyncio
async def test_play(mock_factory):
    skill = TTSSkill(default_engine="mock")
    res = await skill.play(text="test")
    assert res["status"] == "ok"
    assert MockConnector.last_request is not None


@pytest.mark.asyncio
async def test_save(mock_factory):
    skill = TTSSkill(default_engine="mock")
    res = await skill.save(text="test")
    assert res["status"] == "ok"


@pytest.mark.asyncio
async def test_context_manager(mock_factory):
    async with TTSSkill(default_engine="mock") as skill:
        res = await skill.synthesize(text="test")
        assert res["status"] == "ok"
    assert len(skill._cache) == 0


@pytest.mark.asyncio
async def test_unavailable_engine(mock_factory):
    class UnavailableConnector(MockConnector):
        async def is_available(self) -> bool:
            return False

    ConnectorFactory._registry["mock"] = UnavailableConnector
    skill = TTSSkill(default_engine="mock")
    res = await skill.synthesize(text="test")
    assert res["status"] == "error"
    assert "not reachable" in res["message"]


@pytest.mark.asyncio
async def test_play_with_model(mock_factory):
    MockConnector.last_request = None
    skill = TTSSkill(default_engine="mock")
    res = await skill.play(text="test", model="test-voice")
    assert res["status"] == "ok"
    assert MockConnector.last_request is not None
    assert MockConnector.last_request.model == "test-voice"