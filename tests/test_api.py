import pytest
from unittest.mock import patch, AsyncMock
from vox4ai_skill_lib import api
from vox4ai_skill_lib.skill import TTSResult


class FakeSkill:
    def __init__(self, result=None, raise_exc=None):
        if result is None:
            result = TTSResult(
                status="ok",
                engine="mock",
                message="ok",
                audio_base64="YWJj",
            )
        elif isinstance(result, dict):
            result = TTSResult(**result)
        self.result = result
        self.raise_exc = raise_exc
        self.synthesize_calls = []
        self.play_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def synthesize(self, **kwargs):
        self.synthesize_calls.append(kwargs)
        if self.raise_exc:
            raise self.raise_exc
        return self.result

    async def play(self, **kwargs):
        self.play_calls.append(kwargs)
        if self.raise_exc:
            raise self.raise_exc
        return self.result


@pytest.mark.asyncio
async def test_list_engines_with_engines(capsys):
    with patch(
        "vox4ai_skill_lib.api.ConnectorFactory.list_available",
        return_value=["mock1", "mock2"],
    ):
        code = await api.list_engines()
    assert code == 0
    out = capsys.readouterr().out
    assert "mock1" in out
    assert "mock2" in out


@pytest.mark.asyncio
async def test_list_engines_no_engines(capsys):
    with patch("vox4ai_skill_lib.api.ConnectorFactory.list_available", return_value=[]):
        code = await api.list_engines()
    assert code == 0
    out = capsys.readouterr().out
    assert "見つかりません" in out


@pytest.mark.asyncio
async def test_list_engines_exception(capsys):
    with patch(
        "vox4ai_skill_lib.api.ConnectorFactory.list_available",
        side_effect=Exception("boom"),
    ):
        code = await api.list_engines()
    assert code == 1
    out = capsys.readouterr().out
    assert "エラー" in out


@pytest.mark.asyncio
async def test_synthesize_text_to_file(capsys, tmp_path):
    out_file = tmp_path / "out.wav"
    fake = FakeSkill()
    with patch.object(api, "TTSSkill", return_value=fake):
        code = await api.synthesize_text(
            text="hello",
            engine="mock",
            speed=1.0,
            volume=None,
            pitch=None,
            style_id=None,
            output=str(out_file),
            engine_kwargs={},
        )
    assert code == 0
    assert out_file.exists()
    content = out_file.read_bytes()
    assert content == b"abc"
    out = capsys.readouterr().out
    assert "保存しました" in out


@pytest.mark.asyncio
async def test_synthesize_text_to_stdout(capsys):
    fake = FakeSkill()
    with patch.object(api, "TTSSkill", return_value=fake):
        code = await api.synthesize_text(
            text="hello",
            engine="mock",
            speed=1.0,
            volume=None,
            pitch=None,
            style_id=None,
            output=None,
            engine_kwargs={},
        )
    assert code == 0
    out = capsys.readouterr().out
    assert "音声データ (Base64): YWJj" in out


@pytest.mark.asyncio
async def test_synthesize_text_long_base64(capsys):
    long_b64 = "A" * 200
    fake = FakeSkill(
        result={
            "status": "ok",
            "engine": "mock",
            "message": "ok",
            "audio_base64": long_b64,
        }
    )
    with patch.object(api, "TTSSkill", return_value=fake):
        code = await api.synthesize_text(
            text="hello",
            engine="mock",
            speed=1.0,
            volume=None,
            pitch=None,
            style_id=None,
            output=None,
            engine_kwargs={},
        )
    assert code == 0
    out = capsys.readouterr().out
    assert "全長" in out


@pytest.mark.asyncio
async def test_synthesize_text_with_kwargs(capsys):
    fake = FakeSkill()
    with patch.object(api, "TTSSkill", return_value=fake):
        await api.synthesize_text(
            text="hello",
            engine="mock",
            speed=1.2,
            volume=0.8,
            pitch=0.5,
            style_id=3,
            output=None,
            engine_kwargs={},
        )
    call = fake.synthesize_calls[0]
    assert call["volume"] == 0.8
    assert call["pitch"] == 0.5
    assert call["style_id"] == 3


@pytest.mark.asyncio
async def test_synthesize_text_with_chunking(capsys):
    fake = FakeSkill()
    with patch.object(api, "TTSSkill", return_value=fake):
        await api.synthesize_text(
            text="hello",
            engine="mock",
            speed=1.0,
            volume=None,
            pitch=None,
            style_id=None,
            output=None,
            engine_kwargs={},
            chunk=True,
        )
    call = fake.synthesize_calls[0]
    assert call["chunk"] is True
    assert "chunk_config" in call


@pytest.mark.asyncio
async def test_synthesize_text_error_status(capsys):
    fake = FakeSkill(result={"status": "error", "message": "engine failed"})
    with patch.object(api, "TTSSkill", return_value=fake):
        code = await api.synthesize_text(
            text="hello",
            engine="mock",
            speed=1.0,
            volume=None,
            pitch=None,
            style_id=None,
            output=None,
            engine_kwargs={},
        )
    assert code == 1
    out = capsys.readouterr().out
    assert "engine failed" in out


@pytest.mark.asyncio
async def test_synthesize_text_exception(capsys):
    with patch.object(api, "TTSSkill", side_effect=Exception("boom")):
        code = await api.synthesize_text(
            text="hello",
            engine="mock",
            speed=1.0,
            volume=None,
            pitch=None,
            style_id=None,
            output=None,
            engine_kwargs={},
        )
    assert code == 1
    out = capsys.readouterr().out
    assert "boom" in out


@pytest.mark.asyncio
async def test_synthesize_text_no_audio_data_handles_gracefully(capsys):
    """audio_base64=None (e.g. VoiSonaTalk) はエラーを出さず '音声データ: None' を表示するだけ。"""
    fake = FakeSkill(
        result=TTSResult(
            status="ok", engine="mock", message="ok", audio_base64=None
        )
    )
    with patch.object(api, "TTSSkill", return_value=fake):
        code = await api.synthesize_text(
            text="hello",
            engine="mock",
            speed=1.0,
            volume=None,
            pitch=None,
            style_id=None,
            output=None,
            engine_kwargs={},
        )
    assert code == 0
    out = capsys.readouterr().out
    assert "音声データ: None" in out


@pytest.mark.asyncio
async def test_synthesize_text_play_success(capsys, monkeypatch):
    monkeypatch.setattr(api, "_play_audio", AsyncMock(return_value=True))
    fake = FakeSkill()
    with patch.object(api, "TTSSkill", return_value=fake):
        code = await api.synthesize_text(
            text="hello",
            engine="mock",
            speed=1.0,
            volume=None,
            pitch=None,
            style_id=None,
            output=None,
            engine_kwargs={},
            play=True,
        )
    assert code == 0
    out = capsys.readouterr().out
    assert "再生完了" in out


@pytest.mark.asyncio
async def test_synthesize_text_play_no_player(capsys, monkeypatch):
    monkeypatch.setattr(api, "_play_audio", AsyncMock(return_value=False))
    fake = FakeSkill()
    with patch.object(api, "TTSSkill", return_value=fake):
        code = await api.synthesize_text(
            text="hello",
            engine="mock",
            speed=1.0,
            volume=None,
            pitch=None,
            style_id=None,
            output=None,
            engine_kwargs={},
            play=True,
        )
    assert code == 1
    out = capsys.readouterr().out
    assert "paplay" in out


@pytest.mark.asyncio
async def test_play_text_ok(capsys):
    fake = FakeSkill()
    with patch.object(api, "TTSSkill", return_value=fake):
        code = await api.play_text(
            text="hello",
            engine="mock",
            speed=1.0,
            volume=None,
            pitch=None,
            style_id=None,
            model=None,
            engine_kwargs={},
        )
    assert code == 0
    out = capsys.readouterr().out
    assert "✅" in out


@pytest.mark.asyncio
async def test_play_text_error(capsys):
    fake = FakeSkill(result={"status": "error", "message": "play failed"})
    with patch.object(api, "TTSSkill", return_value=fake):
        code = await api.play_text(
            text="hello",
            engine="mock",
            speed=1.0,
            volume=None,
            pitch=None,
            style_id=None,
            model=None,
            engine_kwargs={},
        )
    assert code == 1
    out = capsys.readouterr().out
    assert "play failed" in out


@pytest.mark.asyncio
async def test_play_text_exception(capsys):
    with patch.object(api, "TTSSkill", side_effect=Exception("boom")):
        code = await api.play_text(
            text="hello",
            engine="mock",
            speed=1.0,
            volume=None,
            pitch=None,
            style_id=None,
            model=None,
            engine_kwargs={},
        )
    assert code == 1
    out = capsys.readouterr().out
    assert "boom" in out


@pytest.mark.asyncio
async def test_play_text_with_model(capsys):
    fake = FakeSkill()
    with patch.object(api, "TTSSkill", return_value=fake):
        await api.play_text(
            text="hello",
            engine="mock",
            speed=1.0,
            volume=None,
            pitch=None,
            style_id=2,
            model="ja-voice",
            engine_kwargs={},
        )
    call = fake.play_calls[0]
    assert call["model"] == "ja-voice"
    assert call["style_id"] == 2


@pytest.mark.asyncio
async def test_test_connection_ok(capsys):
    fake = FakeSkill()
    with patch.object(api, "TTSSkill", return_value=fake):
        code = await api.test_connection(engine="mock", style_id=None, engine_kwargs={})
    assert code == 0
    out = capsys.readouterr().out
    assert "接続成功" in out


@pytest.mark.asyncio
async def test_test_connection_fail(capsys):
    fake = FakeSkill(result={"status": "error", "message": "unreachable"})
    with patch.object(api, "TTSSkill", return_value=fake):
        code = await api.test_connection(engine="mock", style_id=None, engine_kwargs={})
    assert code == 1
    out = capsys.readouterr().out
    assert "接続失敗" in out


@pytest.mark.asyncio
async def test_test_connection_exception(capsys):
    with patch.object(api, "TTSSkill", side_effect=Exception("boom")):
        code = await api.test_connection(engine="mock", style_id=None, engine_kwargs={})
    assert code == 1
    out = capsys.readouterr().out
    assert "boom" in out


@pytest.mark.asyncio
async def test_test_connection_with_style_id(capsys):
    fake = FakeSkill()
    with patch.object(api, "TTSSkill", return_value=fake):
        await api.test_connection(engine="mock", style_id=3, engine_kwargs={})
    call = fake.synthesize_calls[0]
    assert call["style_id"] == 3
    assert call["text"] == "テスト"


@pytest.mark.asyncio
async def test_synthesize_text_default_engine(capsys):
    fake = FakeSkill()
    with patch.object(api, "TTSSkill", return_value=fake) as m:
        await api.synthesize_text(
            text="hello",
            engine=None,
            speed=1.0,
            volume=None,
            pitch=None,
            style_id=None,
            output=None,
            engine_kwargs={},
        )
    m.assert_called_once_with(default_engine="piperplus")


@pytest.mark.asyncio
async def test_play_text_default_engine(capsys):
    fake = FakeSkill()
    with patch.object(api, "TTSSkill", return_value=fake) as m:
        await api.play_text(
            text="hello",
            engine=None,
            speed=1.0,
            volume=None,
            pitch=None,
            style_id=None,
            model=None,
            engine_kwargs={},
        )
    m.assert_called_once_with(default_engine="piperplus")
