from tts_plugin_bridge import ConnectorFactory
from tts_plugin_bridge.protocol import ChunkConfig

from .skill import TTSSkill, _play_audio
from typing import Optional


async def list_engines():
    try:
        engines = ConnectorFactory.list_available()
        if engines:
            print("利用可能なTTSプラグイン:")
            for engine in engines:
                print(f"  - {engine}")
        else:
            print("利用可能なTTSプラグインが見つかりません。")
            print("プラグインをインストールするには: uv add tts-plugin-<エンジン名>")
    except Exception as e:
        print(f"エラー: {e}")
        return 1
    return 0


async def synthesize_text(
    text: str,
    engine: Optional[str],
    speed: float,
    volume: Optional[float],
    pitch: Optional[float],
    style_id: Optional[int],
    output: Optional[str],
    engine_kwargs: dict,
    chunk: bool = False,
    play: bool = False,
    chunk_config: Optional[ChunkConfig] = None,
):
    try:
        async with TTSSkill(
            default_engine=engine or "piperplus", **engine_kwargs
        ) as skill:
            kwargs = {}
            if volume is not None:
                kwargs["volume"] = volume
            if pitch is not None:
                kwargs["pitch"] = pitch
            if style_id is not None:
                kwargs["style_id"] = style_id
            if chunk:
                kwargs["chunk"] = True
                kwargs["chunk_config"] = chunk_config

            result = await skill.synthesize(
                text=text, speed=speed, engine=engine, **kwargs
            )

            if result["status"] == "ok":
                import base64

                audio_data = base64.b64decode(result["audio_base64"])
                if output:
                    with open(output, "wb") as f:
                        f.write(audio_data)
                    print(f"音声データを {output} に保存しました。")
                    print(f"エンジン: {result['engine']}")
                    print(f"メッセージ: {result['message']}")
                elif play:
                    ok = await _play_audio(audio_data)
                    if not ok:
                        print(
                            "エラー: 再生に使えるコマンド (paplay / aplay) が見つかりません。"
                        )
                        return 1
                    print(f"✅ 再生完了 (エンジン: {result['engine']})")
                else:
                    print(f"エンジン: {result['engine']}")
                    print(f"メッセージ: {result['message']}")
                    print(
                        f"音声データ (Base64): {result['audio_base64'][:100]}..."
                        if result["audio_base64"]
                        else "音声データ: None"
                    )
                    if len(result["audio_base64"] or "") > 100:
                        print(f"... (全長: {len(result['audio_base64'] or '')} 文字)")
            else:
                print(f"エラー: {result['message']}")
                return 1
    except Exception as e:
        print(f"エラー: {e}")
        return 1
    return 0


async def play_text(
    text: str,
    engine: Optional[str],
    speed: float,
    volume: Optional[float],
    pitch: Optional[float],
    style_id: Optional[int],
    model: Optional[str],
    engine_kwargs: dict,
):
    try:
        async with TTSSkill(
            default_engine=engine or "piperplus", **engine_kwargs
        ) as skill:
            kwargs = {}
            if volume is not None:
                kwargs["volume"] = volume
            if pitch is not None:
                kwargs["pitch"] = pitch
            if style_id is not None:
                kwargs["style_id"] = style_id
            if model is not None:
                kwargs["model"] = model
            result = await skill.play(text=text, speed=speed, engine=engine, **kwargs)
            if result["status"] == "ok":
                print(f"✅ {result['message']} (エンジン: {result['engine']})")
            else:
                print(f"エラー: {result['message']}")
                return 1
    except Exception as e:
        print(f"エラー: {e}")
        return 1
    return 0


async def test_connection(
    engine: Optional[str], style_id: Optional[int], engine_kwargs: dict
):
    try:
        async with TTSSkill(
            default_engine=engine or "piperplus", **engine_kwargs
        ) as skill:
            kwargs = {}
            if style_id is not None:
                kwargs["style_id"] = style_id
            result = await skill.synthesize(
                text="テスト", speed=1.0, engine=engine, **kwargs
            )

            if result["status"] == "ok":
                print("✅ 接続成功!")
                print(f"エンジン: {result['engine']}")
                print(f"メッセージ: {result['message']}")
            else:
                print(f"❌ 接続失敗: {result['message']}")
                return 1
    except Exception as e:
        print(f"❌ エラー: {e}")
        return 1
    return 0
