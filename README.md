# vox4ai-skill-lib

Python ライブラリとしての TTS Skill — AI agent や CLI から `tts-plugin-bridge` の TTS Engine を統一的に操作します。

## インストール

```bash
uv add vox4ai-skill-lib
```

## 使い方

### TTSSkill（推奨）

```python
import asyncio
from vox4ai_skill_lib import TTSSkill

async def main():
    async with TTSSkill(default_engine="edgetts") as skill:
        # 音声合成（Base64 で音声データ取得）
        result = await skill.synthesize(text="こんにちは")
        print(result["status"])  # "ok"

        # ファイル保存
        result = await skill.save(text="こんにちは")

        # ストリーミング再生（ffplay 優先）
        result = await skill.say(text="こんにちは")

        # モデル指定
        result = await skill.synthesize(
            text="こんにちは",
            model="ja-JP-KeitaNeural",
        )

asyncio.run(main())
```

### API 関数（簡易呼び出し用）

```python
import asyncio
from vox4ai_skill_lib.api import list_engines, play_text

async def main():
    await list_engines()
    await play_text("こんにちは", engine="edgetts", speed=1.0,
                    volume=None, pitch=None, style_id=None,
                    model="ja-JP-NanamiNeural", engine_kwargs={})

asyncio.run(main())
```

## TTSSkill API

| メソッド | 説明 |
|----------|------|
| `synthesize()` | 音声合成 → Base64 音声データを dict で返す |
| `save()` | synthesize() のエイリアス |
| `play()` | ストリーミング再生（ffplay）またはファイル再生（paplay/aplay） |
| `say()` | play() のエイリアス |
| `close()` | 全コネクタをクローズ |

全メソッドで `model` / `pitch` / `volume` / `style_id` / `engine` を **kwargs として受け付けます。

コンテキストマネージャ対応:

```python
async with TTSSkill(default_engine="edgetts") as skill:
    result = await skill.synthesize(text="test")
# close() 自動呼び出し
```

## API 関数

| 関数 | 説明 |
|------|------|
| `list_engines()` | 利用可能な TTS Engine 一覧を表示 |
| `synthesize_text()` | 音声合成 + ファイル保存 / Base64 表示 |
| `play_text()` | テキスト読み上げ（ストリーミング + ファイルフォールバック） |
| `test_connection()` | TTS Engine への接続テスト |

## 依存

- `tts-plugin-bridge` — コアフレームワーク（protocol / factory / chunker）
- `aiohttp` — 非同期 HTTP

## ライセンス

MIT License