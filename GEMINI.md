# vox4ai-skill-lib

Python ライブラリとしての TTS Skill — AI agent や CLI から `tts-plugin-bridge` の TTS Engine を統一的に操作します。

## 🛠 概要
- **役割**: `tts-plugin-bridge` を利用した TTS 操作の Python API を提供。
- **主要機能**:
    - `TTSSkill` クラス: コンテキストマネージャ対応、非同期操作（synthesize, save, say/play）。
    - `api` 関数群: `list_engines`, `play_text` などの簡易呼び出し。

## 🚀 開発・実行
- **パッケージ管理**: `uv`
- **テスト**: `pytest`

## 🔗 関連リポジトリ
- `repos/tts-plugin-bridge`: コアフレームワーク
