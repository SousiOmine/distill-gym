# distill-gym

AI Harness 実行トレース収集・SFT データセット生成ツール。

OpenAI 互換のロギングプロキシを介して AI コーディングハーネス（opencode, codex, qwen-code 等）から実行トレースを収集し、ローカル LLM の教師ありファインチューニング（SFT）に適した OpenAI messages 形式の JSONL データセットを生成します。

## 実行方法
```bash
git clone <repo>
uv tool install --install .
```

## 使い方

### キャッシュディレクトリの初期化

```bash
distill-gym init
```

### 設定の検証

```bash
distill-gym validate examples/git_repo_opencode.yaml
```

### 収集ジョブの実行（モックハーネス）

```bash
distill-gym run examples/mock_run.yaml
```

### ドライラン（サンドボックスなし）

```bash
distill-gym run examples/mock_run.yaml --dry-run
```

### ロギングプロキシの起動

```bash
distill-gym proxy --config examples/git_repo_opencode.yaml
```

### SFT データセットのエクスポート

```bash
distill-gym export --run-id <run_id> --format openai-messages --output out.jsonl
```

### Web UI の起動

```bash
distill-gym web --host 127.0.0.1 --port 8000
```

### リソースのクリーンアップ

```bash
distill-gym cleanup
```

## 設定

完全な設定例は [examples/git_repo_opencode.yaml](examples/git_repo_opencode.yaml) を参照してください。

主な設定項目:

- **run**: 実行名、タスク数、同時実行数、タイムアウト、クリーンアップポリシー
- **provider**: OpenAI 互換 API エンドポイント、モデル、API キー（環境変数経由）
- **logging_proxy**: リッスンアドレス、キャプチャ設定、推論内容の正規化
- **sandbox**: Git リポジトリ、コンテナイメージ、ボリューム、ネットワーク、環境変数
- **harness**: インストールコマンド、実行コマンド、完了条件
- **taskgen**: タスク生成戦略とテンプレート
- **artifacts**: 収集する成果物（stdout, stderr, diff, テスト結果）
- **export**: 出力形式、推論内容/ツール呼び出し/失敗実行のフィルタリング

## ロギングプロキシ

ロギングプロキシはサンドボックスからの OpenAI 互換 API 呼び出しをインターセプトし、リクエスト/レスポンスのトレースを記録して実際のプロバイダに転送します。

```bash
distill-gym proxy --config config.yaml
```

サンドボックスは `OPENAI_BASE_URL=http://host.containers.internal:5002/v1` を使用してプロキシ経由で通信します。プロキシは `provider.api_key_env` から実際の API キーを注入するため、サンドボックスは実際のキーにアクセスできません。

## 必要条件

- **Podman**（Docker も一部制限付きで可）
  - macOS/Windows: Podman machine が必要
- Python 3.12+
- uv

## セキュリティ

- API キーは環境変数から読み取られ、設定ファイルに保存されることはありません
- ロギングプロキシは保存時に `Authorization` ヘッダーを除去します
- メタデータにシークレットや絶対ローカルパスが含まれることはありません
- サンドボックスコンテナは設計上プロキシ専用ネットワークモードで動作します

## SFT データセット形式

出力 JSONL（1 行が 1 会話に対応）:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "reasoning_content": "...", "content": "...", "tool_calls": []},
    {"role": "tool", "tool_call_id": "...", "content": "..."}
  ],
  "metadata": {
    "run_id": "...",
    "task_id": "...",
    "harness": {"name": "...", "version": "...", "command": "..."},
    "provider": {"name": "...", "base_url": "...", "model": "..."},
    "sandbox": {"type": "git_repository", "engine": "podman", "image": "...", "repo_url": "...", "commit": "..."},
    "result": {"success": true, "exit_code": 0, "tests_passed": true, "changed_files": []},
    "artifacts": {"raw_trace": "...", "stdout": "...", "stderr": "...", "diff": "...", "test_result": "..."}
  }
}
```

## 制限事項（MVP）

- `network.mode: proxy_only` は厳密に強制されておらず、コンテナは現在ブリッジネットワークを使用
- `openai_compatible` プロバイダタイプのみ対応
- `git_repository` サンドボックスタイプのみ実装
- Docker バックエンドは将来の拡張として保留（現状 Podman のみ）
- タスク生成は静的設定タスクまたはダミーテンプレートを使用。LLM ベースの生成は未実装
- ストリーミングモードのキャプチャは動作しますが、プロバイダ間の異種チャンク形式でエッジケースが存在する可能性があります

## 開発

```bash
# テスト実行
uv run pytest

# バリデーション実行
uv run distill-gym validate examples/git_repo_opencode.yaml

# モック収集の実行
uv run distill-gym run examples/mock_run.yaml
```
