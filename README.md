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
ハーネスでタスク生成も行う例は [examples/git_repo_harness_taskgen_opencode.yaml](examples/git_repo_harness_taskgen_opencode.yaml) を参照してください。

主な設定項目:

- **run**: 実行名、タスク数、同時実行数、タイムアウト、クリーンアップポリシー
- **provider**: OpenAI 互換 API エンドポイント、モデル、API キー（環境変数経由）
- **logging_proxy**: リッスンアドレス、キャプチャ設定、推論内容の正規化
- **sandbox**: Git リポジトリ、コンテナイメージ、ボリューム、ネットワーク、環境変数
- **harness**: インストールコマンド、実行コマンド、完了条件
- **taskgen**: タスク生成戦略、静的タスク、または生成専用ハーネスと複数プロンプト
- **artifacts**: 収集する成果物（stdout, stderr, diff, テスト結果）
- **export**: 出力形式、推論内容/ツール呼び出し/失敗実行のフィルタリング

## ロギングプロキシ

ロギングプロキシはサンドボックスからの OpenAI 互換 API 呼び出しをインターセプトし、リクエスト/レスポンスのトレースを記録して実際のプロバイダに転送します。

```bash
distill-gym proxy --config config.yaml
```

サンドボックスはプロキシ経由で通信します。接続先ホストは `logging_proxy.sandbox_host` で指定でき、既定値 `auto` では Podman は `host.containers.internal`、Windows の Docker は `host.docker.internal` を使います。Windows 上の Podman machine が WSL VM の場合は、WSL の default gateway を自動検出して Windows 側プロキシへ接続します。Windows で内部実行する場合、プロキシはサンドボックスから到達できるように `127.0.0.1` ではなく `0.0.0.0` へ bind します。プロキシは `provider.api_key_env` から実際の API キーを注入するため、サンドボックスは実際のキーにアクセスできません。

Windows + WSL + Podman でプロキシへ到達できない場合も、通常は `logging_proxy.sandbox_host: auto` のまま使ってください。手動で回避する場合は `podman machine ssh "ip route show default"` を実行し、`default via 172.22.224.1` のように表示される IP を `logging_proxy.sandbox_host: 172.22.224.1` として指定します。Windows ファイアウォールが Python/uvicorn の受信をブロックしている場合も接続できないため、許可ルールを確認してください。

## ハーネスベースのタスク生成

`taskgen.type: harness` を指定すると、実行用 `harness` とは別の `taskgen.harness` を使ってタスクを生成できます。`taskgen.prompts` に複数の生成プロンプトを定義すると、`run.task_count` に達するまでラウンドロビンで実行されます。

生成ハーネスには `{output_file}` に JSON 配列を書き出させます。stdout はタスク結果として扱いません。

```yaml
taskgen:
  type: harness
  output_file: .distill-gym/taskgen/tasks.json
  batch_size: 2
  max_rounds: 6
  harness:
    type: opencode
    run:
      command: opencode run --format json {task.prompt.shell}
  prompts:
    - id: bugfix
      prompt: |
        Generate up to {batch_size} bugfix tasks.
        Avoid duplicating these tasks: {existing_tasks_json}
        Write only the JSON array to {output_file}.
```

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
- タスク生成は静的設定タスク、既存の `repo_auto`、または `taskgen.type: harness` に対応
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
