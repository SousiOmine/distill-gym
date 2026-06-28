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

## 進化的タスク生成

`taskgen.type: evolutionary` を指定すると、複数の研究論文に基づく進化的アルゴリズムで高難易度のコーディングタスクを自動生成します。

### 統合されている手法

| 手法 | 論文 | 役割 |
|------|------|------|
| 概念グラフ + ランダムウォーク | QueST (arXiv:2510.17715) | シードタスクから概念を抽出し、共起グラフ上の重み付きランダムウォークで新規な概念組み合わせをサンプリング |
| 多戦略変異 | UniCode (arXiv:2510.17868) | 単一問題拡張・同型融合・異型融合の3つの変異戦略でタスクを難化 |
| 制約追加変異 | BenchEvolver (arXiv:2606.01286) | 計算量制約・エッジケース・バッチ処理などの制約を追加して難易度を向上 |
| 概念組み合わせ生成 | QueST (arXiv:2510.17715) | サンプリングされた概念組み合わせから新規タスクを生成 |
| 経験的難易度推定 | ACES (NeurIPS 2024) | ソルバーモデルでタスクを複数回解き、`1 -成功率` を難易度スコアとする |
| Quality-Diversity アーカイブ | ACES (NeurIPS 2024) | 概念ニッチごとに最も難しいタスクを保持し、多様性と品質を両立 |

### 進化のサイクル

1. **初期化**: シードタスクから概念を抽出し、概念グラフとQDアーカイブを構築
2. **世代ループ**（各世代）:
   - アーカイブからトーナメント選択で親タスクを選択
   - 概念グラフからランダムウォークで概念組み合わせをサンプリング
   - 5種類の変異戦略からランダムに選択し、親タスクを難化変異
   - ソルバーモデルで難易度を推定（`1 - 成功率`）
   - 難易度が閾値以上のタスクをQDアーカイブに追加
3. **結果返却**: アーカイブの上位N件を難易度順で返却

### 設定例

完全な設定例は [examples/evolutionary_taskgen.yaml](examples/evolutionary_taskgen.yaml) を参照してください。

```yaml
taskgen:
  type: evolutionary
  evolutionary:
    seed_tasks:
      - id: seed_001
        title: "DP: Longest Increasing Subsequence"
        prompt: "動的計画法を用いて最長増加部分列の長さを求める関数を実装せよ。"
    max_generations: 10
    population_size: 15
    difficulty_min: 0.4
    solver_attempts: 3
    mutation_strategies:
      - extend
      - fuse_same_type
      - fuse_cross_type
      - add_constraint
      - combine_concepts
```

### 主なパラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `seed_tasks` | `[]` | 進化の起点となるシードタスク |
| `max_generations` | 10 | 進化の最大世代数 |
| `population_size` | 20 | 1世代あたりの変異試行数 |
| `difficulty_min` | 0.3 | アーカイブ追加の難易度閾値 (0.0=易, 1.0=難) |
| `solver_attempts` | 3 | 難易度推定のソルバー試行回数 |
| `concept_graph_steps` | 6 | 概念グラフのランダムウォーク歩数 |
| `archive_capacity` | 100 | QDアーカイブの最大容量 |
| `temperature` | 0.7 | LLM生成の温度パラメータ |
| `tournament_size` | 3 | トーナメント選択のサイズ |

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
