import { useEffect, useState, useRef } from 'react'
import { useParams, Link } from 'react-router-dom'
import { api, Run, Task, Artifact } from '../api/client'

const STATUS_LABELS: Record<string, string> = {
  completed: '完了',
  failed: '失敗',
  running: 'Running',
  pending: '待機中',
}

const STATUS_COLORS: Record<string, string> = {
  completed: '#2e7d32',
  failed: '#c62828',
  running: '#1565c0',
  pending: '#f57f17',
}

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>()
  const [run, setRun] = useState<Run | null>(null)
  const [tasks, setTasks] = useState<Task[]>([])
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [elapsed, setElapsed] = useState('')
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const isActive = run?.status === 'running' || run?.status === 'pending'

  const fetchData = () => {
    if (!runId) return
    Promise.all([
      api.getRun(runId),
      api.listTasks(runId),
      api.listArtifacts(runId),
    ])
      .then(([r, t, a]) => {
        setRun(r)
        setTasks(t)
        setArtifacts(a)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    fetchData()
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [runId])

  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    if (isActive) {
      intervalRef.current = setInterval(fetchData, 3000)
    }
  }, [isActive])

  useEffect(() => {
    if (!isActive || !run?.created_at) {
      setElapsed('')
      return
    }
    const update = () => {
      const start = new Date(run.created_at).getTime()
      const sec = Math.floor((Date.now() - start) / 1000)
      const m = Math.floor(sec / 60)
      const s = sec % 60
      setElapsed(`${m}分${s}秒`)
    }
    update()
    const t = setInterval(update, 1000)
    return () => clearInterval(t)
  }, [isActive, run?.created_at])

  if (loading) return <p>読み込み中...</p>
  if (error) return <p style={{ color: 'red' }}>{error}</p>
  if (!run) return <p>Runが見つかりません</p>

  const taskProgress = tasks.length > 0
    ? `${tasks.filter(t => t.status === 'completed' || t.status === 'failed').length} / ${tasks.length}`
    : '-'

  return (
    <div>
      <Link to="/">← Run一覧に戻る</Link>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2>{run.name}</h2>
        <button onClick={fetchData} style={{ padding: '0.3rem 0.8rem', cursor: 'pointer' }}>
          再読み込み
        </button>
      </div>
      <table style={{ borderCollapse: 'collapse', width: '100%' }}>
        <tbody>
          {[
            ['ID', run.id],
            ['状態', <span key="s" style={{ color: STATUS_COLORS[run.status], fontWeight: 600 }}>{STATUS_LABELS[run.status] || run.status}</span>],
            ['経過時間', isActive ? elapsed : '-'],
            ['ハーネス', run.harness_type],
            ['モデル', run.model],
            ['プロバイダ', run.provider_name],
            ['サンドボックス', run.sandbox_type],
            ['リポジトリ', run.repo_url],
            ['コミット', run.commit_hash || '-'],
            ['成功', run.success === null ? '-' : run.success ? 'はい' : 'いいえ'],
            ['作成日時', new Date(run.created_at).toLocaleString()],
            ['Task進捗', taskProgress],
          ].map(([label, value]) => (
            <tr key={label} style={{ borderBottom: '1px solid #eee' }}>
              <td style={{ fontWeight: 600, padding: '0.3rem 0.5rem', width: 140 }}>{label}</td>
              <td style={{ padding: '0.3rem 0.5rem' }}>{value}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {run.error_message && (
        <div style={{ background: '#fde', padding: '0.5rem', borderRadius: 4, margin: '1rem 0' }}>
          <strong>エラー:</strong> {run.error_message}
        </div>
      )}

      <h3>Task ({tasks.length})</h3>
      {tasks.length > 0 && (
        <div style={{ background: '#e8f5e9', borderRadius: 4, padding: '0.3rem 0.6rem', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
          進捗: {taskProgress}
        </div>
      )}
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ textAlign: 'left', borderBottom: '2px solid #ddd' }}>
            <th>タイトル</th>
            <th>状態</th>
            <th>終了コード</th>
            <th>テスト結果</th>
          </tr>
        </thead>
        <tbody>
          {tasks.map(task => (
            <tr key={task.id} style={{ borderBottom: '1px solid #eee' }}>
              <td><Link to={`/runs/${runId}/tasks/${task.id}`}>{task.title || task.id}</Link></td>
              <td><span style={{ color: STATUS_COLORS[task.status], fontWeight: 600, fontSize: '0.85rem' }}>{STATUS_LABELS[task.status] || task.status}</span></td>
              <td>{task.exit_code ?? '-'}</td>
              <td>{task.tests_passed === null ? '-' : task.tests_passed ? '✓' : '✗'}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>アーティファクト ({artifacts.length})</h3>
      <ul>
        {artifacts.map(a => (
          <li key={a.id}>
            <strong>{a.kind}</strong> — {a.size} bytes — <a href={`/artifacts/${a.path}`}>表示</a>
          </li>
        ))}
      </ul>
    </div>
  )
}
