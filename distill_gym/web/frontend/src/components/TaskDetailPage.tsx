import { useEffect, useState, useRef } from 'react'
import { useParams, Link } from 'react-router-dom'
import { api, Run, Task, Artifact } from '../api/client'

const STATUS_LABELS: Record<string, string> = {
  completed: '完了',
  failed: '失敗',
  running: '実行中',
  pending: '待機中',
}

export function TaskDetailPage() {
  const { runId, taskId } = useParams<{ runId: string; taskId: string }>()
  const [task, setTask] = useState<Task | null>(null)
  const [runName, setRunName] = useState('')
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [elapsed, setElapsed] = useState('')
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const isActive = task?.status === 'running' || task?.status === 'pending'

  const fetchData = () => {
    if (!runId || !taskId) return
    Promise.all([
      api.getTask(runId, taskId),
      api.listArtifacts(runId, taskId),
      api.getRun(runId),
    ])
      .then(([t, a, r]) => {
        setTask(t)
        setArtifacts(a)
        setRunName(r.name)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    fetchData()
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [runId, taskId])

  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    if (isActive) {
      intervalRef.current = setInterval(fetchData, 3000)
    }
  }, [isActive])

  useEffect(() => {
    if (!isActive || !task?.started_at) {
      setElapsed('')
      return
    }
    const update = () => {
      const start = new Date(task.started_at!).getTime()
      const sec = Math.floor((Date.now() - start) / 1000)
      const m = Math.floor(sec / 60)
      const s = sec % 60
      setElapsed(`${m}分${s}秒`)
    }
    update()
    const t = setInterval(update, 1000)
    return () => clearInterval(t)
  }, [isActive, task?.started_at])

  if (loading) return <p>読み込み中...</p>
  if (error) return <p style={{ color: 'red' }}>{error}</p>
  if (!task) return <p>タスクが見つかりません</p>

  return (
    <div>
      <Link to={`/runs/${runId}`}>← 実行に戻る</Link>
      <h2>{task.title || task.id}</h2>
      <table style={{ borderCollapse: 'collapse', width: '100%' }}>
        <tbody>
          {[
            ['ID', task.id],
            ['状態', STATUS_LABELS[task.status] || task.status],
            ['経過時間', isActive ? elapsed : '-'],
            ['終了コード', task.exit_code?.toString() ?? '-'],
            ['成功', task.success === null ? '-' : task.success ? 'はい' : 'いいえ'],
            ['テスト結果', task.tests_passed === null ? '-' : task.tests_passed ? '合格' : '不合格'],
            ['開始', task.started_at ? new Date(task.started_at).toLocaleString() : '-'],
            ['終了', task.finished_at ? new Date(task.finished_at).toLocaleString() : '-'],
          ].map(([label, value]) => (
            <tr key={label} style={{ borderBottom: '1px solid #eee' }}>
              <td style={{ fontWeight: 600, padding: '0.3rem 0.5rem', width: 140 }}>{label}</td>
              <td style={{ padding: '0.3rem 0.5rem' }}>{value}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {task.error_message && (
        <div style={{ background: '#fde', padding: '0.5rem', borderRadius: 4, margin: '1rem 0' }}>
          <strong>エラー:</strong> {task.error_message}
        </div>
      )}

      <h3>プロンプト</h3>
      <pre style={{ background: '#f5f5f5', padding: '1rem', borderRadius: 4, overflow: 'auto', maxHeight: 300, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
        {task.prompt}
      </pre>

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
