import { useEffect, useState, useRef } from 'react'
import { Link } from 'react-router-dom'
import { api, Run } from '../api/client'

const STATUS_LABELS: Record<string, string> = {
  completed: '完了',
  failed: '失敗',
  running: '実行中',
  pending: '待機中',
}

const STATUS_COLORS: Record<string, string> = {
  completed: '#2e7d32',
  failed: '#c62828',
  running: '#1565c0',
  pending: '#f57f17',
}

function relativeTime(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime()
  const sec = Math.floor(diff / 1000)
  if (sec < 60) return `${sec}秒前`
  const min = Math.floor(sec / 60)
  if (min < 60) return `${min}分前`
  const hour = Math.floor(min / 60)
  if (hour < 24) return `${hour}時間前`
  return `${Math.floor(hour / 24)}日前`
}

export function RunsPage() {
  const [runs, setRuns] = useState<Run[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchRuns = () => {
    api.listRuns()
      .then(res => setRuns(res.runs))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    fetchRuns()
    intervalRef.current = setInterval(fetchRuns, 5000)
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [])

  if (loading) return <p>読み込み中...</p>
  if (error) return <p style={{ color: 'red' }}>{error}</p>

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
        <h2 style={{ margin: 0 }}>実行一覧</h2>
        <button onClick={fetchRuns} style={{ padding: '0.3rem 0.8rem', cursor: 'pointer' }}>
          再読み込み
        </button>
      </div>
      {runs.length === 0 && <p>まだ実行がありません。<Link to="/new">新規作成</Link></p>}
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ textAlign: 'left', borderBottom: '2px solid #ddd' }}>
            <th>名前</th>
            <th>状態</th>
            <th>ハーネス</th>
            <th>モデル</th>
            <th>作成日時</th>
            <th>結果</th>
          </tr>
        </thead>
        <tbody>
          {runs.map(run => (
            <tr key={run.id} style={{ borderBottom: '1px solid #eee' }}>
              <td><Link to={`/runs/${run.id}`}>{run.name}</Link></td>
              <td>
                <span style={{
                  color: STATUS_COLORS[run.status] || '#999',
                  fontWeight: 600,
                  fontSize: '0.85rem',
                }}>
                  {STATUS_LABELS[run.status] || run.status}
                </span>
              </td>
              <td>{run.harness_type}</td>
              <td style={{ fontSize: '0.85rem' }}>{run.model}</td>
              <td style={{ fontSize: '0.85rem' }} title={new Date(run.created_at).toLocaleString()}>
                {relativeTime(run.created_at)}
              </td>
              <td>
                {run.success === true ? '✓' : run.success === false ? '✗' : '-'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
