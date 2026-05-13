import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { api, Run, Task, Artifact } from '../api/client'

export function TaskDetailPage() {
  const { runId, taskId } = useParams<{ runId: string; taskId: string }>()
  const [task, setTask] = useState<Task | null>(null)
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!runId || !taskId) return
    Promise.all([
      api.getTask(runId, taskId),
      api.listArtifacts(runId, taskId),
    ])
      .then(([t, a]) => {
        setTask(t)
        setArtifacts(a)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [runId, taskId])

  if (loading) return <p>Loading...</p>
  if (error) return <p style={{ color: 'red' }}>{error}</p>
  if (!task) return <p>Task not found</p>

  return (
    <div>
      <Link to={`/runs/${runId}`}>← Back to run</Link>
      <h2>{task.title || task.id}</h2>
      <table style={{ borderCollapse: 'collapse', width: '100%' }}>
        <tbody>
          {[
            ['ID', task.id],
            ['Status', task.status],
            ['Exit Code', task.exit_code?.toString() ?? '-'],
            ['Success', task.success === null ? '-' : task.success ? 'Yes' : 'No'],
            ['Tests Passed', task.tests_passed === null ? '-' : task.tests_passed ? 'Yes' : 'No'],
            ['Started', task.started_at ? new Date(task.started_at).toLocaleString() : '-'],
            ['Finished', task.finished_at ? new Date(task.finished_at).toLocaleString() : '-'],
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
          <strong>Error:</strong> {task.error_message}
        </div>
      )}

      <h3>Prompt</h3>
      <pre style={{ background: '#f5f5f5', padding: '1rem', borderRadius: 4, overflow: 'auto', maxHeight: 300 }}>
        {task.prompt}
      </pre>

      <h3>Artifacts ({artifacts.length})</h3>
      <ul>
        {artifacts.map(a => (
          <li key={a.id}>
            <strong>{a.kind}</strong> — {a.size} bytes — <a href={`/artifacts/${a.path}`}>view</a>
          </li>
        ))}
      </ul>
    </div>
  )
}
