import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { api, Run, Task, Artifact } from '../api/client'

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>()
  const [run, setRun] = useState<Run | null>(null)
  const [tasks, setTasks] = useState<Task[]>([])
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
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
  }, [runId])

  if (loading) return <p>Loading...</p>
  if (error) return <p style={{ color: 'red' }}>{error}</p>
  if (!run) return <p>Run not found</p>

  return (
    <div>
      <Link to="/">← Back to runs</Link>
      <h2>{run.name}</h2>
      <table style={{ borderCollapse: 'collapse', width: '100%' }}>
        <tbody>
          {[
            ['ID', run.id],
            ['Status', run.status],
            ['Harness', run.harness_type],
            ['Model', run.model],
            ['Provider', run.provider_name],
            ['Sandbox', run.sandbox_type],
            ['Repository', run.repo_url],
            ['Commit', run.commit_hash],
            ['Success', run.success === null ? '-' : run.success ? 'Yes' : 'No'],
            ['Created', new Date(run.created_at).toLocaleString()],
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
          <strong>Error:</strong> {run.error_message}
        </div>
      )}

      <h3>Tasks ({tasks.length})</h3>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ textAlign: 'left', borderBottom: '2px solid #ddd' }}>
            <th>Title</th>
            <th>Status</th>
            <th>Exit Code</th>
            <th>Tests Passed</th>
          </tr>
        </thead>
        <tbody>
          {tasks.map(task => (
            <tr key={task.id} style={{ borderBottom: '1px solid #eee' }}>
              <td><Link to={`/runs/${runId}/tasks/${task.id}`}>{task.title || task.id}</Link></td>
              <td>{task.status}</td>
              <td>{task.exit_code ?? '-'}</td>
              <td>{task.tests_passed === null ? '-' : task.tests_passed ? '✓' : '✗'}</td>
            </tr>
          ))}
        </tbody>
      </table>

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
