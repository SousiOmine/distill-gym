import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, Run } from '../api/client'

const STATUS_COLORS: Record<string, string> = {
  completed: '#2e7d32',
  failed: '#c62828',
  running: '#1565c0',
  pending: '#f57f17',
}

export function RunsPage() {
  const [runs, setRuns] = useState<Run[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    api.listRuns()
      .then(setRuns)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <p>Loading runs...</p>
  if (error) return <p style={{ color: 'red' }}>{error}</p>

  return (
    <div>
      <h2>Runs</h2>
      {runs.length === 0 && <p>No runs yet. <Link to="/new">Create one</Link>.</p>}
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ textAlign: 'left', borderBottom: '2px solid #ddd' }}>
            <th>Name</th>
            <th>Status</th>
            <th>Harness</th>
            <th>Model</th>
            <th>Created</th>
            <th>Result</th>
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
                  {run.status}
                </span>
              </td>
              <td>{run.harness_type}</td>
              <td style={{ fontSize: '0.85rem' }}>{run.model}</td>
              <td style={{ fontSize: '0.85rem' }}>{new Date(run.created_at).toLocaleString()}</td>
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
