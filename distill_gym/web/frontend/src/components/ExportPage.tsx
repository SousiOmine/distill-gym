import { useEffect, useState } from 'react'
import { api, Run } from '../api/client'

export function ExportPage() {
  const [runs, setRuns] = useState<Run[]>([])
  const [selected, setSelected] = useState<string[]>([])
  const [format, setFormat] = useState('openai-messages')
  const [includeFailed, setIncludeFailed] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [result, setResult] = useState<{ count: number; url?: string } | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api.listRuns()
      .then(res => setRuns(res.runs))
      .catch(() => {})
  }, [])

  const toggleRun = (id: string) => {
    setSelected(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    )
  }

  const handleExport = async () => {
    if (selected.length === 0) return
    setExporting(true)
    setError('')
    setResult(null)
    try {
      if (selected.length === 1) {
        const res = await api.exportRun(selected[0], format, includeFailed)
        setResult(res)
      } else {
        const res = await api.mergeExport(selected, format)
        setResult(res)
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'エクスポートに失敗しました')
    } finally {
      setExporting(false)
    }
  }

  return (
    <div>
      <h2>データセットエクスポート</h2>

      <div style={{ marginBottom: '1rem' }}>
        <label style={{ fontWeight: 600 }}>形式:</label>
        <select value={format} onChange={e => setFormat(e.target.value)}
          style={{ marginLeft: '0.5rem', padding: '0.25rem' }}>
          <option value="openai-messages">OpenAI Messages</option>
        </select>
      </div>

      <div style={{ marginBottom: '1rem' }}>
        <label>
          <input type="checkbox" checked={includeFailed} onChange={e => setIncludeFailed(e.target.checked)} />
          {' '}失敗した実行も含める
        </label>
      </div>

      <h3>実行を選択</h3>
      {runs.length === 0 && <p>実行がありません。</p>}
      <div style={{ maxHeight: 300, overflow: 'auto', border: '1px solid #ddd', borderRadius: 4, padding: '0.5rem' }}>
        {runs.map(run => (
          <label key={run.id} style={{ display: 'block', padding: '0.25rem 0' }}>
            <input
              type="checkbox"
              checked={selected.includes(run.id)}
              onChange={() => toggleRun(run.id)}
            />
            {' '}{run.name} ({run.id.slice(0, 20)}…) — {run.status}
            {run.success === true ? ' ✓' : run.success === false ? ' ✗' : ''}
          </label>
        ))}
      </div>

      <button
        onClick={handleExport}
        disabled={exporting || selected.length === 0}
        style={{
          marginTop: '1rem',
          padding: '0.5rem 1.5rem',
          background: selected.length === 0 ? '#ccc' : '#2e7d32',
          color: '#fff',
          border: 'none',
          borderRadius: 4,
          cursor: exporting ? 'not-allowed' : 'pointer',
        }}
      >
        {exporting ? 'エクスポート中...' : `${selected.length}件の実行をエクスポート`}
      </button>

      {error && <p style={{ color: 'red' }}>{error}</p>}
      {result && (
        <p style={{ color: '#2e7d32' }}>
          {result.count}件の会話をエクスポートしました。
          {result.url && <> <a href={result.url}>ダウンロード</a></>}
        </p>
      )}
    </div>
  )
}
