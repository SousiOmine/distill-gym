import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, BatchRunResult } from '../api/client'

interface ConfigEntry {
  id: string
  name: string
  yaml: string
  selected: boolean
}

let nextId = 1
function genId() { return `cfg_${nextId++}` }

function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = () => reject(new Error(`Failed to read ${file.name}`))
    reader.readAsText(file)
  })
}

export function NewRunPage() {
  const navigate = useNavigate()
  const [configs, setConfigs] = useState<ConfigEntry[]>([])
  const [results, setResults] = useState<Record<string, BatchRunResult>>({})
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [examples, setExamples] = useState<string[]>([])
  const [loadingExamples, setLoadingExamples] = useState(false)
  const [manualYaml, setManualYaml] = useState('')
  const [showManualEditor, setShowManualEditor] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

  const addConfig = (name: string, yaml: string) => {
    setConfigs(prev => [...prev, { id: genId(), name, yaml, selected: true }])
  }

  const removeConfig = (id: string) => {
    setConfigs(prev => prev.filter(c => c.id !== id))
  }

  const toggleSelect = (id: string) => {
    setConfigs(prev => prev.map(c => c.id === id ? { ...c, selected: !c.selected } : c))
  }

  const selectAll = () => {
    setConfigs(prev => prev.map(c => ({ ...c, selected: true })))
  }

  const deselectAll = () => {
    setConfigs(prev => prev.map(c => ({ ...c, selected: false })))
  }

  const handleFiles = async (files: FileList) => {
    const fileArray = Array.from(files).filter(f =>
      f.name.endsWith('.yaml') || f.name.endsWith('.yml')
    )
    if (fileArray.length === 0) {
      setError('YAMLファイル（.yaml / .yml）のみ対応しています')
      return
    }
    setError('')
    for (const file of fileArray) {
      try {
        const yaml = await readFileAsText(file)
        addConfig(file.name, yaml)
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : `${file.name}の読み込みに失敗しました`)
      }
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) handleFiles(e.target.files)
    if (e.target) e.target.value = ''
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    if (e.dataTransfer.files.length > 0) handleFiles(e.dataTransfer.files)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = () => setDragOver(false)

  const loadExamples = async () => {
    if (examples.length > 0) return
    setLoadingExamples(true)
    try {
      const names = await api.listExamples()
      setExamples(names)
    } catch {
      // ignore
    } finally {
      setLoadingExamples(false)
    }
  }

  const handleExampleSelect = async (name: string) => {
    if (!name) return
    try {
      const result = await api.loadExampleConfig(name)
      addConfig(name, result.yaml)
      setError('')
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : '設定ファイルの読み込みに失敗しました')
    }
  }

  const handleAddManual = () => {
    if (!manualYaml.trim()) return
    addConfig(`設定${configs.length + 1}`, manualYaml)
    setManualYaml('')
    setShowManualEditor(false)
  }

  const handleSubmit = async () => {
    const selected = configs.filter(c => c.selected)
    if (selected.length === 0) return

    setSubmitting(true)
    setError('')
    setResults({})

    const yamls = selected.map(c => c.yaml)

    try {
      const res = await api.createRunsBatch(yamls)
      const resultMap: Record<string, BatchRunResult> = {}
      selected.forEach((c, i) => {
        resultMap[c.id] = res.runs[i]
      })
      setResults(resultMap)

      const successIds = res.runs.filter(r => r.run_id).map(r => r.run_id!)
      if (successIds.length === 1) {
        navigate(`/runs/${successIds[0]}`)
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : '一括実行に失敗しました')
    } finally {
      setSubmitting(false)
    }
  }

  const selectedCount = configs.filter(c => c.selected).length
  const hasResults = Object.keys(results).length > 0
  const allSuccessful = hasResults && Object.values(results).every(r => r.run_id)
  const anyFailed = hasResults && Object.values(results).some(r => !r.run_id)

  return (
    <div>
      <h2>新規Run</h2>

      <div
        onClick={() => fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        style={{
          border: `2px dashed ${dragOver ? '#1565c0' : '#ccc'}`,
          borderRadius: 8,
          padding: '1.5rem',
          textAlign: 'center',
          cursor: 'pointer',
          background: dragOver ? '#e3f2fd' : '#fafafa',
          marginBottom: '0.75rem',
          transition: 'all 0.2s',
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".yaml,.yml"
          multiple
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        <div>
          <div style={{ fontSize: '1.2rem', marginBottom: '0.3rem' }}>YAMLファイルをドロップ</div>
          <div style={{ fontSize: '0.85rem', color: '#666' }}>またはクリックで複数選択（.yaml / .yml）</div>
        </div>
      </div>

      <div style={{ marginBottom: '0.75rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
        <button onClick={loadExamples} style={{ padding: '0.3rem 0.8rem', cursor: 'pointer' }}>
          サンプル設定を読み込む
        </button>
        {loadingExamples && <span style={{ fontSize: '0.85rem', color: '#666' }}>読み込み中...</span>}
        {examples.length > 0 && (
          <select
            onChange={e => e.target.value && handleExampleSelect(e.target.value)}
            style={{ padding: '0.25rem' }}
          >
            <option value="">選択してください</option>
            {examples.map(name => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        )}

        <button
          onClick={() => setShowManualEditor(!showManualEditor)}
          style={{ padding: '0.3rem 0.8rem', cursor: 'pointer', marginLeft: 'auto' }}
        >
          {showManualEditor ? '閉じる' : '設定を追加'}
        </button>
      </div>

      {showManualEditor && (
        <div style={{ marginBottom: '0.75rem', padding: '0.75rem', border: '1px solid #ddd', borderRadius: 4, background: '#f9f9f9' }}>
          <div style={{ marginBottom: '0.5rem' }}>
            <label style={{ fontWeight: 600 }}>YAML設定を入力:</label>
          </div>
          <textarea
            value={manualYaml}
            onChange={e => setManualYaml(e.target.value)}
            rows={10}
            placeholder="YAML設定を貼り付けてください..."
            style={{
              width: '100%',
              fontFamily: 'monospace',
              fontSize: '0.85rem',
              padding: '0.5rem',
              border: '1px solid #ccc',
              borderRadius: 4,
              boxSizing: 'border-box',
            }}
          />
          <div style={{ marginTop: '0.5rem' }}>
            <button
              onClick={handleAddManual}
              disabled={!manualYaml.trim()}
              style={{
                padding: '0.3rem 1rem',
                background: manualYaml.trim() ? '#1565c0' : '#ccc',
                color: '#fff',
                border: 'none',
                borderRadius: 4,
                cursor: manualYaml.trim() ? 'pointer' : 'not-allowed',
              }}
            >
              追加
            </button>
          </div>
        </div>
      )}

      {configs.length === 0 && !showManualEditor && (
        <p style={{ color: '#888', fontStyle: 'italic' }}>
          設定がありません。YAMLファイルをドロップするか、「設定を追加」ボタンから追加してください。
        </p>
      )}

      {configs.length > 0 && (
        <div style={{ marginBottom: '0.75rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <span style={{ fontWeight: 600 }}>設定一覧 ({configs.length}件)</span>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button onClick={selectAll} style={{ padding: '0.2rem 0.6rem', cursor: 'pointer', fontSize: '0.85rem' }}>全て選択</button>
              <button onClick={deselectAll} style={{ padding: '0.2rem 0.6rem', cursor: 'pointer', fontSize: '0.85rem' }}>選択解除</button>
            </div>
          </div>

          {configs.map(cfg => {
            const res = results[cfg.id]
            const isSubmitting = submitting && cfg.selected
            const done = res && (res.status === 'submitted' || res.status === 'error')
            return (
              <div
                key={cfg.id}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '0.5rem',
                  padding: '0.5rem',
                  marginBottom: '0.4rem',
                  border: '1px solid #ddd',
                  borderRadius: 4,
                  background: done ? (res.run_id ? '#f1f8e9' : '#fce4ec') : '#fff',
                  opacity: isSubmitting ? 0.6 : 1,
                }}
              >
                <input
                  type="checkbox"
                  checked={cfg.selected}
                  onChange={() => toggleSelect(cfg.id)}
                  disabled={submitting}
                  style={{ marginTop: '0.2rem' }}
                />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: 500, fontSize: '0.9rem' }}>{cfg.name}</div>
                  <details style={{ fontSize: '0.8rem', color: '#666', marginTop: '0.2rem' }}>
                    <summary>YAMLを表示</summary>
                    <pre style={{
                      marginTop: '0.3rem',
                      padding: '0.5rem',
                      background: '#f5f5f5',
                      borderRadius: 4,
                      overflow: 'auto',
                      maxHeight: 200,
                      fontSize: '0.75rem',
                      lineHeight: 1.3,
                    }}>{cfg.yaml}</pre>
                  </details>
                  {done && (
                    <div style={{ marginTop: '0.3rem', fontSize: '0.85rem' }}>
                      {res.run_id ? (
                        <a href={`/runs/${res.run_id}`} onClick={e => { e.preventDefault(); navigate(`/runs/${res.run_id}`) }}
                          style={{ color: '#2e7d32' }}>
                          ✅ {res.run_id.slice(0, 20)}…
                        </a>
                      ) : (
                        <span style={{ color: '#c62828' }}>⚠ {res.error || '失敗'}</span>
                      )}
                    </div>
                  )}
                  {isSubmitting && !done && (
                    <span style={{ fontSize: '0.85rem', color: '#1565c0' }}>送信中...</span>
                  )}
                </div>
                <button
                  onClick={() => removeConfig(cfg.id)}
                  disabled={submitting}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: submitting ? '#ccc' : '#999',
                    cursor: submitting ? 'not-allowed' : 'pointer',
                    fontSize: '1.1rem',
                    padding: '0 0.3rem',
                  }}
                  title="削除"
                >
                  ✕
                </button>
              </div>
            )
          })}
        </div>
      )}

      <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
        <button
          onClick={handleSubmit}
          disabled={submitting || selectedCount === 0}
          style={{
            padding: '0.5rem 1.5rem',
            background: selectedCount === 0 || submitting ? '#ccc' : '#1565c0',
            color: '#fff',
            border: 'none',
            borderRadius: 4,
            cursor: submitting || selectedCount === 0 ? 'not-allowed' : 'pointer',
          }}
        >
          {submitting ? '実行中...' : `${selectedCount}件の設定を一括実行`}
        </button>

        {allSuccessful && !anyFailed && (
          <span style={{ fontSize: '0.9rem', color: '#2e7d32' }}>✅ 全てのRunを作成しました</span>
        )}
        {hasResults && anyFailed && (
          <span style={{ fontSize: '0.9rem', color: '#c62828' }}>
            ⚠ 一部のRunの作成に失敗しました
          </span>
        )}
      </div>

      {error && <p style={{ color: 'red', marginTop: '0.5rem' }}>{error}</p>}
    </div>
  )
}
