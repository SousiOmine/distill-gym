import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'

const DEFAULT_YAML = `version: 1
run:
  name: my-run
  task_count: 2
  cleanup: always

provider:
  type: openai_compatible
  name: default
  base_url: https://api.deepinfra.com/v1/openai
  api_key_env: OPENAI_API_KEY
  model: deepseek-coder-v2

sandbox:
  type: git_repository
  engine: podman
  repo_url: https://github.com/example/project.git
  ref: main
  image: docker.io/library/python:3.12-bookworm

harness:
  type: opencode
  install:
    commands:
      - npm install -g opencode-ai
  completion:
    max_idle_seconds: 120

taskgen:
  type: repo_auto
`

export function NewRunPage() {
  const navigate = useNavigate()
  const [yaml, setYaml] = useState(DEFAULT_YAML)
  const [filename, setFilename] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [examples, setExamples] = useState<string[]>([])
  const [loadingExamples, setLoadingExamples] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

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

  const handleFile = async (file: File) => {
    if (!file.name.endsWith('.yaml') && !file.name.endsWith('.yml')) {
      setError('YAMLファイル（.yaml / .yml）のみ対応しています')
      return
    }
    setError('')
    try {
      const result = await api.uploadConfig(file)
      setYaml(result.yaml)
      setFilename(result.filename)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'ファイルのアップロードに失敗しました')
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
    if (e.target) e.target.value = ''
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = () => setDragOver(false)

  const handleExampleSelect = async (name: string) => {
    try {
      const result = await api.loadExampleConfig(name)
      setYaml(result.yaml)
      setFilename(name)
      setError('')
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : '設定ファイルの読み込みに失敗しました')
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSaving(true)
    setError('')
    try {
      const result = await api.createRun(yaml)
      navigate(`/runs/${result.run_id}`)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : '実行の作成に失敗しました')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div>
      <h2>新規実行</h2>

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
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        {filename ? (
          <div>
            <strong>{filename}</strong>
            <div style={{ fontSize: '0.85rem', color: '#666', marginTop: '0.3rem' }}>クリックまたはドロップで変更</div>
          </div>
        ) : (
          <div>
            <div style={{ fontSize: '1.2rem', marginBottom: '0.3rem' }}>YAMLファイルをドロップ</div>
            <div style={{ fontSize: '0.85rem', color: '#666' }}>またはクリックして選択（.yaml / .yml）</div>
          </div>
        )}
      </div>

      <div style={{ marginBottom: '0.75rem' }}>
        <button
          onClick={loadExamples}
          style={{ padding: '0.3rem 0.8rem', cursor: 'pointer', marginRight: '0.5rem' }}
        >
          サンプル設定を読み込む
        </button>
        {loadingExamples && <span style={{ fontSize: '0.85rem', color: '#666' }}>読み込み中...</span>}
        {examples.length > 0 && (
          <select
            onChange={e => e.target.value && handleExampleSelect(e.target.value)}
            style={{ padding: '0.25rem', marginLeft: '0.5rem' }}
          >
            <option value="">選択してください</option>
            {examples.map(name => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        )}
      </div>

      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '0.5rem' }}>
          <label style={{ fontWeight: 600 }}>YAML設定:</label>
        </div>
        <textarea
          value={yaml}
          onChange={e => setYaml(e.target.value)}
          rows={24}
          style={{
            width: '100%',
            fontFamily: 'monospace',
            fontSize: '0.85rem',
            padding: '0.5rem',
            border: '1px solid #ccc',
            borderRadius: 4,
          }}
        />
        <div style={{ marginTop: '0.5rem' }}>
          <button
            type="submit"
            disabled={saving}
            style={{
              padding: '0.5rem 1.5rem',
              background: '#1565c0',
              color: '#fff',
              border: 'none',
              borderRadius: 4,
              cursor: saving ? 'not-allowed' : 'pointer',
            }}
          >
            {saving ? '実行中...' : '実行開始'}
          </button>
        </div>
        {error && <p style={{ color: 'red' }}>{error}</p>}
      </form>
    </div>
  )
}
