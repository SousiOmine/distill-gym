import { useState } from 'react'
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
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSaving(true)
    setError('')
    try {
      const result = await api.createRun(yaml)
      navigate(`/runs/${result.run_id}`)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to create run')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div>
      <h2>New Run</h2>
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '0.5rem' }}>
          <label style={{ fontWeight: 600 }}>YAML Configuration:</label>
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
            {saving ? 'Starting...' : 'Start Run'}
          </button>
        </div>
        {error && <p style={{ color: 'red' }}>{error}</p>}
      </form>
    </div>
  )
}
