const API_BASE = '/api'

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const err = await res.text()
    throw new Error(`API error ${res.status}: ${err}`)
  }
  return res.json()
}

export interface Run {
  id: string
  name: string
  status: string
  harness_type: string
  model: string
  provider_name: string
  sandbox_type: string
  repo_url: string
  success: boolean | null
  error_message: string | null
  created_at: string
  updated_at: string
  commit_hash: string
  task_count: number
}

export interface Task {
  id: string
  run_id: string
  title: string
  prompt: string
  status: string
  exit_code: number | null
  success: boolean | null
  test_command: string | null
  tests_passed: boolean | null
  error_message: string | null
  started_at: string | null
  finished_at: string | null
}

export interface Artifact {
  id: string
  task_id: string
  run_id: string
  kind: string
  path: string
  size: number
}

export const api = {
  listRuns: () => fetchJSON<Run[]>('/runs'),

  getRun: (id: string) => fetchJSON<Run>(`/runs/${id}`),

  listTasks: (runId: string) => fetchJSON<Task[]>(`/runs/${runId}/tasks`),

  getTask: (runId: string, taskId: string) =>
    fetchJSON<Task>(`/runs/${runId}/tasks/${taskId}`),

  listArtifacts: (runId: string, taskId?: string) => {
    const qs = taskId ? `?task_id=${taskId}` : ''
    return fetchJSON<Artifact[]>(`/runs/${runId}/artifacts${qs}`)
  },

  createRun: (config: string) =>
    fetchJSON<{ run_id: string }>('/runs', {
      method: 'POST',
      body: JSON.stringify({ config_yaml: config }),
    }),

  exportRun: (runId: string, format: string, includeFailed: boolean) =>
    fetchJSON<{ count: number; url: string }>('/export', {
      method: 'POST',
      body: JSON.stringify({ run_id: runId, format, include_failed: includeFailed }),
    }),

  mergeExport: (runIds: string[], format: string) =>
    fetchJSON<{ count: number; url: string }>('/export/merge', {
      method: 'POST',
      body: JSON.stringify({ run_ids: runIds, format }),
    }),

  listExamples: () =>
    fetchJSON<string[]>('/config/examples'),

  loadExampleConfig: (name: string) =>
    fetchJSON<{ yaml: string }>(`/config/examples/${name}`),

  uploadConfig: async (file: File) => {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(`${API_BASE}/config/upload`, {
      method: 'POST',
      body: form,
    })
    if (!res.ok) {
      const err = await res.text()
      throw new Error(`Upload error ${res.status}: ${err}`)
    }
    return res.json() as Promise<{ yaml: string; filename: string }>
  },
}
