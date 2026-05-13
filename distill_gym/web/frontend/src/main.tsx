import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { RunsPage } from './components/RunsPage'
import { RunDetailPage } from './components/RunDetailPage'
import { TaskDetailPage } from './components/TaskDetailPage'
import { NewRunPage } from './components/NewRunPage'
import { ExportPage } from './components/ExportPage'
import { Layout } from './components/Layout'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<RunsPage />} />
          <Route path="/runs/:runId" element={<RunDetailPage />} />
          <Route path="/runs/:runId/tasks/:taskId" element={<TaskDetailPage />} />
          <Route path="/new" element={<NewRunPage />} />
          <Route path="/export" element={<ExportPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
)
