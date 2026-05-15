import { Outlet, Link } from 'react-router-dom'

export function Layout() {
  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '1rem', fontFamily: 'system-ui, sans-serif' }}>
      <nav style={{ display: 'flex', gap: '1rem', padding: '0.5rem 0', borderBottom: '2px solid #eee', marginBottom: '1rem' }}>
        <Link to="/" style={{ fontWeight: 700, color: '#333', textDecoration: 'none' }}>distill-gym</Link>
        <Link to="/" style={{ color: '#555' }}>Run一覧</Link>
        <Link to="/new" style={{ color: '#555' }}>新規Run</Link>
        <Link to="/export" style={{ color: '#555' }}>エクスポート</Link>
      </nav>
      <main>
        <Outlet />
      </main>
    </div>
  )
}
