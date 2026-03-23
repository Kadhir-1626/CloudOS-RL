import { Cpu, Activity, Zap } from 'lucide-react'

export default function Layout({ children }) {
  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header style={{
        background: 'var(--surface)',
        borderBottom: '1px solid var(--border)',
        padding: '0 32px',
        height: 56,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        zIndex: 100,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: 'linear-gradient(135deg, var(--accent), var(--accent2))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Zap size={16} color="#fff" />
          </div>
          <span style={{ fontWeight: 700, fontSize: 15 }}>CloudOS</span>
          <span style={{
            background: 'var(--surface2)', border: '1px solid var(--border)',
            padding: '1px 7px', borderRadius: 4, fontSize: 10,
            color: 'var(--accent)', fontWeight: 600, letterSpacing: '0.05em',
          }}>RL</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--muted)', fontSize: 12 }}>
          <Activity size={13} />
          <span>AI-Native Multi-Cloud Scheduler</span>
        </div>
      </header>

      <main style={{ flex: 1, padding: '28px 32px', maxWidth: 1280, margin: '0 auto', width: '100%' }}>
        {children}
      </main>
    </div>
  )
}