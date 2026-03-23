import { useEffect, useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { getDecisions } from '../api/client'

const CLOUD_COLORS = { aws: '#f59e0b', gcp: '#3b82f6', azure: '#6366f1', hybrid: '#10b981' }

export default function DecisionTable() {
  const [decisions, setDecisions] = useState([])
  const [loading,   setLoading]   = useState(false)
  const [lastRefresh, setLastRefresh] = useState(null)

  const load = async () => {
    setLoading(true)
    try {
      const d = await getDecisions(50)
      setDecisions(d.decisions || [])
      setLastRefresh(new Date())
    } catch {}
    finally { setLoading(false) }
  }

  useEffect(() => { load(); const t = setInterval(load, 10000); return () => clearInterval(t) }, [])

  const th = {
    padding: '8px 12px', textAlign: 'left',
    fontSize: 11, color: 'var(--muted)',
    textTransform: 'uppercase', letterSpacing: '0.05em',
    borderBottom: '1px solid var(--border)',
    fontWeight: 600,
  }
  const td = {
    padding: '10px 12px',
    fontSize: 12,
    borderBottom: '1px solid var(--border)',
    verticalAlign: 'middle',
  }

  return (
    <div className="card">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <div>
          <span style={{ fontWeight: 600, fontSize: 15 }}>Decision History</span>
          <span style={{ color: 'var(--muted)', fontSize: 12, marginLeft: 8 }}>
            {decisions.length} recent
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {lastRefresh && (
            <span style={{ color: 'var(--muted)', fontSize: 11 }}>
              {lastRefresh.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={load}
            disabled={loading}
            style={{
              padding: '6px 10px', background: 'var(--surface2)',
              border: '1px solid var(--border)', color: 'var(--text)',
              display: 'flex', alignItems: 'center', gap: 5,
            }}
          >
            <RefreshCw size={12} style={loading ? { animation: 'spin 1s linear infinite' } : {}} />
            Refresh
          </button>
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        {decisions.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '32px 0', color: 'var(--muted)' }}>
            No decisions yet. Submit a workload above.
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {['ID', 'Cloud', 'Region', 'Instance', 'Purchase', 'Cost/hr', 'Cost Save', 'CO₂ Save', 'Latency'].map(h => (
                  <th key={h} style={th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {decisions.map(d => (
                <tr key={d.decision_id}
                  style={{ transition: 'background 0.1s' }}
                  onMouseEnter={e => e.currentTarget.style.background = 'var(--surface2)'}
                  onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                >
                  <td style={{ ...td, fontFamily: 'monospace', color: 'var(--muted)' }}>
                    {d.decision_id.slice(0, 8)}
                  </td>
                  <td style={td}>
                    <span style={{
                      fontWeight: 700,
                      color: CLOUD_COLORS[d.cloud] || 'var(--text)',
                      textTransform: 'uppercase',
                    }}>
                      {d.cloud}
                    </span>
                  </td>
                  <td style={td}>{d.region}</td>
                  <td style={{ ...td, fontFamily: 'monospace', fontSize: 11 }}>{d.instance_type}</td>
                  <td style={td}>
                    <span className={`badge ${d.purchase_option === 'spot' ? 'badge-green' : 'badge-blue'}`}>
                      {d.purchase_option.replace('_', ' ')}
                    </span>
                  </td>
                  <td style={{ ...td, fontFamily: 'monospace' }}>
                    ${d.estimated_cost_per_hr?.toFixed(4)}
                  </td>
                  <td style={{ ...td, color: 'var(--green)', fontWeight: 600 }}>
                    {d.cost_savings_pct?.toFixed(1)}%
                  </td>
                  <td style={{ ...td, color: '#34d399', fontWeight: 600 }}>
                    {d.carbon_savings_pct?.toFixed(1)}%
                  </td>
                  <td style={{ ...td, color: d.latency_ms < 200 ? 'var(--green)' : 'var(--yellow)' }}>
                    {d.latency_ms?.toFixed(0)}ms
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}