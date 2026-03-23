import { useEffect, useState } from 'react'
import { Cpu, TrendingDown, Leaf, Clock } from 'lucide-react'
import { getStatus, getDecisions } from '../api/client'

function Metric({ icon: Icon, label, value, sub, color }) {
  return (
    <div className="card" style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 14 }}>
      <div style={{
        width: 42, height: 42, borderRadius: 10,
        background: `${color}18`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexShrink: 0,
      }}>
        <Icon size={18} color={color} />
      </div>
      <div>
        <div style={{ color: 'var(--muted)', fontSize: 11, marginBottom: 2 }}>{label}</div>
        <div style={{ fontSize: 22, fontWeight: 700, lineHeight: 1 }}>{value}</div>
        {sub && <div style={{ color: 'var(--muted)', fontSize: 11, marginTop: 3 }}>{sub}</div>}
      </div>
    </div>
  )
}

export default function MetricsBar() {
  const [status,    setStatus]    = useState(null)
  const [decisions, setDecisions] = useState([])

  const refresh = async () => {
    try { setStatus(await getStatus()) }    catch {}
    try {
      const d = await getDecisions(50)
      setDecisions(d.decisions || [])
    } catch {}
  }

  useEffect(() => { refresh(); const t = setInterval(refresh, 8000); return () => clearInterval(t) }, [])

  const avgLatency = decisions.length
    ? (decisions.reduce((s, d) => s + (d.latency_ms || 0), 0) / decisions.length).toFixed(0)
    : '—'

  const avgCost = decisions.length
    ? (decisions.reduce((s, d) => s + (d.cost_savings_pct || 0), 0) / decisions.length).toFixed(1)
    : '—'

  const avgCarbon = decisions.length
    ? (decisions.reduce((s, d) => s + (d.carbon_savings_pct || 0), 0) / decisions.length).toFixed(1)
    : '—'

  return (
    <div style={{ display: 'flex', gap: 16, marginBottom: 24 }}>
      <Metric
        icon={Cpu}
        label="Decisions Served"
        value={status?.decisions_served ?? '—'}
        sub={status?.agent_loaded ? 'PPO model active' : 'Loading…'}
        color="var(--accent)"
      />
      <Metric
        icon={Clock}
        label="Avg Latency"
        value={avgLatency === '—' ? '—' : `${avgLatency}ms`}
        sub="inference p50"
        color="var(--accent2)"
      />
      <Metric
        icon={TrendingDown}
        label="Avg Cost Savings"
        value={avgCost === '—' ? '—' : `${avgCost}%`}
        sub="vs on-demand baseline"
        color="var(--green)"
      />
      <Metric
        icon={Leaf}
        label="Avg Carbon Savings"
        value={avgCarbon === '—' ? '—' : `${avgCarbon}%`}
        sub="vs us-east-1 baseline"
        color="#34d399"
      />
    </div>
  )
}