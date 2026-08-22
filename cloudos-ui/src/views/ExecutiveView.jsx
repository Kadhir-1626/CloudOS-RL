import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import CountUp from 'react-countup'
import {
  TrendingDown, Leaf, Clock, Cpu, BarChart2,
  Cloud, MapPin, Server, Tag, Activity,
  Brain, Zap, Shield,
} from 'lucide-react'
import { getDecisions, getStatus } from '../api/client'
import { useAuth } from '../auth/AuthContext'

const CLOUD_COLORS = {
  aws: '#f59e0b', gcp: '#3b82f6', azure: '#6366f1', hybrid: '#10b981',
}

function KpiCard({ icon: Icon, label, value, rawValue, sub, color, index }) {
  return (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.08, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ y: -3, boxShadow: `0 12px 32px rgba(0,0,0,0.3), 0 0 0 1px ${color}30` }}
      style={{ flex: 1, display: 'flex', gap: 16, alignItems: 'center', cursor: 'default' }}
    >
      <motion.div
        animate={{ boxShadow: [`0 0 0px ${color}40`, `0 0 16px ${color}40`, `0 0 0px ${color}40`] }}
        transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut', delay: index * 0.5 }}
        style={{
          width: 48, height: 48, borderRadius: 12,
          background: `${color}18`, border: `1px solid ${color}30`,
          display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
        }}
      >
        <Icon size={20} color={color} />
      </motion.div>
      <div>
        <div style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 700, marginBottom: 3 }}>
          {label}
        </div>
        <div style={{ fontSize: 26, fontWeight: 900, lineHeight: 1, letterSpacing: '-0.02em', color }}>
          {rawValue != null ? (
            <CountUp end={rawValue} duration={1.5} decimals={Number.isInteger(rawValue) ? 0 : 1}
              suffix={value.includes('%') ? '%' : value.includes('ms') ? 'ms' : ''} preserveValue />
          ) : value}
        </div>
        {sub && <div style={{ color: 'var(--muted)', fontSize: 11, marginTop: 4 }}>{sub}</div>}
      </div>
    </motion.div>
  )
}

function CloudDonut({ decisions }) {
  const counts = { aws: 0, azure: 0, gcp: 0, hybrid: 0 }
  decisions.forEach(d => { if (d.cloud && counts[d.cloud] !== undefined) counts[d.cloud]++ })
  const total = Object.values(counts).reduce((a, b) => a + b, 0)
  if (total === 0) return (
    <div style={{ textAlign: 'center', color: 'var(--muted)', padding: '40px 0', fontSize: 12 }}>No decisions yet</div>
  )

  const r = 52, cx = 70, cy = 70, stroke = 14
  const circumference = 2 * Math.PI * r
  let offset = 0
  const segments = Object.entries(counts).filter(([, v]) => v > 0).map(([cloud, count]) => {
    const pct = count / total
    const dasharray = pct * circumference
    const seg = { cloud, count, pct, dasharray, offset }
    offset += dasharray
    return seg
  })

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
      <svg width={140} height={140} style={{ flexShrink: 0 }}>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--surface2)" strokeWidth={stroke} />
        {segments.map((seg, i) => (
          <motion.circle
            key={seg.cloud}
            cx={cx} cy={cy} r={r}
            fill="none"
            stroke={CLOUD_COLORS[seg.cloud]}
            strokeWidth={stroke}
            strokeDasharray={`${seg.dasharray} ${circumference}`}
            strokeDashoffset={-seg.offset}
            strokeLinecap="round"
            initial={{ strokeDasharray: `0 ${circumference}` }}
            animate={{ strokeDasharray: `${seg.dasharray} ${circumference}` }}
            transition={{ delay: i * 0.15 + 0.2, duration: 0.8, ease: 'easeOut' }}
            style={{ transform: `rotate(-90deg)`, transformOrigin: `${cx}px ${cy}px` }}
          />
        ))}
        <text x={cx} y={cy - 6} textAnchor="middle" fill="var(--text)" fontSize="18" fontWeight="800">{total}</text>
        <text x={cx} y={cy + 10} textAnchor="middle" fill="var(--muted)" fontSize="9" fontWeight="600">DECISIONS</text>
      </svg>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {segments.map(seg => (
          <motion.div
            key={seg.cloud}
            initial={{ opacity: 0, x: 10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}
          >
            <div style={{ width: 10, height: 10, borderRadius: 2, background: CLOUD_COLORS[seg.cloud], flexShrink: 0 }} />
            <span style={{ color: 'var(--text2)', textTransform: 'uppercase', fontWeight: 600, fontSize: 11 }}>{seg.cloud}</span>
            <span style={{ color: 'var(--muted)', marginLeft: 'auto' }}>{(seg.pct * 100).toFixed(0)}%</span>
          </motion.div>
        ))}
      </div>
    </div>
  )
}

function DecisionSnapshotCard({ decision }) {
  if (!decision) return (
    <motion.div
      className="card"
      initial={{ opacity: 0 }} animate={{ opacity: 1 }}
      style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: 180, gap: 10, color: 'var(--muted)' }}
    >
      <motion.div animate={{ y: [0, -6, 0] }} transition={{ duration: 2.5, repeat: Infinity }}>
        <Activity size={24} />
      </motion.div>
      <div style={{ fontSize: 13, fontWeight: 600 }}>No decisions yet</div>
      <div style={{ fontSize: 12 }}>Decisions will appear here once the system processes workloads</div>
    </motion.div>
  )

  const cloudColor = CLOUD_COLORS[decision.cloud] || 'var(--accent)'

  return (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      style={{ borderLeft: `3px solid ${cloudColor}` }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 14 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span style={{ fontWeight: 700, color: cloudColor, textTransform: 'uppercase', fontSize: 13 }}>
              {decision.cloud || '—'}
            </span>
            <span className="badge badge-green">
              {decision.purchase_option?.replace(/_/g, ' ') || '—'}
            </span>
          </div>
          <div style={{ color: 'var(--muted)', fontSize: 11, fontFamily: 'monospace' }}>
            {decision.decision_id ? `${decision.decision_id.slice(0, 16)}…` : '—'}
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Latency</div>
          <div style={{ fontSize: 20, fontWeight: 800, color: 'var(--green)' }}>
            {typeof decision.latency_ms === 'number' ? `${decision.latency_ms.toFixed(0)}ms` : '—'}
          </div>
        </div>
      </div>

      {[
        { icon: MapPin, label: 'Region',   value: decision.region || '—' },
        { icon: Server, label: 'Instance', value: decision.instance_type || '—' },
        { icon: Tag,    label: 'Est. Cost/hr', value: typeof decision.estimated_cost_per_hr === 'number' ? `$${decision.estimated_cost_per_hr.toFixed(4)}` : '—' },
      ].map(({ icon: Icon, label, value }) => (
        <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '7px 0', borderBottom: '1px solid var(--border)', fontSize: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--muted)' }}><Icon size={11} />{label}</div>
          <span style={{ fontWeight: 600 }}>{value}</span>
        </div>
      ))}

      <div style={{ display: 'flex', gap: 10, marginTop: 12 }}>
        {[
          { label: 'Cost Savings', value: typeof decision.cost_savings_pct === 'number' ? `${decision.cost_savings_pct.toFixed(1)}%` : '—', raw: decision.cost_savings_pct, color: 'var(--green)', bg: 'rgba(16,185,129,0.08)', border: 'rgba(16,185,129,0.2)' },
          { label: 'Carbon Savings', value: typeof decision.carbon_savings_pct === 'number' ? `${decision.carbon_savings_pct.toFixed(1)}%` : '—', raw: decision.carbon_savings_pct, color: 'var(--green2)', bg: 'rgba(52,211,153,0.08)', border: 'rgba(52,211,153,0.2)' },
        ].map(({ label, value, raw, color, bg, border }) => (
          <div key={label} style={{ flex: 1, textAlign: 'center', padding: '8px 0', background: bg, borderRadius: 8, border: `1px solid ${border}` }}>
            <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 4 }}>{label}</div>
            <div style={{ fontWeight: 900, color, fontSize: 18 }}>{value}</div>
            {raw != null && raw > 0 && (
              <motion.div
                initial={{ width: 0 }} animate={{ width: `${Math.min(raw, 100)}%` }}
                transition={{ duration: 1, delay: 0.3 }}
                style={{ height: 3, background: color, borderRadius: 2, margin: '6px auto 0', maxWidth: '80%' }}
              />
            )}
          </div>
        ))}
      </div>
    </motion.div>
  )
}

function CloudIntelligenceCard() {
  const rows = [
    { icon: Brain, label: 'AI Model', value: 'PPO · 2M Steps' },
    { icon: Zap, label: 'Explainability', value: 'SHAP Attribution' },
    { icon: Leaf, label: 'Carbon Signal', value: '50 Regions Tracked' },
    { icon: Shield, label: 'SLA Enforcement', value: 'Real-time' },
  ]

  return (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      style={{ marginTop: 24 }}
    >
      <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
        <Brain size={14} color="var(--accent)" /> Placement Intelligence
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
        {rows.map((row, i) => (
          <motion.div
            key={row.label}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.35 + i * 0.08, duration: 0.35 }}
            style={{
              display: 'flex', alignItems: 'center', gap: 12,
              padding: '14px 16px',
              background: i % 2 === 0 ? 'transparent' : 'var(--surface2)',
              borderBottom: i < rows.length - 1 ? '1px solid var(--border)' : 'none',
              borderLeft: `3px solid var(--accent)`,
            }}
          >
            <row.icon size={16} color="var(--accent)" style={{ flexShrink: 0 }} />
            <div style={{ flex: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontWeight: 600, fontSize: 13 }}>{row.label}</span>
              <span style={{ color: 'var(--muted)', fontSize: 12 }}>{row.value}</span>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  )
}

export default function ExecutiveView() {
  const { user } = useAuth()
  const [decisions, setDecisions] = useState([])
  const [status, setStatus] = useState(null)
  const [latestDecision, setLatestDecision] = useState(null)

  useEffect(() => {
    let isMounted = true
    const load = async () => {
      try { const s = await getStatus(); if (isMounted) setStatus(s) } catch { /* non-fatal */ }
      try {
        const d = await getDecisions(100)
        const list = d?.decisions || []
        if (!isMounted) return
        setDecisions(list)
        setLatestDecision(list.length > 0 ? list[0] : null)
      } catch { if (isMounted) { setDecisions([]); setLatestDecision(null) } }
    }
    load()
    const t = setInterval(load, 15000)
    return () => { isMounted = false; clearInterval(t) }
  }, [])

  const n = decisions.length
  const avgCost    = n ? decisions.reduce((s, d) => s + (d.cost_savings_pct || 0), 0) / n : 0
  const avgCarbon  = n ? decisions.reduce((s, d) => s + (d.carbon_savings_pct || 0), 0) / n : 0
  const avgLatency = n ? decisions.reduce((s, d) => s + (d.latency_ms || 0), 0) / n : 0
  const monthlySavingsEst = n > 0 ? `$${((avgCost / 100) * 0.096 * 730 * n).toFixed(0)}` : '—'

  return (
    <div>
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 24 }}
      >
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 800, marginBottom: 4 }}>Executive Dashboard</h1>
          <p style={{ color: 'var(--muted)', fontSize: 13 }}>Velox performance overview · Live data · Refreshes every 15s</p>
        </div>
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          style={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8, padding: '8px 14px', fontSize: 12, color: 'var(--muted)' }}
        >
          Welcome, <strong style={{ color: 'var(--text)' }}>{user?.username}</strong>
          <span style={{ marginLeft: 8, color: 'var(--accent2)', fontWeight: 700, fontSize: 10, textTransform: 'uppercase' }}>{user?.role}</span>
        </motion.div>
      </motion.div>

      {/* KPI row */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 24 }}>
        <KpiCard index={0} icon={Cpu}        label="Total Decisions"   value={String(status?.decisions_served ?? 0)} rawValue={status?.decisions_served ?? 0} sub="AI placements made"       color="var(--accent)"  />
        <KpiCard index={1} icon={TrendingDown} label="Avg Cost Savings" value={`${avgCost.toFixed(1)}%`}              rawValue={avgCost}                          sub="vs on-demand baseline"  color="var(--green)"   />
        <KpiCard index={2} icon={Leaf}        label="Avg Carbon Savings" value={`${avgCarbon.toFixed(1)}%`}           rawValue={avgCarbon}                        sub="vs us-east-1 baseline"  color="var(--green2)"  />
        <KpiCard index={3} icon={Clock}       label="Avg Latency"       value={`${avgLatency.toFixed(0)}ms`}          rawValue={avgLatency}                       sub="scheduling decision time" color="var(--accent2)" />
      </div>

      {/* Cloud Intelligence Card */}
      <CloudIntelligenceCard />

      {/* Two column */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginTop: 24, marginBottom: 24 }}>
        {/* Cloud distribution donut */}
        <motion.div
          className="card"
          initial={{ opacity: 0, x: -16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4, duration: 0.4 }}
        >
          <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 20, display: 'flex', alignItems: 'center', gap: 8 }}>
            <BarChart2 size={14} />Cloud Distribution
          </div>
          <CloudDonut decisions={decisions} />
        </motion.div>

        {/* Business impact */}
        <motion.div
          className="card"
          initial={{ opacity: 0, x: 16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.45, duration: 0.4 }}
        >
          <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 16 }}>What Velox Saved You</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[
              { label: 'Avg cost reduction per decision',   value: `${avgCost.toFixed(1)}%`    },
              { label: 'Avg carbon reduction per decision', value: `${avgCarbon.toFixed(1)}%`   },
              { label: 'Avg scheduling latency',            value: `${avgLatency.toFixed(0)}ms` },
              { label: 'Projected monthly savings',  value: monthlySavingsEst            },
              { label: 'AI scheduler', value: status?.agent_loaded ? '✅ Operational' : '⚠ Initialising' },
            ].map(({ label, value }, i) => (
              <motion.div
                key={label}
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 + i * 0.06 }}
                style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: '9px 12px', background: 'var(--surface2)', borderRadius: 8, fontSize: 13,
                }}
              >
                <span style={{ color: 'var(--muted)' }}>{label}</span>
                <span style={{ fontWeight: 700 }}>{value}</span>
              </motion.div>
            ))}
          </div>
          <p style={{ color: 'var(--muted)', fontSize: 11, marginTop: 12, lineHeight: 1.6 }}>
            Based on session activity. Actual savings scale with workload volume.
          </p>
        </motion.div>
      </div>

      {/* Latest decision snapshot */}
      <div>
        <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
          <Cloud size={14} />Latest Decision Snapshot
          <span style={{ color: 'var(--muted)', fontWeight: 400, fontSize: 12 }}>· read-only</span>
        </div>
        <AnimatePresence mode="wait">
          <DecisionSnapshotCard key={latestDecision?.decision_id || 'empty'} decision={latestDecision} />
        </AnimatePresence>
      </div>
    </div>
  )
}