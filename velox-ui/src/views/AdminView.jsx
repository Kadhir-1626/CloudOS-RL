import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import CountUp from 'react-countup'
import {
  Users, Shield, Activity, Settings, Database,
  AlertTriangle, CheckCircle, Clock, Zap,
} from 'lucide-react'
import { getStatus, getDecisions } from '../api/client'
import { useAuth } from '../auth/AuthContext'

const CLOUD_COLORS = {
  aws: '#f59e0b', gcp: '#3b82f6', azure: '#6366f1', hybrid: '#10b981',
}

function AdminKpiCard({ icon: Icon, label, value, rawValue, sub, color, index }) {
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
            <CountUp end={rawValue} duration={1.5} decimals={0} preserveValue />
          ) : value}
        </div>
        {sub && <div style={{ color: 'var(--muted)', fontSize: 11, marginTop: 4 }}>{sub}</div>}
      </div>
    </motion.div>
  )
}

function RolePermissionsMatrix() {
  const roles = [
    { name: 'viewer',    cells: [false, true,  false, false, false] },
    { name: 'user',      cells: [true,  true,  false, false, false] },
    { name: 'engineer',  cells: [true,  true,  true,  true,  false] },
    { name: 'admin',     cells: [true,  true,  true,  true,  true]  },
    { name: 'executive', cells: [false, true,  false, true,  false] },
  ]

  const columns = ['Schedule', 'Decisions', 'Explain', 'Metrics', 'Admin']

  return (
    <motion.div
      className="card"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.3, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
    >
      <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
        <Shield size={14} color="var(--accent)" /> Role Permissions Matrix
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: 'var(--surface2)' }}>
              <th style={{ textAlign: 'left', padding: '10px 12px', fontWeight: 700, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontSize: 10 }}>Role</th>
              {columns.map(col => (
                <th key={col} style={{ textAlign: 'center', padding: '10px 12px', fontWeight: 700, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontSize: 10 }}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {roles.map((role, ri) => (
              <tr key={role.name} style={{ background: ri % 2 === 0 ? 'transparent' : 'var(--surface2)' }}>
                <td style={{ padding: '10px 12px', fontWeight: 600, textTransform: 'capitalize', color: 'var(--text2)' }}>{role.name}</td>
                {role.cells.map((allowed, ci) => (
                  <td key={ci} style={{ textAlign: 'center', padding: '10px 12px' }}>
                    {allowed ? (
                      <CheckCircle size={14} color="var(--green)" style={{ margin: '0 auto' }} />
                    ) : (
                      <span style={{ color: 'var(--muted)', fontSize: 14 }}>—</span>
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </motion.div>
  )
}

function ActivityFeed({ decisions }) {
  if (!decisions.length) {
    return (
      <motion.div
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        style={{ minHeight: 300, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12, textAlign: 'center' }}
      >
        <Activity size={32} color="var(--muted)" />
        <div style={{ color: 'var(--muted)', fontSize: 13 }}>No activity yet</div>
        <div style={{ color: 'var(--muted)', fontSize: 11 }}>Decisions will appear here as workloads are scheduled</div>
      </motion.div>
    )
  }

  return (
    <motion.div
      className="card"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.3, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      style={{ minHeight: 300 }}
    >
      <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
        <Activity size={14} color="var(--accent)" /> Activity Feed
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
        {decisions.slice(0, 5).map((d, i) => {
          const cloudColor = CLOUD_COLORS[d.cloud] || 'var(--accent)'
          return (
            <motion.div
              key={d.decision_id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.35 + i * 0.08, duration: 0.35 }}
              style={{
                display: 'flex', alignItems: 'center', gap: 16,
                padding: '14px 16px',
                background: i % 2 === 0 ? 'transparent' : 'var(--surface2)',
                borderBottom: i < 4 ? '1px solid var(--border)' : 'none',
              }}
            >
              <div style={{ width: 10, height: 10, borderRadius: '50%', background: cloudColor, flexShrink: 0, boxShadow: `0 0 8px ${cloudColor}80` }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, fontSize: 13 }}>
                  <span style={{ color: cloudColor, textTransform: 'uppercase', fontSize: 11 }}>{d.cloud}</span>
                  <span style={{ color: 'var(--muted)' }}>→</span>
                  <span style={{ color: 'var(--text2)' }}>{d.region}</span>
                </div>
                <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 2 }}>
                  {d.instance_type} · {d.purchase_option?.replace(/_/g, ' ') || 'on-demand'}
                </div>
              </div>
              <div style={{ textAlign: 'right', minWidth: 80 }}>
                <div style={{ color: 'var(--green)', fontWeight: 900, fontSize: 14 }}>
                  {d.cost_savings_pct?.toFixed(1)}% saved
                </div>
                <div style={{ color: 'var(--muted)', fontSize: 10, marginTop: 2 }}>
                  {d.latency_ms?.toFixed(0)}ms
                </div>
              </div>
            </motion.div>
          )
        })}
      </div>
    </motion.div>
  )
}

function SystemConfigDisplay() {
  const config = [
    { key: 'MODEL_PATH',      value: 'models/best/best_model.zip',           comment: null },
    { key: 'VECNORM_PATH',    value: 'models/vec_normalize.pkl',             comment: null },
    { key: 'REWARD_ALPHA',    value: '0.30',                                 comment: 'cost' },
    { key: 'REWARD_BETA',     value: '0.28',                                 comment: 'latency' },
    { key: 'REWARD_GAMMA',    value: '0.22',                                 comment: 'carbon' },
    { key: 'REWARD_DELTA',    value: '0.20',                                 comment: 'sla' },
    { key: 'REWARD_EPSILON',  value: '0.00',                                 comment: 'migration — disabled' },
    { key: 'ENT_COEF',        value: '0.05',                                 comment: null },
    { key: 'TIMESTEPS',       value: '2,000,000',                            comment: null },
  ]

  return (
    <motion.div
      id="section-control"
      className="card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      style={{ background: '#0a0f1e', border: '1px solid var(--border)' }}
    >
      <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text)' }}>
        <Settings size={14} color="var(--accent)" /> Runtime Configuration
      </div>
      <pre style={{ margin: 0, fontFamily: 'monospace', fontSize: 12, lineHeight: 2, color: '#e2e8f0' }}>
        {config.map((line, i) => (
          <span key={line.key} style={{ display: 'block' }}>
            <span style={{ color: '#64748b', marginRight: 16, minWidth: 160, display: 'inline-block' }}>{line.key}</span>
            <span style={{ color: '#94a3b8', marginRight: 16, minWidth: 80, display: 'inline-block' }}>{line.value}</span>
            {line.comment && <span style={{ color: '#475569' }}># {line.comment}</span>}
          </span>
        ))}
      </pre>
    </motion.div>
  )
}

export default function AdminView() {
  const { user } = useAuth()
  const [status, setStatus] = useState(null)
  const [decisions, setDecisions] = useState([])

  useEffect(() => {
    let isMounted = true
    const load = async () => {
      try {
        const s = await getStatus()
        if (isMounted) setStatus(s)
      } catch { /* non-fatal */ }
      try {
        const d = await getDecisions(5)
        if (isMounted) setDecisions(d?.decisions || [])
      } catch { if (isMounted) setDecisions([]) }
    }
    load()
    const t = setInterval(load, 15000)
    return () => { isMounted = false; clearInterval(t) }
  }, [])

  return (
    <div style={{ padding: '24px', maxWidth: 1400, margin: '0 auto' }}>
      {/* SECTION 1 — HEADER */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 32 }}
      >
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 800, marginBottom: 4 }}>Control Center</h1>
          <p style={{ color: 'var(--muted)', fontSize: 13 }}>Full visibility. Full control.</p>
        </div>
        <motion.div
          style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: 'var(--green)' }}
        >
          <motion.div
            animate={{ scale: [1, 1.3, 1], opacity: [1, 0.5, 1] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
            style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--green)', boxShadow: '0 0 12px var(--green)' }}
          />
          <span style={{ fontWeight: 600 }}>Velox is healthy</span>
        </motion.div>
      </motion.div>

      {/* SECTION 2 — ADMIN KPI CARDS */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 32 }}>
        <AdminKpiCard index={0} icon={Shield}   label="Security"         value="Active"                rawValue={null} sub="JWT + RBAC enforced"          color="var(--green)"   />
        <AdminKpiCard index={1} icon={Users}    label="Active Roles"     value="5"                     rawValue={5}    sub="viewer · user · engineer · admin · executive" color="var(--accent)"  />
        <AdminKpiCard index={2} icon={Database} label="Decisions Stored" value={status?.decisions_served ?? 0} rawValue={status?.decisions_served ?? 0} sub="In-memory store"              color="var(--accent2)" />
        <AdminKpiCard index={3} icon={Activity} label="API Status"       value="Healthy"               rawValue={null} sub="FastAPI + Uvicorn"            color="var(--green2)"  />
      </div>

      {/* SECTION 3 — TWO COLUMNS */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 32 }}>
        {/* Left column — Role Permissions Matrix */}
        <RolePermissionsMatrix />

        {/* Right column — Recent Activity Feed */}
        <ActivityFeed decisions={decisions} />
      </div>

      {/* SECTION 4 — SYSTEM CONFIGURATION DISPLAY */}
      <SystemConfigDisplay />

      {/* Footer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        style={{ marginTop: 32, textAlign: 'center', color: 'var(--muted)', fontSize: 11, borderTop: '1px solid var(--border)', paddingTop: 20 }}
      >
        Velox · Multi-Cloud Workload Scheduler · PPO + SHAP + Kafka + Kubernetes
      </motion.div>
    </div>
  )
}