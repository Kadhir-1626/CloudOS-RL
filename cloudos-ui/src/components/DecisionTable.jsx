import { useEffect, useState, useCallback, useRef, memo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { RefreshCw, ArrowDown, Filter } from 'lucide-react'
import { getDecisions } from '../api/client'
import { SkeletonTableRows } from './Skeleton'

const CLOUD_COLORS = {
  aws: '#f59e0b', gcp: '#3b82f6', azure: '#6366f1', hybrid: '#10b981',
}

const HEADERS = ['ID', 'Cloud', 'Region', 'Instance', 'Purchase', 'Cost/hr', 'Cost Δ', 'CO₂ Δ', 'Latency']

const cellStyle = {
  padding: '10px 12px', fontSize: 12,
  borderBottom: '1px solid var(--border)', verticalAlign: 'middle',
}

const DecisionRow = memo(function DecisionRow({ decision, index, isNew }) {
  const estimatedCost  = decision?.estimated_cost_per_hr != null ? Number(decision.estimated_cost_per_hr) : null
  const costSavings    = decision?.cost_savings_pct != null ? Number(decision.cost_savings_pct) : null
  const carbonSavings  = decision?.carbon_savings_pct != null ? Number(decision.carbon_savings_pct) : null
  const latencyMs      = decision?.latency_ms != null ? Number(decision.latency_ms) : null
  const purchaseOption = decision?.purchase_option || 'on_demand'
  const cloudColor     = CLOUD_COLORS[decision?.cloud] || 'var(--accent)'

  return (
    <motion.tr
      initial={{ opacity: 0, x: -8 }}
      animate={{
        opacity: 1, x: 0,
        backgroundColor: isNew ? ['rgba(16,185,129,0.12)', 'transparent'] : 'transparent',
      }}
      transition={{
        opacity: { delay: index * 0.04, duration: 0.25 },
        x: { delay: index * 0.04, duration: 0.25 },
        backgroundColor: { duration: 1.5, delay: 0.1 },
      }}
      whileHover={{ backgroundColor: 'var(--surface2)' }}
      style={{ borderLeft: `2px solid ${cloudColor}20`, cursor: 'default' }}
    >
      <td style={{ ...cellStyle, fontFamily: 'monospace', color: 'var(--muted)', fontSize: 11 }}>
        {decision?.decision_id ? decision.decision_id.slice(0, 8) : '—'}
      </td>

      <td style={cellStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <motion.div
            animate={{ opacity: [1, 0.4, 1] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut', delay: index * 0.3 }}
            style={{ width: 6, height: 6, borderRadius: '50%', background: cloudColor, flexShrink: 0 }}
          />
          <span style={{ fontWeight: 700, color: cloudColor, textTransform: 'uppercase', fontSize: 11 }}>
            {decision?.cloud || '—'}
          </span>
        </div>
      </td>

      <td style={{ ...cellStyle, color: 'var(--text2)' }}>{decision?.region || '—'}</td>

      <td style={{ ...cellStyle, fontFamily: 'monospace', fontSize: 11 }}>{decision?.instance_type || '—'}</td>

      <td style={cellStyle}>
        <span className={`badge ${purchaseOption === 'spot' ? 'badge-green' : 'badge-blue'}`}>
          {purchaseOption.replace(/_/g, ' ')}
        </span>
      </td>

      <td style={{ ...cellStyle, fontFamily: 'monospace' }}>
        {estimatedCost != null ? `$${estimatedCost.toFixed(4)}` : '—'}
      </td>

      <td style={{ ...cellStyle, color: 'var(--green)', fontWeight: 700 }}>
        {costSavings != null ? `${costSavings.toFixed(1)}%` : '—'}
      </td>

      <td style={{ ...cellStyle, color: 'var(--green2)', fontWeight: 700 }}>
        {carbonSavings != null ? `${carbonSavings.toFixed(1)}%` : '—'}
      </td>

      <td style={{
        ...cellStyle, fontWeight: 700,
        color: latencyMs == null ? 'var(--muted)' : latencyMs < 200 ? 'var(--green)' : latencyMs < 500 ? 'var(--yellow)' : 'var(--red)',
      }}>
        {latencyMs != null ? `${latencyMs.toFixed(0)}ms` : '—'}
      </td>
    </motion.tr>
  )
})

export default function DecisionTable() {
  const [decisions, setDecisions]   = useState([])
  const [loading, setLoading]       = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [lastTs, setLastTs]         = useState(null)
  const [newCount, setNewCount]     = useState(0)
  const [newIds, setNewIds]         = useState(new Set())
  const [cloudFilter, setCloudFilter] = useState('all')

  const prevCountRef       = useRef(0)
  const prevIdsRef         = useRef(new Set())
  const mountedRef         = useRef(true)
  const newBadgeTimeoutRef = useRef(null)

  const load = useCallback(async (manual = false) => {
    if (manual) setRefreshing(true)
    try {
      const response = await getDecisions(50)
      if (!mountedRef.current) return
      const list = response?.decisions || []
      setDecisions(list)
      setLastTs(new Date())

      const currentIds = new Set(list.map(d => d.decision_id))
      const fresh = list.filter(d => !prevIdsRef.current.has(d.decision_id)).map(d => d.decision_id)

      if (!manual && fresh.length > 0 && prevCountRef.current > 0) {
        setNewCount(fresh.length)
        setNewIds(new Set(fresh))
        if (newBadgeTimeoutRef.current) clearTimeout(newBadgeTimeoutRef.current)
        newBadgeTimeoutRef.current = setTimeout(() => {
          if (mountedRef.current) { setNewCount(0); setNewIds(new Set()) }
        }, 3000)
      }

      prevCountRef.current = list.length
      prevIdsRef.current   = currentIds
    } catch { /* silent */ }
    finally {
      if (mountedRef.current) { setLoading(false); setRefreshing(false) }
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    load(false)
    const id = setInterval(() => load(false), 12000)
    return () => {
      mountedRef.current = false
      clearInterval(id)
      if (newBadgeTimeoutRef.current) clearTimeout(newBadgeTimeoutRef.current)
    }
  }, [load])

  const filtered = cloudFilter === 'all' ? decisions : decisions.filter(d => d.cloud === cloudFilter)
  const clouds   = ['all', ...Object.keys(CLOUD_COLORS)]

  const headerCellStyle = {
    padding: '9px 12px', textAlign: 'left', fontSize: 10, color: 'var(--muted)',
    textTransform: 'uppercase', letterSpacing: '0.07em', borderBottom: '1px solid var(--border)',
    fontWeight: 700, background: 'var(--surface)', position: 'sticky', top: 0, zIndex: 1,
  }

  return (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontWeight: 700, fontSize: 15 }}>Decision History</span>
          <span style={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 20, padding: '2px 8px', fontSize: 11, color: 'var(--muted)', fontWeight: 600 }}>
            {filtered.length}
          </span>
          <AnimatePresence>
            {newCount > 0 && (
              <motion.span
                className="badge badge-green"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
              >
                <ArrowDown size={9} /> +{newCount} new
              </motion.span>
            )}
          </AnimatePresence>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {lastTs && <span style={{ color: 'var(--muted)', fontSize: 11 }}>{lastTs.toLocaleTimeString()}</span>}
          <motion.button
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            type="button"
            onClick={() => load(true)}
            disabled={refreshing}
            style={{
              padding: '6px 12px', background: 'var(--surface2)',
              border: '1px solid var(--border)', color: 'var(--text2)',
              display: 'flex', alignItems: 'center', gap: 6, fontWeight: 600,
            }}
          >
            <motion.div animate={refreshing ? { rotate: 360 } : { rotate: 0 }} transition={{ duration: 0.6, ease: 'linear', repeat: refreshing ? Infinity : 0 }}>
              <RefreshCw size={12} />
            </motion.div>
            {refreshing ? 'Refreshing…' : 'Refresh'}
          </motion.button>
        </div>
      </div>

      {/* Cloud filter pills */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 14 }}>
        <Filter size={11} color="var(--muted)" />
        {clouds.map(cloud => (
          <motion.button
            key={cloud}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setCloudFilter(cloud)}
            style={{
              padding: '3px 10px', borderRadius: 20, fontSize: 11, fontWeight: 600,
              border: `1px solid ${cloudFilter === cloud ? (CLOUD_COLORS[cloud] || 'var(--accent)') : 'var(--border)'}`,
              background: cloudFilter === cloud ? `${CLOUD_COLORS[cloud] || 'var(--accent)'}18` : 'transparent',
              color: cloudFilter === cloud ? (CLOUD_COLORS[cloud] || 'var(--accent)') : 'var(--muted)',
              textTransform: 'uppercase', cursor: 'pointer',
            }}
          >
            {cloud}
          </motion.button>
        ))}
      </div>

      {/* Table */}
      <div style={{ overflowX: 'auto', overflowY: 'auto', maxHeight: 400 }}>
        {filtered.length === 0 && !loading ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '48px 0', color: 'var(--muted)', gap: 10 }}>
            <motion.div
              animate={{ y: [0, -6, 0] }}
              transition={{ duration: 2.5, repeat: Infinity }}
              style={{ width: 48, height: 48, borderRadius: '50%', background: 'var(--surface2)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 20, border: '1px solid var(--border)' }}
            >
              📋
            </motion.div>
            <div style={{ fontWeight: 600, color: 'var(--text2)' }}>No decisions yet</div>
            <div style={{ fontSize: 12 }}>Submit a workload to see decisions appear here</div>
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {HEADERS.map(h => <th key={h} style={headerCellStyle}>{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {loading
                ? <SkeletonTableRows count={5} />
                : filtered.map((decision, i) => (
                    <DecisionRow
                      key={decision?.decision_id || i}
                      decision={decision}
                      index={i}
                      isNew={newIds.has(decision?.decision_id)}
                    />
                  ))
              }
            </tbody>
          </table>
        )}
      </div>
    </motion.div>
  )
}