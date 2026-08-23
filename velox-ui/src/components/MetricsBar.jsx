import { useEffect, useState, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import CountUp from 'react-countup'
import { Cpu, TrendingDown, Leaf, Clock, Wifi, WifiOff, RefreshCw } from 'lucide-react'
import { getStatus, getDecisions } from '../api/client'
import { SkeletonMetricsBar } from './Skeleton'

function Sparkline({ data, color }) {
  if (!data || data.length < 2) return null
  const w = 64, h = 28, pad = 2
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const pts = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * (w - pad * 2)
    const y = h - pad - ((v - min) / range) * (h - pad * 2)
    return `${x},${y}`
  })
  const polyline = pts.join(' ')
  const firstPt = pts[0].split(',')
  const lastPt = pts[pts.length - 1].split(',')
  const fillPath = `M${firstPt[0]},${h} L${polyline.replace(' ', ' L').split(' ').join(' L')} L${lastPt[0]},${h} Z`

  return (
    <svg width={w} height={h} style={{ overflow: 'visible', opacity: 0.8 }}>
      <defs>
        <linearGradient id={`sg-${color}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={fillPath} fill={`url(#sg-${color})`} />
      <polyline points={polyline} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={lastPt[0]} cy={lastPt[1]} r="2.5" fill={color} />
    </svg>
  )
}

function Metric({ icon: Icon, label, value, rawValue, sub, color, highlight, sparkData, index }) {
  const prevValue = useRef(0)

  useEffect(() => {
    if (rawValue != null) prevValue.current = rawValue
  }, [rawValue])

  return (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.08, duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ y: -2, boxShadow: `0 8px 32px rgba(0,0,0,0.3), 0 0 0 1px ${color}30` }}
      style={{
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        gap: 14,
        cursor: 'default',
        borderColor: highlight ? `${color}40` : undefined,
        boxShadow: highlight ? `0 0 0 1px ${color}22` : undefined,
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Background glow for highlighted cards */}
      {highlight && (
        <motion.div
          animate={{ opacity: [0.03, 0.07, 0.03] }}
          transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
          style={{
            position: 'absolute', inset: 0,
            background: `radial-gradient(ellipse at 20% 50%, ${color}20, transparent 70%)`,
            pointerEvents: 'none',
          }}
        />
      )}

      <div style={{
        width: 44, height: 44, borderRadius: 12,
        background: `${color}18`, border: `1px solid ${color}30`,
        display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
      }}>
        <Icon size={19} color={color} />
      </div>

      <div style={{ minWidth: 0, flex: 1 }}>
        <div style={{
          color: 'var(--muted)', fontSize: 11, marginBottom: 3,
          textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600,
        }}>
          {label}
        </div>

        <div style={{ fontSize: 24, fontWeight: 800, lineHeight: 1, letterSpacing: '-0.02em' }}>
          {rawValue != null ? (
            <CountUp
              start={prevValue.current}
              end={rawValue}
              duration={1.2}
              decimals={Number.isInteger(rawValue) ? 0 : 1}
              suffix={typeof value === 'string' && value.includes('ms') ? 'ms' : typeof value === 'string' && value.includes('%') ? '%' : ''}
              preserveValue
            />
          ) : (
            <span style={{ color: 'var(--muted)', fontSize: 20 }}>—</span>
          )}
        </div>

        {sub && (
          <div style={{ color: 'var(--muted)', fontSize: 11, marginTop: 4 }}>{sub}</div>
        )}
      </div>

      {sparkData && sparkData.length >= 2 && (
        <div style={{ flexShrink: 0 }}>
          <Sparkline data={sparkData} color={color} />
        </div>
      )}
    </motion.div>
  )
}

export default function MetricsBar() {
  const [status, setStatus] = useState(null)
  const [decisions, setDecisions] = useState([])
  const [loading, setLoading] = useState(true)
  const [online, setOnline] = useState(true)
  const [lastRefresh, setLastRefresh] = useState(null)
  const [refreshing, setRefreshing] = useState(false)
  const mountedRef = useRef(true)

  const refresh = useCallback(async (manual = false) => {
    if (manual) setRefreshing(true)
    try {
      const [statusResult, decisionsResult] = await Promise.allSettled([
        getStatus(),
        getDecisions(100),
      ])
      if (!mountedRef.current) return
      if (statusResult.status === 'fulfilled') {
        setStatus(statusResult.value || null)
        setOnline(true)
      } else {
        setOnline(false)
      }
      if (decisionsResult.status === 'fulfilled') {
        setDecisions(decisionsResult.value?.decisions || [])
      }
      setLastRefresh(new Date())
    } catch {
      // silent
    } finally {
      if (mountedRef.current) {
        setLoading(false)
        setRefreshing(false)
      }
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    refresh()
    const id = setInterval(refresh, 10000)
    return () => { mountedRef.current = false; clearInterval(id) }
  }, [refresh])

  if (loading) return <SkeletonMetricsBar />

  const count = decisions.length
  const recent = decisions.slice(-10)

  const avgLatency = count > 0
    ? Math.round(decisions.reduce((s, d) => s + (Number(d?.latency_ms) || 0), 0) / count)
    : null

  const avgCost = count > 0
    ? parseFloat((decisions.reduce((s, d) => s + (Number(d?.cost_savings_pct) || 0), 0) / count).toFixed(1))
    : null

  const avgCarbon = count > 0
    ? parseFloat((decisions.reduce((s, d) => s + (Number(d?.carbon_savings_pct) || 0), 0) / count).toFixed(1))
    : null

  const latencySpark = recent.map(d => Number(d?.latency_ms) || 0)
  const costSpark = recent.map(d => Number(d?.cost_savings_pct) || 0)
  const carbonSpark = recent.map(d => Number(d?.carbon_savings_pct) || 0)

  return (
    <div style={{ marginBottom: 24 }}>
      <div style={{ display: 'flex', gap: 14 }}>
        <Metric
          index={0}
          icon={Cpu}
          label="Placements Made"
          value={String(status?.decisions_served ?? 0)}
          rawValue={status?.decisions_served ?? 0}
          sub={status?.agent_loaded ? 'AI running' : 'Agent loading…'}
          color="var(--accent)"
          highlight={Boolean(status?.agent_loaded)}
          sparkData={decisions.slice(-10).map((_, i) => i + 1)}
        />
        <Metric
          index={1}
          icon={Clock}
          label="Decision Speed"
          value={avgLatency != null ? `${avgLatency}ms` : '—'}
          rawValue={avgLatency}
          sub={avgLatency != null ? (avgLatency < 200 ? '✓ Within target' : 'Inference observed') : 'No data yet'}
          color="var(--accent2)"
          sparkData={latencySpark}
        />
        <Metric
          index={2}
          icon={TrendingDown}
          label="Cost Reduction"
          value={avgCost != null ? `${avgCost}%` : '—'}
          rawValue={avgCost}
          sub="vs on-demand"
          color="var(--green)"
          highlight={avgCost != null && avgCost > 20}
          sparkData={costSpark}
        />
        <Metric
          index={3}
          icon={Leaf}
          label="Carbon Saved"
          value={avgCarbon != null ? `${avgCarbon}%` : '—'}
          rawValue={avgCarbon}
          sub="vs baseline"
          color="var(--green2)"
          sparkData={carbonSpark}
        />
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          marginTop: 10, padding: '6px 12px',
          background: 'var(--surface)', border: '1px solid var(--border)',
          borderRadius: 8, fontSize: 11, color: 'var(--muted)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <AnimatePresence mode="wait">
            {online ? (
              <motion.div key="online" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <motion.div
                  animate={{ opacity: [1, 0.3, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--green)' }}
                />
                <span style={{ color: 'var(--green)' }}>API connected</span>
              </motion.div>
            ) : (
              <motion.div key="offline" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <WifiOff size={11} color="var(--red)" />
                <span style={{ color: 'var(--red)' }}>API unreachable</span>
              </motion.div>
            )}
          </AnimatePresence>
          {status?.shap_ready && (
            <span style={{ color: 'var(--accent2)' }}>· SHAP ready</span>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {lastRefresh && (
            <span>{lastRefresh.toLocaleTimeString()}</span>
          )}
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => refresh(true)}
            style={{
              background: 'none', border: 'none', color: 'var(--muted)',
              padding: 2, display: 'flex', alignItems: 'center', borderRadius: 4,
            }}
            title="Refresh now"
          >
            <motion.div animate={refreshing ? { rotate: 360 } : {}} transition={{ duration: 0.6, ease: 'linear' }}>
              <RefreshCw size={11} />
            </motion.div>
          </motion.button>
        </div>
      </motion.div>
    </div>
  )
}