import { useEffect, useState, useCallback, useRef } from 'react'
import { Cpu, TrendingDown, Leaf, Clock, Wifi, WifiOff } from 'lucide-react'
import { getStatus, getDecisions } from '../api/client'
import { SkeletonMetricsBar } from './Skeleton'

function Metric({ icon: Icon, label, value, sub, color, highlight = false }) {
  return (
    <div
      className="card card-hover"
      style={{
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        transition: 'all 0.2s',
        ...(highlight ? { borderColor: color, boxShadow: `0 0 0 1px ${color}22` } : {}),
      }}
    >
      <div
        style={{
          width: 44,
          height: 44,
          borderRadius: 12,
          background: `${color}18`,
          border: `1px solid ${color}30`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
        }}
      >
        <Icon size={19} color={color} />
      </div>

      <div style={{ minWidth: 0 }}>
        <div
          style={{
            color: 'var(--muted)',
            fontSize: 11,
            marginBottom: 3,
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            fontWeight: 600,
          }}
        >
          {label}
        </div>

        <div
          style={{
            fontSize: 24,
            fontWeight: 800,
            lineHeight: 1,
            letterSpacing: '-0.02em',
          }}
        >
          {value}
        </div>

        {sub ? (
          <div style={{ color: 'var(--muted)', fontSize: 11, marginTop: 4 }}>
            {sub}
          </div>
        ) : null}
      </div>
    </div>
  )
}

export default function MetricsBar() {
  const [status, setStatus] = useState(null)
  const [decisions, setDecisions] = useState([])
  const [loading, setLoading] = useState(true)
  const [online, setOnline] = useState(true)
  const mountedRef = useRef(true)

  const refresh = useCallback(async () => {
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
    } catch {
      // Intentionally silent to avoid UI crash on polling failures
    } finally {
      if (mountedRef.current) {
        setLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    refresh()

    const intervalId = setInterval(refresh, 10000)

    return () => {
      mountedRef.current = false
      clearInterval(intervalId)
    }
  }, [refresh])

  if (loading) {
    return <SkeletonMetricsBar />
  }

  const count = decisions.length

  const avgLatency =
    count > 0
      ? Math.round(
          decisions.reduce((sum, decision) => sum + (Number(decision?.latency_ms) || 0), 0) / count
        )
      : null

  const avgCost =
    count > 0
      ? (
          decisions.reduce(
            (sum, decision) => sum + (Number(decision?.cost_savings_pct) || 0),
            0
          ) / count
        ).toFixed(1)
      : null

  const avgCarbon =
    count > 0
      ? (
          decisions.reduce(
            (sum, decision) => sum + (Number(decision?.carbon_savings_pct) || 0),
            0
          ) / count
        ).toFixed(1)
      : null

  return (
    <div style={{ marginBottom: 24 }}>
      <div style={{ display: 'flex', gap: 16 }}>
        <Metric
          icon={Cpu}
          label="Decisions Served"
          value={status?.decisions_served ?? 0}
          sub={status?.agent_loaded ? 'PPO model active' : 'Agent loading…'}
          color="var(--accent)"
          highlight={Boolean(status?.agent_loaded)}
        />

        <Metric
          icon={Clock}
          label="Avg Latency"
          value={avgLatency != null ? `${avgLatency}ms` : '—'}
          sub={avgLatency != null ? (avgLatency < 200 ? '✓ Within target' : 'Inference observed') : 'No data yet'}
          color="var(--accent2)"
        />

        <Metric
          icon={TrendingDown}
          label="Avg Cost Savings"
          value={avgCost != null ? `${avgCost}%` : '—'}
          sub="vs on-demand baseline"
          color="var(--green)"
          highlight={avgCost != null && Number(avgCost) > 20}
        />

        <Metric
          icon={Leaf}
          label="Avg Carbon Savings"
          value={avgCarbon != null ? `${avgCarbon}%` : '—'}
          sub="vs us-east-1 baseline"
          color="var(--green2)"
        />
      </div>

      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginTop: 10,
          padding: '6px 12px',
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: 8,
          fontSize: 11,
          color: 'var(--muted)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {online ? (
            <>
              <Wifi size={11} color="var(--green)" />
              <span style={{ color: 'var(--green)' }}>API connected</span>
            </>
          ) : (
            <>
              <WifiOff size={11} color="var(--red)" />
              <span style={{ color: 'var(--red)' }}>API unreachable</span>
            </>
          )}

          {status?.shap_ready ? (
            <span style={{ marginLeft: 8, color: 'var(--accent2)' }}>
              · SHAP ready
            </span>
          ) : null}
        </div>

        <span>Auto-refresh every 10s</span>
      </div>
    </div>
  )
}