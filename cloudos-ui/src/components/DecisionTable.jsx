import { useEffect, useState, useCallback, useRef, memo } from 'react'
import { RefreshCw, ArrowDown } from 'lucide-react'
import { getDecisions } from '../api/client'
import { SkeletonTableRows } from './Skeleton'

const CLOUD_COLORS = {
  aws: '#f59e0b',
  gcp: '#3b82f6',
  azure: '#6366f1',
  hybrid: '#10b981',
}

const HEADERS = [
  'ID',
  'Cloud',
  'Region',
  'Instance',
  'Purchase',
  'Cost/hr',
  'Cost Δ',
  'CO₂ Δ',
  'Latency',
]

const cellStyle = {
  padding: '10px 12px',
  fontSize: 12,
  borderBottom: '1px solid var(--border)',
  verticalAlign: 'middle',
}

const DecisionRow = memo(function DecisionRow({ decision }) {
  const estimatedCost =
    decision?.estimated_cost_per_hr != null
      ? Number(decision.estimated_cost_per_hr)
      : null

  const costSavings =
    decision?.cost_savings_pct != null
      ? Number(decision.cost_savings_pct)
      : null

  const carbonSavings =
    decision?.carbon_savings_pct != null
      ? Number(decision.carbon_savings_pct)
      : null

  const latencyMs =
    decision?.latency_ms != null
      ? Number(decision.latency_ms)
      : null

  const purchaseOption = decision?.purchase_option || 'on_demand'

  return (
    <tr
      style={{ transition: 'background 0.12s' }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = 'var(--surface2)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'transparent'
      }}
    >
      <td
        style={{
          ...cellStyle,
          fontFamily: 'monospace',
          color: 'var(--muted)',
          fontSize: 11,
        }}
      >
        {decision?.decision_id ? decision.decision_id.slice(0, 8) : '—'}
      </td>

      <td style={cellStyle}>
        <span
          style={{
            fontWeight: 700,
            color: CLOUD_COLORS[decision?.cloud] || 'var(--text)',
            textTransform: 'uppercase',
            fontSize: 11,
          }}
        >
          {decision?.cloud || '—'}
        </span>
      </td>

      <td style={{ ...cellStyle, color: 'var(--text2)' }}>
        {decision?.region || '—'}
      </td>

      <td style={{ ...cellStyle, fontFamily: 'monospace', fontSize: 11 }}>
        {decision?.instance_type || '—'}
      </td>

      <td style={cellStyle}>
        <span
          className={`badge ${
            purchaseOption === 'spot' ? 'badge-green' : 'badge-blue'
          }`}
        >
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

      <td
        style={{
          ...cellStyle,
          fontWeight: 700,
          color:
            latencyMs == null
              ? 'var(--muted)'
              : latencyMs < 200
              ? 'var(--green)'
              : latencyMs < 500
              ? 'var(--yellow)'
              : 'var(--red)',
        }}
      >
        {latencyMs != null ? `${latencyMs.toFixed(0)}ms` : '—'}
      </td>
    </tr>
  )
})

export default function DecisionTable() {
  const [decisions, setDecisions] = useState([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [lastTs, setLastTs] = useState(null)
  const [newCount, setNewCount] = useState(0)

  const prevCountRef = useRef(0)
  const mountedRef = useRef(true)
  const newBadgeTimeoutRef = useRef(null)

  const load = useCallback(async (manual = false) => {
    if (manual) {
      setRefreshing(true)
    }

    try {
      const response = await getDecisions(50)

      if (!mountedRef.current) return

      const list = response?.decisions || []
      setDecisions(list)
      setLastTs(new Date())

      if (!manual && list.length > prevCountRef.current && prevCountRef.current > 0) {
        const diff = list.length - prevCountRef.current
        setNewCount(diff)

        if (newBadgeTimeoutRef.current) {
          clearTimeout(newBadgeTimeoutRef.current)
        }

        newBadgeTimeoutRef.current = setTimeout(() => {
          if (mountedRef.current) {
            setNewCount(0)
          }
        }, 3000)
      }

      prevCountRef.current = list.length
    } catch {
      // Silent fail to avoid breaking dashboard polling UX
    } finally {
      if (mountedRef.current) {
        setLoading(false)
        setRefreshing(false)
      }
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    load(false)

    const intervalId = setInterval(() => {
      load(false)
    }, 12000)

    return () => {
      mountedRef.current = false
      clearInterval(intervalId)

      if (newBadgeTimeoutRef.current) {
        clearTimeout(newBadgeTimeoutRef.current)
      }
    }
  }, [load])

  const headerCellStyle = {
    padding: '9px 12px',
    textAlign: 'left',
    fontSize: 10,
    color: 'var(--muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.07em',
    borderBottom: '1px solid var(--border)',
    fontWeight: 700,
    background: 'var(--surface)',
    position: 'sticky',
    top: 0,
    zIndex: 1,
  }

  return (
    <div className="card">
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 16,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontWeight: 700, fontSize: 15 }}>
            Decision History
          </span>

          <span
            style={{
              background: 'var(--surface2)',
              border: '1px solid var(--border)',
              borderRadius: 20,
              padding: '2px 8px',
              fontSize: 11,
              color: 'var(--muted)',
              fontWeight: 600,
            }}
          >
            {decisions.length}
          </span>

          {newCount > 0 && (
            <span className="badge badge-green slide-in">
              <ArrowDown size={9} /> +{newCount} new
            </span>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {lastTs && (
            <span style={{ color: 'var(--muted)', fontSize: 11 }}>
              {lastTs.toLocaleTimeString()}
            </span>
          )}

          <button
            type="button"
            onClick={() => load(true)}
            disabled={refreshing}
            style={{
              padding: '6px 12px',
              background: 'var(--surface2)',
              border: '1px solid var(--border)',
              color: 'var(--text2)',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              fontWeight: 600,
            }}
          >
            <RefreshCw
              size={12}
              style={
                refreshing
                  ? { animation: 'spin 0.8s linear infinite' }
                  : {}
              }
            />
            {refreshing ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      <div style={{ overflowX: 'auto', overflowY: 'auto', maxHeight: 420 }}>
        {decisions.length === 0 && !loading ? (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '48px 0',
              color: 'var(--muted)',
              gap: 10,
            }}
          >
            <div
              style={{
                width: 48,
                height: 48,
                borderRadius: '50%',
                background: 'var(--surface2)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 20,
                border: '1px solid var(--border)',
              }}
            >
              📋
            </div>

            <div style={{ fontWeight: 600, color: 'var(--text2)' }}>
              No decisions yet
            </div>

            <div style={{ fontSize: 12 }}>
              Submit a workload to see decisions appear here
            </div>
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {HEADERS.map((header) => (
                  <th key={header} style={headerCellStyle}>
                    {header}
                  </th>
                ))}
              </tr>
            </thead>

            <tbody>
              {loading
                ? <SkeletonTableRows count={5} />
                : decisions.map((decision) => (
                    <DecisionRow
                      key={decision?.decision_id || Math.random()}
                      decision={decision}
                    />
                  ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}