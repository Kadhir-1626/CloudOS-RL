import { useState, useCallback, useEffect, useRef } from 'react'
import {
  MapPin,
  Server,
  Tag,
  TrendingDown,
  Leaf,
  Zap,
  ChevronDown,
  ChevronUp,
  Loader,
  AlertCircle,
} from 'lucide-react'
import { explainDecision, getDecision } from '../api/client'
import { toast } from './Toast'

const CLOUD_COLORS = {
  aws: '#f59e0b',
  gcp: '#3b82f6',
  azure: '#6366f1',
  hybrid: '#10b981',
}

const CLOUD_LABELS = {
  aws: 'Amazon Web Services',
  gcp: 'Google Cloud',
  azure: 'Microsoft Azure',
  hybrid: 'Hybrid Cloud',
}

const PURCHASE_BADGE = {
  spot: 'badge-green',
  on_demand: 'badge-blue',
  reserved_1yr: 'badge-purple',
  reserved_3yr: 'badge-purple',
}

function Row({ icon: Icon, label, value, valueColor, mono = false }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '9px 0',
        borderBottom: '1px solid var(--border)',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          color: 'var(--muted)',
        }}
      >
        <Icon size={13} />
        <span style={{ fontSize: 12 }}>{label}</span>
      </div>

      <span
        style={{
          fontWeight: 600,
          color: valueColor || 'var(--text)',
          fontSize: 13,
          fontFamily: mono ? 'monospace' : 'inherit',
        }}
      >
        {value}
      </span>
    </div>
  )
}

function SavingsBox({ label, value, color, icon: Icon }) {
  return (
    <div
      style={{
        flex: 1,
        background: `${color}10`,
        border: `1px solid ${color}25`,
        borderRadius: 10,
        padding: '12px 14px',
        textAlign: 'center',
      }}
    >
      <div
        style={{
          color: 'var(--muted)',
          fontSize: 11,
          marginBottom: 4,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 4,
        }}
      >
        <Icon size={11} />
        {label}
      </div>

      <div
        style={{
          fontSize: 26,
          fontWeight: 900,
          color,
          letterSpacing: '-0.02em',
          lineHeight: 1,
        }}
      >
        {value}
      </div>
    </div>
  )
}

function DriverBar({ label, value, direction }) {
  const numericValue = Number(value) || 0
  const abs = Math.abs(numericValue)
  const width = Math.min((abs / 0.5) * 100, 100)
  const color = direction === 'positive' ? 'var(--green)' : 'var(--red)'

  return (
    <div style={{ marginBottom: 10 }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 11,
          marginBottom: 4,
        }}
      >
        <span
          style={{
            color: 'var(--text2)',
            maxWidth: 220,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {label}
        </span>

        <span
          style={{
            color,
            fontWeight: 700,
            fontFamily: 'monospace',
          }}
        >
          {numericValue > 0 ? '+' : ''}
          {numericValue.toFixed(4)}
        </span>
      </div>

      <div
        style={{
          height: 5,
          background: 'var(--surface2)',
          borderRadius: 3,
        }}
      >
        <div
          style={{
            height: '100%',
            width: `${width}%`,
            background: color,
            borderRadius: 3,
            transition: 'width 0.5s ease',
          }}
        />
      </div>
    </div>
  )
}

export default function DecisionCard({ decision: initial }) {
  const [decision, setDecision] = useState(initial)
  const [showExplain, setShowExplain] = useState(false)
  const [explainState, setExplainState] = useState('idle') // idle | requesting | polling | done | error
  const [explainMsg, setExplainMsg] = useState('')

  const pollRef = useRef(null)

  const safeToast = {
    info: (msg) => toast?.info?.(msg),
    success: (msg) => toast?.success?.(msg),
    error: (msg) => toast?.error?.(msg),
  }

  useEffect(() => {
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [])

  const cloudColor = CLOUD_COLORS[decision?.cloud] || 'var(--accent)'
  const cloudLabel = CLOUD_LABELS[decision?.cloud] || decision?.cloud || 'Unknown Cloud'

  const hasExplanation =
    decision?.explanation &&
    typeof decision.explanation === 'object' &&
    !!decision.explanation.summary

  const handleExplain = useCallback(async () => {
    if (!decision?.decision_id) return
    if (explainState === 'polling' || explainState === 'requesting') return

    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }

    setExplainState('requesting')
    setExplainMsg('')

    try {
      await explainDecision(decision.decision_id)
      setExplainState('polling')
      setExplainMsg('SHAP computing… this takes ~10 seconds')
      safeToast.info('SHAP explanation requested — computing in background')

      let attempts = 0

      pollRef.current = setInterval(async () => {
        attempts += 1

        try {
          const updated = await getDecision(decision.decision_id)

          if (updated?.explanation?.summary) {
            setDecision(updated)
            setShowExplain(true)
            setExplainState('done')
            setExplainMsg('')

            if (pollRef.current) {
              clearInterval(pollRef.current)
              pollRef.current = null
            }

            safeToast.success('SHAP explanation ready')
            return
          }
        } catch {
          // ignore transient polling failures
        }

        if (attempts >= 25) {
          if (pollRef.current) {
            clearInterval(pollRef.current)
            pollRef.current = null
          }

          setExplainState('error')
          setExplainMsg('Timed out waiting for explanation. SHAP may still be computing.')
          safeToast.error('SHAP explanation timed out')
        }
      }, 2500)
    } catch (e) {
      const detail =
        e?.response?.data?.detail ||
        e?.response?.data?.message ||
        e?.message ||
        'Failed to request SHAP explanation'

      setExplainState('error')
      setExplainMsg(detail)
      safeToast.error(`SHAP request failed: ${String(detail).slice(0, 60)}`)
    }
  }, [decision?.decision_id, explainState])

  const isComputing =
    explainState === 'requesting' || explainState === 'polling'

  const latencyMs =
    decision?.latency_ms != null ? Number(decision.latency_ms) : null
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

  return (
    <div
      className="card fade-in"
      style={{ borderLeft: `3px solid ${cloudColor}` }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          marginBottom: 16,
        }}
      >
        <div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              marginBottom: 5,
            }}
          >
            <span style={{ fontWeight: 700, fontSize: 15 }}>
              Placement Decision
            </span>

            <span
              className={`badge ${
                PURCHASE_BADGE[decision?.purchase_option] || 'badge-blue'
              }`}
            >
              {(decision?.purchase_option || 'on_demand').replace(/_/g, ' ')}
            </span>
          </div>

          <div
            style={{
              color: 'var(--muted)',
              fontSize: 11,
              fontFamily: 'monospace',
            }}
          >
            {decision?.decision_id
              ? `${decision.decision_id.slice(0, 20)}…`
              : 'No decision ID'}
          </div>
        </div>

        <div style={{ textAlign: 'right' }}>
          <div
            style={{
              fontSize: 10,
              color: 'var(--muted)',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              marginBottom: 2,
            }}
          >
            Inference
          </div>

          <div
            style={{
              fontSize: 20,
              fontWeight: 800,
              lineHeight: 1,
              color:
                latencyMs == null
                  ? 'var(--muted)'
                  : latencyMs < 200
                  ? 'var(--green)'
                  : 'var(--yellow)',
            }}
          >
            {latencyMs != null ? `${latencyMs.toFixed(0)}ms` : '—'}
          </div>
        </div>
      </div>

      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '8px 12px',
          marginBottom: 4,
          background: `${cloudColor}0d`,
          border: `1px solid ${cloudColor}25`,
          borderRadius: 8,
        }}
      >
        <div
          style={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: cloudColor,
          }}
        />
        <span
          style={{
            fontWeight: 700,
            color: cloudColor,
            fontSize: 13,
          }}
        >
          {cloudLabel}
        </span>
      </div>

      <Row icon={MapPin} label="Region" value={decision?.region || '—'} />
      <Row
        icon={Server}
        label="Instance Type"
        value={decision?.instance_type || '—'}
        mono
      />
      <Row
        icon={Tag}
        label="Est. Cost/hr"
        value={estimatedCost != null ? `$${estimatedCost.toFixed(4)}` : '—'}
        mono
      />

      <div
        style={{
          display: 'flex',
          gap: 12,
          marginTop: 14,
          marginBottom: 14,
        }}
      >
        <SavingsBox
          label="Cost Savings"
          value={costSavings != null ? `${costSavings.toFixed(1)}%` : '—'}
          color="var(--green)"
          icon={TrendingDown}
        />

        <SavingsBox
          label="Carbon Savings"
          value={carbonSavings != null ? `${carbonSavings.toFixed(1)}%` : '—'}
          color="var(--green2)"
          icon={Leaf}
        />
      </div>

      {!hasExplanation && (
        <div>
          <button
            type="button"
            onClick={handleExplain}
            disabled={isComputing}
            style={{
              width: '100%',
              padding: '9px 0',
              background: isComputing
                ? 'rgba(99,102,241,0.08)'
                : 'var(--surface2)',
              border: `1px solid ${
                isComputing ? 'rgba(99,102,241,0.3)' : 'var(--border)'
              }`,
              color: isComputing ? 'var(--accent2)' : 'var(--text2)',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 7,
            }}
          >
            {isComputing ? (
              <>
                <Loader
                  size={13}
                  style={{ animation: 'spin 0.8s linear infinite' }}
                />
                {explainMsg || 'Requesting…'}
              </>
            ) : (
              <>
                <Zap size={13} />
                Explain with SHAP
              </>
            )}
          </button>

          {explainState === 'error' && explainMsg && (
            <div
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: 6,
                marginTop: 8,
                padding: '8px 10px',
                background: 'rgba(239,68,68,0.08)',
                border: '1px solid rgba(239,68,68,0.2)',
                borderRadius: 7,
                fontSize: 11,
                color: '#fca5a5',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}
            >
              <AlertCircle size={12} style={{ flexShrink: 0, marginTop: 1 }} />
              {explainMsg}
            </div>
          )}
        </div>
      )}

      {hasExplanation && (
        <div className="fade-in">
          <button
            type="button"
            onClick={() => setShowExplain((v) => !v)}
            style={{
              width: '100%',
              padding: '9px 14px',
              background: 'rgba(99,102,241,0.08)',
              border: '1px solid rgba(99,102,241,0.25)',
              color: 'var(--accent2)',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <span
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 7,
              }}
            >
              <Zap size={13} />
              SHAP Explanation

              {decision?.explanation?.confidence != null && (
                <span
                  style={{
                    fontWeight: 400,
                    fontSize: 11,
                    color: 'var(--muted)',
                  }}
                >
                  · {(Number(decision.explanation.confidence) * 100).toFixed(0)}%
                  {' '}confidence
                </span>
              )}
            </span>

            {showExplain ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
          </button>

          {showExplain && (
            <div
              className="fade-in"
              style={{
                background: 'var(--surface2)',
                border: '1px solid var(--border)',
                borderTop: 'none',
                borderRadius: '0 0 8px 8px',
                padding: '16px',
              }}
            >
              {decision?.explanation?.summary && (
                <p
                  style={{
                    fontSize: 12,
                    color: 'var(--text2)',
                    lineHeight: 1.8,
                    marginBottom: 16,
                    padding: '10px 12px',
                    background: 'var(--surface)',
                    border: '1px solid var(--border)',
                    borderRadius: 7,
                  }}
                >
                  {decision.explanation.summary}
                </p>
              )}

              {decision?.explanation?.top_drivers?.length > 0 && (
                <>
                  <div
                    style={{
                      fontSize: 10,
                      color: 'var(--muted)',
                      marginBottom: 12,
                      textTransform: 'uppercase',
                      letterSpacing: '0.06em',
                      fontWeight: 700,
                    }}
                  >
                    Top Feature Drivers
                  </div>

                  {decision.explanation.top_drivers.map((driver, index) => (
                    <DriverBar
                      key={index}
                      label={driver.label || driver.feature || 'Unknown feature'}
                      value={driver.shap_value}
                      direction={driver.direction}
                    />
                  ))}
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}