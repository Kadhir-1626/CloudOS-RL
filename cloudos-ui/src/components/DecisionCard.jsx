import { useState } from 'react'
import { Cloud, MapPin, Server, Tag, TrendingDown, Leaf, Zap, ChevronDown, ChevronUp } from 'lucide-react'
import { explainDecision, getDecision } from '../api/client'

const CLOUD_COLORS = { aws: '#f59e0b', gcp: '#3b82f6', azure: '#6366f1', hybrid: '#10b981' }

function Row({ icon: Icon, label, value, color }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--muted)' }}>
        <Icon size={13} />
        <span style={{ fontSize: 12 }}>{label}</span>
      </div>
      <span style={{ fontWeight: 600, color: color || 'var(--text)', fontSize: 13 }}>{value}</span>
    </div>
  )
}

function DriverBar({ label, value, direction }) {
  const abs   = Math.abs(value)
  const max   = 0.5
  const width = Math.min((abs / max) * 100, 100)
  const color = direction === 'positive' ? 'var(--green)' : 'var(--red)'
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 3 }}>
        <span style={{ color: 'var(--muted)', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{label}</span>
        <span style={{ color, fontWeight: 600 }}>{value > 0 ? '+' : ''}{value.toFixed(4)}</span>
      </div>
      <div style={{ height: 4, background: 'var(--border)', borderRadius: 2 }}>
        <div style={{ height: '100%', width: `${width}%`, background: color, borderRadius: 2, transition: 'width 0.4s ease' }} />
      </div>
    </div>
  )
}

export default function DecisionCard({ decision: initial }) {
  const [decision,     setDecision]     = useState(initial)
  const [showExplain,  setShowExplain]  = useState(false)
  const [explaining,   setExplaining]   = useState(false)
  const [explainError, setExplainError] = useState(null)
  const [polling,      setPolling]      = useState(false)

  const cloudColor = CLOUD_COLORS[decision.cloud] || 'var(--accent)'

  const purchaseBadge = {
    spot:         'badge-green',
    on_demand:    'badge-blue',
    reserved_1yr: 'badge-purple',
    reserved_3yr: 'badge-purple',
  }[decision.purchase_option] || 'badge-blue'

  const handleExplain = async () => {
    setExplaining(true); setExplainError(null)
    try {
      await explainDecision(decision.decision_id)
      // Poll for explanation attachment (max 30 attempts × 2s = 60s)
      setPolling(true)
      let attempts = 0
      const timer = setInterval(async () => {
        attempts++
        try {
          const updated = await getDecision(decision.decision_id)
          if (updated.explanation && Object.keys(updated.explanation).length > 0 &&
              updated.explanation.summary) {
            setDecision(updated)
            setShowExplain(true)
            clearInterval(timer)
            setPolling(false)
          }
        } catch {}
        if (attempts >= 30) {
          clearInterval(timer)
          setPolling(false)
          setExplainError('Explanation timed out. SHAP may still be computing.')
        }
      }, 2000)
    } catch (e) {
      setExplainError(e?.response?.data?.detail || e.message)
    } finally {
      setExplaining(false)
    }
  }

  const drivers = decision.explanation?.top_drivers || []
  const summary = decision.explanation?.summary

  return (
    <div className="card" style={{ borderLeft: `3px solid ${cloudColor}` }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 16 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span style={{ fontWeight: 700, fontSize: 15 }}>Scheduling Decision</span>
            <span className={`badge ${purchaseBadge}`}>{decision.purchase_option.replace('_', ' ')}</span>
          </div>
          <span style={{ color: 'var(--muted)', fontSize: 11, fontFamily: 'monospace' }}>
            {decision.decision_id.slice(0, 16)}…
          </span>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 11, color: 'var(--muted)' }}>Inference</div>
          <div style={{ fontWeight: 700, color: 'var(--green)', fontSize: 16 }}>
            {decision.latency_ms?.toFixed(0)}ms
          </div>
        </div>
      </div>

      {/* Placement */}
      <Row icon={Cloud}  label="Cloud Provider" value={decision.cloud.toUpperCase()}    color={cloudColor} />
      <Row icon={MapPin} label="Region"          value={decision.region} />
      <Row icon={Server} label="Instance Type"   value={decision.instance_type} />
      <Row icon={Tag}    label="Est. Cost/hr"    value={`$${decision.estimated_cost_per_hr?.toFixed(4)}`} />

      {/* Savings */}
      <div style={{ display: 'flex', gap: 12, marginTop: 14, marginBottom: 14 }}>
        <div style={{
          flex: 1, background: '#10b98112', border: '1px solid #10b98130',
          borderRadius: 8, padding: '10px 14px', textAlign: 'center',
        }}>
          <div style={{ color: 'var(--muted)', fontSize: 11, marginBottom: 2 }}>
            <TrendingDown size={11} style={{ display: 'inline', marginRight: 4 }} />
            Cost Savings
          </div>
          <div style={{ fontSize: 22, fontWeight: 800, color: 'var(--green)' }}>
            {decision.cost_savings_pct?.toFixed(1)}%
          </div>
        </div>
        <div style={{
          flex: 1, background: '#34d39912', border: '1px solid #34d39930',
          borderRadius: 8, padding: '10px 14px', textAlign: 'center',
        }}>
          <div style={{ color: 'var(--muted)', fontSize: 11, marginBottom: 2 }}>
            <Leaf size={11} style={{ display: 'inline', marginRight: 4 }} />
            Carbon Savings
          </div>
          <div style={{ fontSize: 22, fontWeight: 800, color: '#34d399' }}>
            {decision.carbon_savings_pct?.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Explain button */}
      {!summary && (
        <button
          onClick={handleExplain}
          disabled={explaining || polling}
          style={{
            width: '100%', padding: '9px 0',
            background: 'var(--surface2)', border: '1px solid var(--border)',
            color: 'var(--text)', fontWeight: 600,
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
          }}
        >
          <Zap size={13} />
          {explaining ? 'Requesting SHAP…' : polling ? 'Computing explanation…' : 'Explain with SHAP'}
        </button>
      )}

      {explainError && (
        <div style={{ color: 'var(--red)', fontSize: 12, marginTop: 8 }}>{explainError}</div>
      )}

      {/* Explanation */}
      {summary && (
        <div style={{ marginTop: 14 }}>
          <button
            onClick={() => setShowExplain(v => !v)}
            style={{
              width: '100%', padding: '8px 12px',
              background: '#6366f112', border: '1px solid #6366f130',
              color: 'var(--accent2)', fontWeight: 600,
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            }}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Zap size={12} /> SHAP Explanation
              {decision.explanation?.confidence != null && (
                <span style={{ color: 'var(--muted)', fontWeight: 400, fontSize: 11 }}>
                  confidence {(decision.explanation.confidence * 100).toFixed(0)}%
                </span>
              )}
            </span>
            {showExplain ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
          </button>

          {showExplain && (
            <div style={{
              background: 'var(--surface2)', border: '1px solid var(--border)',
              borderTop: 'none', borderRadius: '0 0 8px 8px', padding: '14px 16px',
            }}>
              {summary && (
                <p style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 14, lineHeight: 1.7 }}>
                  {summary}
                </p>
              )}
              {drivers.length > 0 && (
                <>
                  <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Top Feature Drivers
                  </div>
                  {drivers.map((d, i) => (
                    <DriverBar
                      key={i}
                      label={d.label || d.feature}
                      value={d.shap_value}
                      direction={d.direction}
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