import { useState, useCallback } from 'react'
import { Send, Loader, Zap } from 'lucide-react'
import { scheduleWorkload } from '../api/client'
import { toast } from './Toast'

const DEFAULTS = {
  workload_type: 'training',
  cpu_request_vcpu: 4,
  memory_request_gb: 8,
  gpu_count: 0,
  storage_gb: 100,
  expected_duration_hours: 2,
  priority: 2,
  sla_latency_ms: 200,
  is_spot_tolerant: false,
}

const PRESETS = [
  {
    label: 'ML Training',
    values: {
      workload_type: 'training',
      cpu_request_vcpu: 8,
      memory_request_gb: 32,
      gpu_count: 1,
      is_spot_tolerant: true,
    },
  },
  {
    label: 'API Inference',
    values: {
      workload_type: 'inference',
      cpu_request_vcpu: 4,
      memory_request_gb: 8,
      gpu_count: 0,
      is_spot_tolerant: false,
    },
  },
  {
    label: 'ETL Batch',
    values: {
      workload_type: 'batch',
      cpu_request_vcpu: 2,
      memory_request_gb: 4,
      gpu_count: 0,
      is_spot_tolerant: true,
    },
  },
]

export default function ScheduleForm({ onResult, onLoading }) {
  const [form, setForm] = useState(DEFAULTS)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const set = useCallback((key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }))
  }, [])

  const safeToast = {
    info: (msg) => toast?.info?.(msg),
    success: (msg) => toast?.success?.(msg),
    error: (msg) => toast?.error?.(msg),
  }

  const applyPreset = (preset) => {
    setForm((prev) => ({ ...prev, ...preset.values }))
    safeToast.info(`Preset applied: ${preset.label}`)
  }

  const submit = async () => {
    setLoading(true)
    setError(null)
    onLoading?.(true)

    try {
      const payload = {
        ...form,
        cpu_request_vcpu: Number(form.cpu_request_vcpu),
        memory_request_gb: Number(form.memory_request_gb),
        gpu_count: Number(form.gpu_count),
        storage_gb: Number(form.storage_gb),
        expected_duration_hours: Number(form.expected_duration_hours),
        priority: Number(form.priority),
        sla_latency_ms: Number(form.sla_latency_ms),
      }

      const result = await scheduleWorkload(payload)

      console.log('[CloudOS] Schedule response:', result)

      if (!result?.decision_id) {
        throw new Error('Unexpected response format from API')
      }

      safeToast.success(
        `Decision made: ${(result.cloud || 'unknown').toUpperCase()} / ${result.region || 'unknown'}`
      )

      if (typeof onResult === 'function') {
        onResult(result)
      }
    } catch (e) {
      const detail =
        e?.response?.data?.detail ||
        e?.response?.data?.message ||
        e?.message ||
        'Unknown error'

      const code = e?.response?.status
      const message = code ? `[${code}] ${detail}` : detail

      setError(message)
      safeToast.error(`Scheduling failed: ${String(detail).slice(0, 80)}`)
      console.error('[CloudOS] Schedule error:', e?.response?.data || e)
    } finally {
      setLoading(false)
      onLoading?.(false)
    }
  }

  const row = { display: 'flex', gap: 12, marginBottom: 12 }
  const col = { flex: 1 }

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 18,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontWeight: 700, fontSize: 15 }}>Submit Workload</span>
          <span className="badge badge-blue">PPO Scheduler</span>
        </div>

        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {PRESETS.map((preset) => (
            <button
              key={preset.label}
              type="button"
              onClick={() => applyPreset(preset)}
              disabled={loading}
              style={{
                padding: '4px 10px',
                background: 'var(--surface2)',
                border: '1px solid var(--border)',
                color: 'var(--text2)',
                fontSize: 11,
                fontWeight: 600,
              }}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      <div style={row}>
        <div style={col}>
          <label>Workload Type</label>
          <select
            value={form.workload_type}
            onChange={(e) => set('workload_type', e.target.value)}
            disabled={loading}
          >
            <option value="training">Training</option>
            <option value="inference">Inference</option>
            <option value="batch">Batch</option>
            <option value="streaming">Streaming</option>
          </select>
        </div>

        <div style={col}>
          <label>Priority</label>
          <select
            value={form.priority}
            onChange={(e) => set('priority', e.target.value)}
            disabled={loading}
          >
            <option value={1}>1 — Low</option>
            <option value={2}>2 — Normal</option>
            <option value={3}>3 — High</option>
            <option value={4}>4 — Critical</option>
          </select>
        </div>
      </div>

      <div style={row}>
        <div style={col}>
          <label>CPU (vCPU)</label>
          <input
            type="number"
            min={0.25}
            step={0.25}
            value={form.cpu_request_vcpu}
            onChange={(e) => set('cpu_request_vcpu', e.target.value)}
            disabled={loading}
          />
        </div>

        <div style={col}>
          <label>Memory (GB)</label>
          <input
            type="number"
            min={0.5}
            step={0.5}
            value={form.memory_request_gb}
            onChange={(e) => set('memory_request_gb', e.target.value)}
            disabled={loading}
          />
        </div>

        <div style={col}>
          <label>GPU Count</label>
          <input
            type="number"
            min={0}
            max={16}
            value={form.gpu_count}
            onChange={(e) => set('gpu_count', e.target.value)}
            disabled={loading}
          />
        </div>
      </div>

      <div style={row}>
        <div style={col}>
          <label>Storage (GB)</label>
          <input
            type="number"
            min={1}
            value={form.storage_gb}
            onChange={(e) => set('storage_gb', e.target.value)}
            disabled={loading}
          />
        </div>

        <div style={col}>
          <label>Duration (hrs)</label>
          <input
            type="number"
            min={0.1}
            step={0.1}
            value={form.expected_duration_hours}
            onChange={(e) => set('expected_duration_hours', e.target.value)}
            disabled={loading}
          />
        </div>

        <div style={col}>
          <label>SLA Latency (ms)</label>
          <input
            type="number"
            min={10}
            value={form.sla_latency_ms}
            onChange={(e) => set('sla_latency_ms', e.target.value)}
            disabled={loading}
          />
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
        <button
          type="button"
          onClick={() => set('is_spot_tolerant', !form.is_spot_tolerant)}
          disabled={loading}
          style={{
            padding: '7px 14px',
            background: form.is_spot_tolerant
              ? 'rgba(16,185,129,0.12)'
              : 'var(--surface2)',
            border: `1px solid ${
              form.is_spot_tolerant ? 'rgba(16,185,129,0.4)' : 'var(--border)'
            }`,
            color: form.is_spot_tolerant ? 'var(--green)' : 'var(--muted)',
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}
        >
          <Zap size={12} />
          {form.is_spot_tolerant ? 'Spot Enabled' : 'Spot Disabled'}
        </button>

        <span style={{ color: 'var(--muted)', fontSize: 12 }}>
          {form.is_spot_tolerant
            ? 'Eligible for up to 70% cost reduction'
            : 'On-demand pricing — no interruption risk'}
        </span>
      </div>

      {error && (
        <div
          style={{
            background: 'rgba(239,68,68,0.1)',
            border: '1px solid rgba(239,68,68,0.3)',
            borderRadius: 8,
            padding: '10px 14px',
            color: '#fca5a5',
            fontSize: 12,
            marginBottom: 14,
            display: 'flex',
            alignItems: 'flex-start',
            gap: 8,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          <span style={{ flexShrink: 0, marginTop: 1 }}>⚠</span>
          <span>{error}</span>
        </div>
      )}

      <button
        type="button"
        onClick={submit}
        disabled={loading}
        style={{
          width: '100%',
          padding: '12px 0',
          background: loading
            ? 'var(--surface2)'
            : 'linear-gradient(135deg, var(--accent), var(--accent2))',
          color: loading ? 'var(--muted)' : '#fff',
          fontWeight: 700,
          fontSize: 14,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 8,
          border: loading ? '1px solid var(--border)' : 'none',
          letterSpacing: '0.02em',
        }}
      >
        {loading ? (
          <>
            <Loader size={15} style={{ animation: 'spin 0.8s linear infinite' }} />
            Computing placement…
          </>
        ) : (
          <>
            <Send size={15} />
            Schedule Workload
          </>
        )}
      </button>
    </div>
  )
}