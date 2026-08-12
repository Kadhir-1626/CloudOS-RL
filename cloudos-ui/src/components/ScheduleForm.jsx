import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Loader, Zap, ChevronRight, ChevronLeft, Cpu, Database, Clock, Shield } from 'lucide-react'
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
  { label: 'ML Training',   values: { workload_type: 'training',  cpu_request_vcpu: 8,  memory_request_gb: 32, gpu_count: 1, is_spot_tolerant: true  } },
  { label: 'API Inference', values: { workload_type: 'inference', cpu_request_vcpu: 4,  memory_request_gb: 8,  gpu_count: 0, is_spot_tolerant: false } },
  { label: 'ETL Batch',     values: { workload_type: 'batch',     cpu_request_vcpu: 2,  memory_request_gb: 4,  gpu_count: 0, is_spot_tolerant: true  } },
]

const STEPS = [
  { id: 'type',      label: 'Workload',   icon: Zap },
  { id: 'resources', label: 'Resources',  icon: Cpu },
  { id: 'constraints', label: 'Constraints', icon: Shield },
]

function SliderInput({ label, value, min, max, step, onChange, disabled, unit, color = 'var(--accent)' }) {
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
        <label style={{ margin: 0 }}>{label}</label>
        <span style={{ fontSize: 13, fontWeight: 700, color }}>{value}{unit}</span>
      </div>
      <div style={{ position: 'relative', height: 6, borderRadius: 3, background: 'var(--surface2)', border: '1px solid var(--border)' }}>
        <motion.div
          animate={{ width: `${pct}%` }}
          transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          style={{ height: '100%', borderRadius: 3, background: `linear-gradient(90deg, var(--accent), ${color})` }}
        />
      </div>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
        style={{
          width: '100%', marginTop: -6, opacity: 0, height: 20,
          cursor: disabled ? 'not-allowed' : 'pointer', position: 'relative', zIndex: 1,
        }}
      />
    </div>
  )
}

function StepIndicator({ current, total }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 0, marginBottom: 24 }}>
      {STEPS.map((step, i) => {
        const done = i < current
        const active = i === current
        return (
          <div key={step.id} style={{ display: 'flex', alignItems: 'center', flex: i < STEPS.length - 1 ? 1 : 0 }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
              <motion.div
                animate={{
                  background: done ? 'var(--green)' : active ? 'linear-gradient(135deg, var(--accent), var(--accent2))' : 'var(--surface2)',
                  borderColor: done ? 'var(--green)' : active ? 'var(--accent)' : 'var(--border)',
                  scale: active ? 1.1 : 1,
                }}
                transition={{ duration: 0.25 }}
                style={{
                  width: 32, height: 32, borderRadius: '50%',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  border: '2px solid',
                  fontSize: 12, fontWeight: 700,
                  color: done || active ? '#fff' : 'var(--muted)',
                }}
              >
                {done ? '✓' : i + 1}
              </motion.div>
              <span style={{ fontSize: 10, color: active ? 'var(--accent)' : 'var(--muted)', fontWeight: active ? 700 : 500, whiteSpace: 'nowrap' }}>
                {step.label}
              </span>
            </div>
            {i < STEPS.length - 1 && (
              <motion.div
                animate={{ background: done ? 'var(--green)' : 'var(--border)' }}
                transition={{ duration: 0.3 }}
                style={{ flex: 1, height: 2, margin: '0 8px', marginBottom: 20, borderRadius: 1 }}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}

export default function ScheduleForm({ onResult, onLoading }) {
  const [form, setForm] = useState(DEFAULTS)
  const [step, setStep] = useState(0)
  const [dir, setDir] = useState(1)
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

  const goNext = () => { setDir(1); setStep(s => Math.min(s + 1, STEPS.length - 1)) }
  const goPrev = () => { setDir(-1); setStep(s => Math.max(s - 1, 0)) }

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
      if (!result?.decision_id) throw new Error('Unexpected response format from API')
      safeToast.success(`Decision: ${(result.cloud || 'unknown').toUpperCase()} / ${result.region || 'unknown'}`)
      if (typeof onResult === 'function') onResult(result)
      setStep(0)
    } catch (e) {
      const detail = e?.response?.data?.detail || e?.message || 'Unknown error'
      const code = e?.response?.status
      setError(code ? `[${code}] ${detail}` : detail)
      safeToast.error(`Scheduling failed: ${String(detail).slice(0, 80)}`)
    } finally {
      setLoading(false)
      onLoading?.(false)
    }
  }

  const variants = {
    enter: (d) => ({ opacity: 0, x: d > 0 ? 40 : -40 }),
    center: { opacity: 1, x: 0 },
    exit: (d) => ({ opacity: 0, x: d > 0 ? -40 : 40 }),
  }

  return (
    <div className="card" style={{ overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontWeight: 700, fontSize: 15 }}>Submit Workload</span>
          <span className="badge badge-blue">PPO Scheduler</span>
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          {PRESETS.map((preset) => (
            <motion.button
              key={preset.label}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
              type="button"
              onClick={() => applyPreset(preset)}
              disabled={loading}
              style={{
                padding: '4px 10px', background: 'var(--surface2)',
                border: '1px solid var(--border)', color: 'var(--text2)',
                fontSize: 11, fontWeight: 600,
              }}
            >
              {preset.label}
            </motion.button>
          ))}
        </div>
      </div>

      <StepIndicator current={step} total={STEPS.length} />

      {/* Step content */}
      <div style={{ minHeight: 220, position: 'relative', overflow: 'hidden' }}>
        <AnimatePresence custom={dir} mode="wait">
          <motion.div
            key={step}
            custom={dir}
            variants={variants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.22, ease: 'easeOut' }}
          >
            {step === 0 && (
              <div>
                <div style={{ marginBottom: 16 }}>
                  <label>Workload Type</label>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8, marginTop: 6 }}>
                    {['training', 'inference', 'batch', 'streaming'].map((type) => (
                      <motion.button
                        key={type}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.97 }}
                        onClick={() => set('workload_type', type)}
                        disabled={loading}
                        style={{
                          padding: '10px 0', borderRadius: 8, fontWeight: 600,
                          fontSize: 13, textTransform: 'capitalize',
                          background: form.workload_type === type ? 'rgba(59,130,246,0.12)' : 'var(--surface2)',
                          border: `1px solid ${form.workload_type === type ? 'rgba(59,130,246,0.4)' : 'var(--border)'}`,
                          color: form.workload_type === type ? 'var(--accent)' : 'var(--text2)',
                          transition: 'all 0.15s',
                        }}
                      >
                        {type}
                      </motion.button>
                    ))}
                  </div>
                </div>

                <div style={{ display: 'flex', gap: 12 }}>
                  <div style={{ flex: 1 }}>
                    <label>Priority</label>
                    <select value={form.priority} onChange={(e) => set('priority', Number(e.target.value))} disabled={loading}>
                      <option value={1}>1 — Low</option>
                      <option value={2}>2 — Normal</option>
                      <option value={3}>3 — High</option>
                      <option value={4}>4 — Critical</option>
                    </select>
                  </div>
                  <div style={{ flex: 1, display: 'flex', alignItems: 'flex-end' }}>
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.97 }}
                      type="button"
                      onClick={() => set('is_spot_tolerant', !form.is_spot_tolerant)}
                      disabled={loading}
                      style={{
                        width: '100%', padding: '9px 14px',
                        background: form.is_spot_tolerant ? 'rgba(16,185,129,0.12)' : 'var(--surface2)',
                        border: `1px solid ${form.is_spot_tolerant ? 'rgba(16,185,129,0.4)' : 'var(--border)'}`,
                        color: form.is_spot_tolerant ? 'var(--green)' : 'var(--muted)',
                        fontWeight: 600, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                      }}
                    >
                      <Zap size={12} />
                      {form.is_spot_tolerant ? 'Spot ON' : 'Spot OFF'}
                    </motion.button>
                  </div>
                </div>
              </div>
            )}

            {step === 1 && (
              <div>
                <SliderInput label="CPU" value={form.cpu_request_vcpu} min={0.25} max={64} step={0.25}
                  onChange={(v) => set('cpu_request_vcpu', v)} disabled={loading} unit=" vCPU" color="var(--accent)" />
                <SliderInput label="Memory" value={form.memory_request_gb} min={0.5} max={256} step={0.5}
                  onChange={(v) => set('memory_request_gb', v)} disabled={loading} unit=" GB" color="var(--accent2)" />
                <SliderInput label="GPU Count" value={form.gpu_count} min={0} max={8} step={1}
                  onChange={(v) => set('gpu_count', v)} disabled={loading} unit="" color="var(--yellow)" />
                <SliderInput label="Storage" value={form.storage_gb} min={1} max={2000} step={10}
                  onChange={(v) => set('storage_gb', v)} disabled={loading} unit=" GB" color="var(--green)" />
              </div>
            )}

            {step === 2 && (
              <div>
                <SliderInput label="Duration" value={form.expected_duration_hours} min={0.1} max={72} step={0.1}
                  onChange={(v) => set('expected_duration_hours', v)} disabled={loading} unit=" hrs" color="var(--accent)" />
                <SliderInput label="SLA Latency" value={form.sla_latency_ms} min={10} max={5000} step={10}
                  onChange={(v) => set('sla_latency_ms', v)} disabled={loading} unit=" ms" color="var(--accent2)" />

                {/* Summary */}
                <div style={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8, padding: '10px 14px', fontSize: 12 }}>
                  <div style={{ color: 'var(--muted)', marginBottom: 6, fontWeight: 600, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Configuration Summary</div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '4px 16px', color: 'var(--text2)' }}>
                    <span>Type: <b style={{ color: 'var(--text)' }}>{form.workload_type}</b></span>
                    <span>CPU: <b style={{ color: 'var(--text)' }}>{form.cpu_request_vcpu} vCPU</b></span>
                    <span>Memory: <b style={{ color: 'var(--text)' }}>{form.memory_request_gb} GB</b></span>
                    <span>GPU: <b style={{ color: 'var(--text)' }}>{form.gpu_count}</b></span>
                    <span>Storage: <b style={{ color: 'var(--text)' }}>{form.storage_gb} GB</b></span>
                    <span>Spot: <b style={{ color: form.is_spot_tolerant ? 'var(--green)' : 'var(--text)' }}>{form.is_spot_tolerant ? 'Yes' : 'No'}</b></span>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            style={{
              background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
              borderRadius: 8, padding: '10px 14px', color: '#fca5a5',
              fontSize: 12, marginBottom: 14, display: 'flex', gap: 8,
            }}
          >
            <span>⚠</span><span>{error}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Navigation */}
      <div style={{ display: 'flex', gap: 10, marginTop: 16 }}>
        {step > 0 && (
          <motion.button
            whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }}
            onClick={goPrev} disabled={loading}
            style={{
              flex: 1, padding: '11px 0',
              background: 'var(--surface2)', border: '1px solid var(--border)',
              color: 'var(--text2)', fontWeight: 600, fontSize: 14,
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
            }}
          >
            <ChevronLeft size={16} /> Back
          </motion.button>
        )}

        {step < STEPS.length - 1 ? (
          <motion.button
            whileHover={{ scale: 1.02, boxShadow: '0 4px 20px rgba(59,130,246,0.35)' }}
            whileTap={{ scale: 0.97 }}
            onClick={goNext} disabled={loading}
            style={{
              flex: 1, padding: '11px 0',
              background: 'linear-gradient(135deg, var(--accent), var(--accent2))',
              border: 'none', color: '#fff', fontWeight: 700, fontSize: 14,
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
              borderRadius: 10,
            }}
          >
            Next <ChevronRight size={16} />
          </motion.button>
        ) : (
          <motion.button
            whileHover={!loading ? { scale: 1.02, boxShadow: '0 6px 24px rgba(99,102,241,0.5)' } : {}}
            whileTap={!loading ? { scale: 0.97 } : {}}
            onClick={submit} disabled={loading}
            style={{
              flex: 1, padding: '12px 0',
              background: loading ? 'var(--surface2)' : 'linear-gradient(135deg, var(--accent), var(--accent2))',
              border: loading ? '1px solid var(--border)' : 'none',
              color: loading ? 'var(--muted)' : '#fff',
              fontWeight: 700, fontSize: 14,
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              borderRadius: 10, position: 'relative', overflow: 'hidden',
            }}
          >
            {loading ? (
              <>
                <motion.div animate={{ rotate: 360 }} transition={{ duration: 0.8, repeat: Infinity, ease: 'linear' }}>
                  <Loader size={15} />
                </motion.div>
                Computing placement…
              </>
            ) : (
              <>
                <Send size={15} />
                Schedule Workload
              </>
            )}
          </motion.button>
        )}
      </div>
    </div>
  )
}