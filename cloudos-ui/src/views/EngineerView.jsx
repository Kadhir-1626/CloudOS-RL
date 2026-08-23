import { useState, useCallback, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Eye, Zap, Activity, Loader, Cloud, ArrowRight, TrendingUp, DollarSign, Cpu, Globe, Brain } from 'lucide-react'
import MetricsBar from '../components/MetricsBar'
import ScheduleForm from '../components/ScheduleForm'
import DecisionCard from '../components/DecisionCard'
import DecisionTable from '../components/DecisionTable'
import SystemStatus from '../components/SystemStatus'
import { SkeletonDecisionCard } from '../components/Skeleton'
import { getStatus } from '../api/client'

const TYPEWRITER_WORDS = ['Intelligently', 'Efficiently', 'Sustainably', 'Reliably']

function useTypewriter(words, interval = 2200) {
  const [index, setIndex] = useState(0)
  const [displayed, setDisplayed] = useState('')
  const [deleting, setDeleting] = useState(false)
  const timeoutRef = useRef(null)

  useEffect(() => {
    const current = words[index % words.length]
    if (!deleting && displayed.length < current.length) {
      timeoutRef.current = setTimeout(() => setDisplayed(current.slice(0, displayed.length + 1)), 60)
    } else if (!deleting && displayed.length === current.length) {
      timeoutRef.current = setTimeout(() => setDeleting(true), interval)
    } else if (deleting && displayed.length > 0) {
      timeoutRef.current = setTimeout(() => setDisplayed(displayed.slice(0, -1)), 35)
    } else if (deleting && displayed.length === 0) {
      setDeleting(false)
      setIndex(i => i + 1)
    }
    return () => clearTimeout(timeoutRef.current)
  }, [displayed, deleting, index, words, interval])

  return displayed
}

function SectionLabel({ children }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        fontSize: 11,
        fontWeight: 700,
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        color: 'var(--muted)',
        marginBottom: 24,
        marginTop: 0,
      }}
    >
      <span>{children}</span>
      <div style={{ flex: 1, height: 1, background: 'var(--border)', width: '40px' }} />
    </motion.div>
  )
}

function StatusPill({ label, status, delay = 0 }) {
  const color = status === 'ok' ? 'var(--green)' : status === 'warn' ? 'var(--yellow)' : status === 'loading' ? 'var(--muted)' : 'var(--red)'
  const bg = status === 'ok' ? 'rgba(16,185,129,0.12)' : status === 'warn' ? 'rgba(245,158,11,0.12)' : status === 'loading' ? 'rgba(100,116,139,0.12)' : 'rgba(239,68,68,0.12)'

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.85, y: 8 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ delay, duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
      style={{
        display: 'flex', alignItems: 'center', gap: 6,
        background: bg, border: `1px solid ${color}30`,
        borderRadius: 20, padding: '5px 12px',
        fontSize: 12, fontWeight: 600, color,
      }}
    >
      {status === 'loading' ? (
        <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}>
          <Loader size={10} />
        </motion.div>
      ) : (
        <motion.div
          animate={{ opacity: [1, 0.4, 1], scale: [1, 1.2, 1] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut', delay }}
          style={{ width: 6, height: 6, borderRadius: '50%', background: color }}
        />
      )}
      {label}
    </motion.div>
  )
}

function FloatingDecisionCard({ agentStatus }) {
  const cloud = agentStatus?.last_decision?.cloud || 'AWS'
  const region = agentStatus?.last_decision?.region || 'us-east-1'
  const savings = agentStatus?.last_decision?.cost_savings_pct ?? 65

  const cloudColors = {
    AWS: 'var(--orange)',
    Azure: 'var(--blue)',
    GCP: 'var(--green)',
  }
  const cloudColor = cloudColors[cloud] || 'var(--accent)'

  return (
    <motion.div
      animate={{ y: [0, -12, 0] }}
      transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
      style={{
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: 16,
        padding: '24px',
        maxWidth: 360,
        width: '100%',
        boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
        <motion.div
          animate={{ scale: [1, 1.05, 1] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          style={{
            width: 10, height: 10, borderRadius: '50%',
            background: cloudColor,
            boxShadow: `0 0 12px ${cloudColor}80`,
          }}
        />
        <span style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--muted)' }}>LATEST DECISION</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, textAlign: 'center' }}>
        <div>
          <div style={{ fontSize: 28, fontWeight: 900, color: cloudColor, lineHeight: 1 }}>{cloud}</div>
          <div style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginTop: 4 }}>Cloud Provider</div>
        </div>
        <div style={{ borderLeft: '1px solid var(--border)', borderRight: '1px solid var(--border)' }}>
          <div style={{ fontSize: 28, fontWeight: 900, color: 'var(--text)', lineHeight: 1 }}>{region}</div>
          <div style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginTop: 4 }}>Region</div>
        </div>
        <div>
          <div style={{ fontSize: 28, fontWeight: 900, color: 'var(--green)', lineHeight: 1 }}>{savings}%</div>
          <div style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginTop: 4 }}>Cost Saved</div>
        </div>
      </div>

      <motion.div
        animate={{ opacity: [0.6, 1, 0.6] }}
        transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
        style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, fontSize: 11, color: 'var(--muted)' }}
      >
        <TrendingUp size={12} /> Live PPO inference
      </motion.div>
    </motion.div>
  )
}

function SystemIntelligenceCard() {
  const entropy = -10.4
  const entropyPct = Math.min(Math.abs(entropy / 15) * 100, 100)

  const rewardWeights = [
    { label: 'Cost', value: 30, color: 'var(--green)' },
    { label: 'Latency', value: 28, color: 'var(--accent)' },
    { label: 'Carbon', value: 22, color: 'var(--green2)' },
    { label: 'SLA', value: 20, color: 'var(--yellow)' },
  ]

  return (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}
    >
      {/* Card 1 — Model Confidence */}
      <div style={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 12, padding: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
          <Brain size={16} color="var(--accent2)" />
          <span style={{ fontWeight: 700, fontSize: 14 }}>Model Confidence</span>
        </div>
        <div style={{ marginBottom: 16 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
            <span style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Policy Entropy</span>
            <span style={{ fontSize: 11, color: 'var(--muted)', fontFamily: 'monospace' }}>{entropy}</span>
          </div>
          <div style={{ position: 'relative', height: 8, background: 'var(--surface)', borderRadius: 4, overflow: 'hidden' }}>
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${entropyPct}%` }}
              transition={{ delay: 0.2, duration: 0.8, ease: 'easeOut' }}
              style={{
                height: '100%',
                background: 'var(--accent2)',
                borderRadius: 4,
              }}
            />
          </div>
        </div>

        <div style={{ fontSize: 10, color: 'var(--muted)', lineHeight: 1.6, padding: '8px', background: 'var(--surface)', borderRadius: 6, border: '1px solid var(--border)' }}>
          Higher entropy = more exploration. Negative values indicate confident policy convergence.
        </div>
      </div>

      {/* Card 2 — Reward Weights */}
      <div style={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 12, padding: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
          <TrendingUp size={16} color="var(--accent)" />
          <span style={{ fontWeight: 700, fontSize: 14 }}>Reward Weights</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {rewardWeights.map((r, i) => (
            <div key={r.label} style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
                <span style={{ color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>{r.label}</span>
                <span style={{ color: r.color, fontWeight: 700 }}>{r.value}%</span>
              </div>
              <div style={{ position: 'relative', height: 6, background: 'var(--surface)', borderRadius: 3, overflow: 'hidden' }}>
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${r.value}%` }}
                  transition={{ delay: 0.2 + i * 0.1, duration: 0.8, ease: 'easeOut' }}
                  style={{
                    height: '100%',
                    background: r.color,
                    borderRadius: 3,
                  }}
                />
              </div>
            </div>
          ))}
        </div>

        <div style={{ marginTop: 16, padding: '8px', background: 'var(--surface)', borderRadius: 6, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, marginBottom: 6 }}>Reward Formula</div>
          <div style={{ fontSize: 10, color: 'var(--text2)', fontFamily: 'monospace', lineHeight: 1.8 }}>
            R = 0.30·Cost + 0.28·Latency + 0.22·Carbon + 0.20·SLA
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function SectionWrapper({ id, children, delay = 0 }) {
  return (
    <motion.div
      id={id}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.div>
  )
}

export default function EngineerView() {
  const [lastDecision, setLastDecision] = useState(null)
  const [scheduling, setScheduling] = useState(false)
  const [agentStatus, setAgentStatus] = useState(null)

  const scrollToSection = useCallback((anchor) => {
    const el = document.getElementById(`section-${anchor}`)
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])

  useEffect(() => {
    let isMounted = true
    const load = async () => {
      try {
        const s = await getStatus()
        if (isMounted) setAgentStatus(s)
      } catch { /* non-fatal */ }
    }
    load()
    const t = setInterval(load, 15000)
    return () => { isMounted = false; clearInterval(t) }
  }, [])

  const handleResult = useCallback((d) => setLastDecision(d), [])
  const handleLoading = useCallback((v) => setScheduling(v), [])

  const agentLoaded = !!agentStatus?.agent_loaded
  const shapReady = !!agentStatus?.shap_ready
  const typeword = useTypewriter(TYPEWRITER_WORDS)

  return (
    <div style={{ scrollBehavior: 'smooth' }}>
      {/* SECTION 1 — HERO */}
      <SectionWrapper id="section-hero" delay={0} style={{ minHeight: '60vh', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '0 24px 60px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 60, alignItems: 'center', maxWidth: 1200, width: '100%' }}>
          {/* Left side */}
          <div>
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
              style={{
                fontSize: 11, fontWeight: 700, textTransform: 'uppercase',
                letterSpacing: '0.1em', color: 'var(--accent)', marginBottom: 16,
                display: 'flex', alignItems: 'center', gap: 6,
              }}
            >
              <Zap size={11} /> REINFORCEMENT LEARNING · MULTI-CLOUD
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
              style={{ fontSize: 48, fontWeight: 900, lineHeight: 1.1, letterSpacing: '-0.03em', marginBottom: 24, maxWidth: 560 }}
            >
              <span style={{ display: 'block' }}>Schedule Smarter.</span>
              <span style={{ display: 'block', background: 'linear-gradient(135deg, var(--accent), var(--accent2))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                Deploy Faster.
              </span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25, duration: 0.4 }}
              style={{ color: 'var(--muted)', fontSize: 16, lineHeight: 1.7, maxWidth: 480, marginBottom: 28 }}
            >
              Not just fast — intelligent. Velox evaluates thousands of placement options in milliseconds so you never overpay, overpollute, or miss an SLA.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.35 }}
              style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 32 }}
            >
              <motion.button
                whileHover={{ y: -2, boxShadow: '0 8px 24px rgba(59,130,246,0.4)' }}
                whileTap={{ scale: 0.96 }}
                onClick={() => scrollToSection('schedule')}
                style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  padding: '14px 28px',
                  background: 'linear-gradient(135deg, var(--accent), var(--accent2))',
                  border: 'none',
                  color: '#fff',
                  fontWeight: 600, fontSize: 14, borderRadius: 10,
                }}
              >
                Schedule a Workload <ArrowRight size={16} />
              </motion.button>
              <motion.button
                whileHover={{ y: -2, boxShadow: '0 6px 16px rgba(0,0,0,0.2)' }}
                whileTap={{ scale: 0.96 }}
                onClick={() => scrollToSection('decisions')}
                style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  padding: '14px 28px',
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  color: 'var(--text)',
                  fontWeight: 600, fontSize: 14, borderRadius: 10,
                }}
              >
                View Decisions
              </motion.button>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.35, duration: 0.35 }}
              style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}
            >
              <StatusPill label="AI Ready" status={agentLoaded ? 'ok' : 'loading'} delay={0.4} />
              <StatusPill label="XAI Ready" status={shapReady ? 'ok' : 'loading'} delay={0.47} />
              <StatusPill label="Streaming" status="ok" delay={0.54} />
              <StatusPill label="Orchestrated" status="ok" delay={0.61} />
            </motion.div>
          </div>

          {/* Right side - Floating Decision Card */}
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <FloatingDecisionCard agentStatus={agentStatus} />
          </div>
        </div>
      </SectionWrapper>

      {/* SECTION 2 — METRICS */}
      <SectionWrapper id="section-metrics" delay={0.1} style={{ padding: '48px 24px 0', maxWidth: 1200, margin: '0 auto', width: '100%' }}>
        <SectionLabel>LIVE PERFORMANCE</SectionLabel>
        <MetricsBar />
      </SectionWrapper>

      {/* SECTION 3 — SCHEDULE + DECISION */}
      <SectionWrapper id="section-schedule" delay={0.15} style={{ padding: '48px 24px 0', maxWidth: 1200, margin: '0 auto', width: '100%' }}>
        <SectionLabel>WORKLOAD SCHEDULER</SectionLabel>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
          <ScheduleForm onResult={handleResult} onLoading={handleLoading} />
          <div id="section-explainability">
            <AnimatePresence mode="wait">
              {scheduling ? (
                <motion.div key="skeleton" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <SkeletonDecisionCard />
                </motion.div>
              ) : lastDecision ? (
                <DecisionCard key={lastDecision.decision_id} decision={lastDecision} />
              ) : (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.3 }}
                  className="card"
                  style={{
                    display: 'flex', flexDirection: 'column', alignItems: 'center',
                    justifyContent: 'center', minHeight: 380, gap: 14, textAlign: 'center',
                  }}
                >
                  <motion.div
                    animate={{ y: [0, -8, 0] }}
                    transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
                    style={{
                      width: 56, height: 56, borderRadius: 14,
                      background: 'linear-gradient(135deg, rgba(59,130,246,0.15), rgba(99,102,241,0.15))',
                      border: '1px solid rgba(99,102,241,0.2)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}
                  >
                    <Cloud size={22} color="var(--accent2)" />
                  </motion.div>

                  <div>
                    <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 6 }}>Awaiting workload</div>
                    <div style={{ color: 'var(--muted)', fontSize: 12, maxWidth: 240, lineHeight: 1.7 }}>
                      Configure a workload and click{' '}
                      <span style={{ color: 'var(--accent)', fontWeight: 600 }}>Schedule Workload</span>{' '}
                      to get an AI placement decision with SHAP explanation.
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'center' }}>
                    {['PPO Model', 'SHAP XAI', 'Multi-Cloud', 'Carbon-Aware'].map((tag, i) => (
                      <motion.span
                        key={tag}
                        className="badge badge-blue"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: i * 0.08 }}
                      >
                        {tag}
                      </motion.span>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </SectionWrapper>

      {/* SECTION 4 — DECISION HISTORY */}
      <SectionWrapper id="section-decisions" delay={0.2} style={{ padding: '48px 24px 0', maxWidth: 1200, margin: '0 auto', width: '100%' }}>
        <SectionLabel>DECISION LOG</SectionLabel>
        <DecisionTable />
      </SectionWrapper>

      {/* SECTION 5 — SYSTEM INTELLIGENCE */}
      <SectionWrapper id="section-intelligence" delay={0.25} style={{ padding: '48px 24px 48px', maxWidth: 1200, margin: '0 auto', width: '100%' }}>
        <SectionLabel>SYSTEM INTELLIGENCE</SectionLabel>
        <SystemIntelligenceCard />
      </SectionWrapper>

      {/* Footer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        style={{
          marginTop: 32, textAlign: 'center', color: 'var(--muted)',
          fontSize: 11, borderTop: '1px solid var(--border)', paddingTop: 20,
        }}
      >
        Velox · Multi-Cloud Workload Scheduler · PPO + SHAP + Kafka + Kubernetes
      </motion.div>
    </div>
  )
}