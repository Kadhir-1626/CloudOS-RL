import { useState, useCallback, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Eye, Zap, Activity, Loader, Cloud } from 'lucide-react'
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

function HeroBanner({ agentStatus }) {
  const agentLoaded = !!agentStatus?.agent_loaded
  const shapReady = !!agentStatus?.shap_ready
  const typeword = useTypewriter(TYPEWRITER_WORDS)

  const scrollToSection = (anchor) => {
    const el = document.getElementById(`section-${anchor}`)
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      style={{
        background: 'linear-gradient(135deg, rgba(59,130,246,0.08) 0%, rgba(99,102,241,0.06) 100%)',
        border: '1px solid rgba(59,130,246,0.15)',
        borderRadius: 16, padding: '28px 32px',
        marginBottom: 24, position: 'relative', overflow: 'hidden',
      }}
    >
      {/* Animated mesh blobs */}
      <motion.div
        animate={{ x: [0, 20, 0], y: [0, -10, 0] }}
        transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
        style={{
          position: 'absolute', top: -60, right: -60, width: 220, height: 220,
          borderRadius: '50%', background: 'radial-gradient(circle, rgba(99,102,241,0.1), transparent)',
          pointerEvents: 'none',
        }}
      />
      <motion.div
        animate={{ x: [0, -15, 0], y: [0, 15, 0] }}
        transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut', delay: 1 }}
        style={{
          position: 'absolute', bottom: -40, left: '40%', width: 160, height: 160,
          borderRadius: '50%', background: 'radial-gradient(circle, rgba(59,130,246,0.07), transparent)',
          pointerEvents: 'none',
        }}
      />

      <div style={{ position: 'relative' }}>
        <motion.div
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          style={{
            fontSize: 11, fontWeight: 700, textTransform: 'uppercase',
            letterSpacing: '0.1em', color: 'var(--accent)', marginBottom: 10,
            display: 'flex', alignItems: 'center', gap: 6,
          }}
        >
          <Zap size={11} /> AI-Native Multi-Cloud Scheduler
        </motion.div>

        <motion.h2
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          style={{ fontSize: 26, fontWeight: 900, lineHeight: 1.2, letterSpacing: '-0.02em', marginBottom: 8 }}
        >
          Optimize Cloud Placement{' '}
          <span style={{
            background: 'linear-gradient(135deg, var(--accent), var(--accent2))',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
          }}>
            {typeword}
            <motion.span
              animate={{ opacity: [1, 0, 1] }}
              transition={{ duration: 0.8, repeat: Infinity }}
              style={{ display: 'inline-block', width: 2, height: '1em', background: 'var(--accent)', marginLeft: 2, verticalAlign: 'text-bottom' }}
            />
          </span>
        </motion.h2>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          style={{ color: 'var(--muted)', fontSize: 13, marginBottom: 20, maxWidth: 500, lineHeight: 1.7 }}
        >
          PPO reinforcement learning across cost, latency, carbon, and SLA simultaneously.
          SHAP-powered explainability for every placement decision.
        </motion.p>

        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 20 }}>
          <StatusPill label="RL Agent Active" status={agentLoaded ? 'ok' : 'loading'} delay={0.25} />
          <StatusPill label="SHAP Ready"      status={shapReady  ? 'ok' : 'loading'} delay={0.32} />
          <StatusPill label="Kafka Connected" status="ok" delay={0.39} />
          <StatusPill label="Kubernetes Healthy" status="ok" delay={0.46} />
        </div>

        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
          style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}
        >
          {[
            { label: 'Schedule Workload', icon: Send,     color: 'var(--accent)',  anchor: 'schedule',      primary: true  },
            { label: 'View Decisions',    icon: Eye,      color: 'var(--green)',   anchor: 'decisions',     primary: false },
            { label: 'Explain Decision',  icon: Zap,      color: 'var(--accent2)', anchor: 'explainability',primary: false },
            { label: 'Live Metrics',      icon: Activity, color: 'var(--muted)',   anchor: 'metrics',       primary: false },
          ].map(({ label, icon: Icon, color, anchor, primary }) => (
            <motion.button
              key={label}
              whileHover={{ y: -2, boxShadow: primary ? '0 6px 20px rgba(59,130,246,0.4)' : '0 4px 12px rgba(0,0,0,0.2)' }}
              whileTap={{ scale: 0.96 }}
              onClick={() => scrollToSection(anchor)}
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '8px 16px',
                background: primary ? 'linear-gradient(135deg, var(--accent), var(--accent2))' : 'var(--surface)',
                border: `1px solid ${primary ? 'transparent' : 'var(--border)'}`,
                color: primary ? '#fff' : color,
                fontWeight: 600, fontSize: 13, borderRadius: 8,
              }}
            >
              <Icon size={13} />{label}
            </motion.button>
          ))}
        </motion.div>
      </div>
    </motion.div>
  )
}

function EmptyDecisionState() {
  return (
    <motion.div
      className="card"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
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

  return (
    <div>
      <HeroBanner agentStatus={agentStatus} />

      <SectionWrapper id="section-dashboard" delay={0.1}>
        <SystemStatus />
      </SectionWrapper>

      <SectionWrapper id="section-metrics" delay={0.15}>
        <MetricsBar />
      </SectionWrapper>

      <SectionWrapper delay={0.2}>
        <div
          id="section-schedule"
          style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 24 }}
        >
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
                <EmptyDecisionState />
              )}
            </AnimatePresence>
          </div>
        </div>
      </SectionWrapper>

      <SectionWrapper id="section-decisions" delay={0.25}>
        <DecisionTable />
      </SectionWrapper>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        style={{
          marginTop: 32, textAlign: 'center', color: 'var(--muted)',
          fontSize: 11, borderTop: '1px solid var(--border)', paddingTop: 20,
        }}
      >
        CloudOS-RL · AI-Native Multi-Cloud Scheduler · PPO + SHAP + Kafka + Kubernetes
      </motion.div>
    </div>
  )
}