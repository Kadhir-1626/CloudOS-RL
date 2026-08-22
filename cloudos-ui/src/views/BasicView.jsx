import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
  Zap, Brain, Leaf, Clock, TrendingDown,
  Cloud, Shield, Activity,
  Upload, CheckCircle, ArrowRight,
} from 'lucide-react'
import { useAuth } from '../auth/AuthContext'
import MetricsBar from '../components/MetricsBar'
import { getDecisions } from '../api/client'

const FEATURES = [
  {
    icon: Brain,
    title: 'PPO Reinforcement Learning',
    desc: 'Policy-gradient AI trained on 2M+ timesteps to find optimal cloud placement.',
    color: 'var(--accent)',
  },
  {
    icon: Zap,
    title: 'SHAP Explainability',
    desc: 'Every decision explained with feature attribution — no black boxes.',
    color: 'var(--accent2)',
  },
  {
    icon: Leaf,
    title: 'Carbon-Aware Scheduling',
    desc: 'Minimises carbon intensity by preferring low-emission cloud regions.',
    color: 'var(--green)',
  },
  {
    icon: TrendingDown,
    title: 'Cost Optimisation',
    desc: 'Automatically selects spot, reserved, or on-demand based on workload.',
    color: 'var(--yellow)',
  },
  {
    icon: Clock,
    title: 'SLA Enforcement',
    desc: 'Latency and availability targets enforced across all placement decisions.',
    color: 'var(--green2)',
  },
  {
    icon: Shield,
    title: 'Multi-Cloud Fairness',
    desc: 'AWS, Azure, GCP and hybrid — no vendor lock-in by design.',
    color: 'var(--red)',
  },
]

const CLOUD_COLORS = {
  aws: '#f59e0b', gcp: '#3b82f6', azure: '#6366f1', hybrid: '#10b981',
}

function RecentDecisions({ decisions }) {
  if (!decisions.length) return null

  return (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.5, duration: 0.35 }}
      style={{ marginBottom: 24 }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
        <Activity size={14} color="var(--accent)" />
        <span style={{ fontWeight: 700, fontSize: 14 }}>Recent Decisions</span>
        <span style={{
          background: 'var(--surface2)', border: '1px solid var(--border)',
          borderRadius: 20, padding: '1px 8px', fontSize: 11, color: 'var(--muted)', fontWeight: 600,
        }}>{decisions.length}</span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {decisions.slice(0, 5).map((d, i) => {
          const cloudColor = CLOUD_COLORS[d.cloud] || 'var(--accent)'
          return (
            <motion.div
              key={d.decision_id}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.55 + i * 0.06 }}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '8px 12px', background: 'var(--surface2)',
                borderRadius: 8, borderLeft: `3px solid ${cloudColor}`,
                fontSize: 12,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{ fontWeight: 700, color: cloudColor, textTransform: 'uppercase', fontSize: 11 }}>
                  {d.cloud}
                </span>
                <span style={{ color: 'var(--text2)' }}>{d.region}</span>
                <span style={{ color: 'var(--muted)', fontFamily: 'monospace', fontSize: 10 }}>
                  {d.instance_type}
                </span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={{ color: 'var(--green)', fontWeight: 700 }}>
                  {d.cost_savings_pct?.toFixed(1)}% saved
                </span>
                <span style={{ color: 'var(--muted)', fontSize: 11 }}>
                  {d.latency_ms?.toFixed(0)}ms
                </span>
              </div>
            </motion.div>
          )
        })}
      </div>
    </motion.div>
  )
}

function HowItWorksSection() {
  const steps = [
    {
      number: '01',
      icon: Upload,
      title: 'Submit Workload',
      desc: 'Tell Velox what you need to run.',
    },
    {
      number: '02',
      icon: Brain,
      title: 'AI Decides',
      desc: 'The AI weighs cost, speed, carbon and reliability — instantly.',
    },
    {
      number: '03',
      icon: CheckCircle,
      title: 'Optimal Placement',
      desc: 'Deployed to the best cloud. Every time. Automatically.',
    },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.6, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      style={{ marginTop: 40 }}
    >
      <div
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
        }}
      >
        <span>HOW IT WORKS</span>
        <div style={{ flex: 1, height: 1, background: 'var(--border)', width: '40px' }} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 20 }}>
        {steps.map((step, i) => (
          <motion.div
            key={step.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.65 + i * 0.1, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
            className="card"
            style={{
              position: 'relative',
              padding: '28px 24px',
              display: 'flex',
              flexDirection: 'column',
              gap: 16,
            }}
          >
            {/* Number badge */}
            <div style={{
              position: 'absolute', top: -12, left: 20,
              width: 36, height: 36, borderRadius: 10,
              background: 'linear-gradient(135deg, var(--accent), var(--accent2))',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 12, fontWeight: 800, color: '#fff',
              boxShadow: '0 4px 16px rgba(59,130,246,0.4)',
            }}>
              {step.number}
            </div>

            {/* Icon */}
            <motion.div
              whileHover={{ scale: 1.1, rotate: 3 }}
              transition={{ type: 'spring', stiffness: 400 }}
              style={{
                width: 56, height: 56, borderRadius: 14,
                background: 'linear-gradient(135deg, rgba(59,130,246,0.15), rgba(99,102,241,0.15))',
                border: '1px solid rgba(99,102,241,0.2)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}
            >
              <step.icon size={24} color="var(--accent)" />
            </motion.div>

            {/* Content */}
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 8 }}>{step.title}</div>
              <div style={{ color: 'var(--muted)', fontSize: 13, lineHeight: 1.7 }}>{step.desc}</div>
            </div>

            {/* Arrow between cards */}
            {i < steps.length - 1 && (
              <motion.div
                animate={{ opacity: [0.5, 1, 0.5], x: [0, 4, 0] }}
                transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut', delay: i * 0.2 }}
                style={{
                  position: 'absolute', right: -10, top: '50%', transform: 'translateY(-50%)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  width: 20, height: 20, color: 'var(--accent)',
                }}
              >
                <ArrowRight size={16} />
              </motion.div>
            )}
          </motion.div>
        ))}
      </div>
    </motion.div>
  )
}

export default function BasicView() {
  const { user } = useAuth()
  const [decisions, setDecisions] = useState([])

  useEffect(() => {
    const load = async () => {
      try {
        const d = await getDecisions(20)
        setDecisions(d?.decisions || [])
      } catch { /* non-fatal */ }
    }
    load()
    const t = setInterval(load, 15000)
    return () => clearInterval(t)
  }, [])

  const hour = new Date().getHours()
  const greeting = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening'

  return (
    <div>
      {/* Welcome header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
        style={{ marginBottom: 24 }}
      >
        <h1 style={{ fontSize: 22, fontWeight: 800, marginBottom: 4 }}>
          {greeting},{' '}
          <motion.span
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            style={{
              background: 'linear-gradient(135deg, var(--accent), var(--accent2))',
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
            }}
          >
            {user?.username}
          </motion.span>
        </h1>
        <p style={{ color: 'var(--muted)', fontSize: 13 }}>
          Your workloads. Smarter placement. Every time.
        </p>
      </motion.div>

      {/* Metrics */}
      <MetricsBar />

      {/* Recent decisions */}
      <RecentDecisions decisions={decisions} />

      {/* Hero card */}
      <motion.div
        className="card"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        style={{
          textAlign: 'center', padding: '40px 24px', marginBottom: 24,
          background: 'linear-gradient(135deg, rgba(59,130,246,0.06), rgba(99,102,241,0.06))',
          border: '1px solid rgba(99,102,241,0.15)', position: 'relative', overflow: 'hidden',
        }}
      >
        {/* Background blobs */}
        <motion.div
          animate={{ scale: [1, 1.1, 1], opacity: [0.05, 0.1, 0.05] }}
          transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
          style={{
            position: 'absolute', top: -40, right: -40, width: 200, height: 200,
            borderRadius: '50%', background: 'radial-gradient(circle, var(--accent2), transparent)',
            pointerEvents: 'none',
          }}
        />

        <motion.div
          animate={{ y: [0, -10, 0] }}
          transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
          style={{ fontSize: 44, marginBottom: 16, display: 'inline-block' }}
        >
          ⚡
        </motion.div>

        <div style={{ fontWeight: 800, fontSize: 18, marginBottom: 8 }}>
          AI Cloud Scheduling Platform
        </div>

        <p style={{ color: 'var(--muted)', fontSize: 13, maxWidth: 440, margin: '0 auto 20px', lineHeight: 1.7 }}>
          Velox decides where your workload runs — and why. Every decision is explainable, every choice is optimal.
        </p>

        <div style={{ display: 'flex', gap: 8, justifyContent: 'center', flexWrap: 'wrap' }}>
          {['PPO Reinforcement Learning', 'SHAP Explainability', 'Multi-Cloud', 'Carbon-Aware'].map((tag, i) => (
            <motion.span
              key={tag}
              className="badge badge-blue"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.4 + i * 0.08 }}
            >
              {tag}
            </motion.span>
          ))}
        </div>
      </motion.div>

      {/* Feature grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14 }}>
        {FEATURES.map((f, i) => (
          <motion.div
            key={f.title}
            className="card"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35 + i * 0.07, duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
            whileHover={{ y: -3, boxShadow: `0 8px 28px rgba(0,0,0,0.25), 0 0 0 1px ${f.color}30` }}
            style={{ cursor: 'default' }}
          >
            <motion.div
              whileHover={{ scale: 1.1, rotate: 5 }}
              transition={{ type: 'spring', stiffness: 400 }}
              style={{
                width: 40, height: 40, borderRadius: 10,
                background: `${f.color}18`, border: `1px solid ${f.color}30`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                marginBottom: 12,
              }}
            >
              <f.icon size={18} color={f.color} />
            </motion.div>
            <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 6 }}>{f.title}</div>
            <div style={{ color: 'var(--muted)', fontSize: 12, lineHeight: 1.6 }}>{f.desc}</div>
          </motion.div>
        ))}
      </div>

      {/* How It Works section */}
      <HowItWorksSection />

      {/* Footer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.9 }}
        style={{ marginTop: 32, textAlign: 'center', color: 'var(--muted)', fontSize: 11, borderTop: '1px solid var(--border)', paddingTop: 20 }}
      >
        Velox · Multi-Cloud Workload Scheduler · PPO + SHAP + Kafka + Kubernetes
      </motion.div>
    </div>
  )
}