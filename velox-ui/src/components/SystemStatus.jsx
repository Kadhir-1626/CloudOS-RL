import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { getStatus } from '../api/client'

export default function SystemStatus() {
  const [s, setS] = useState(null)

  useEffect(() => {
    const load = async () => {
      try { setS(await getStatus()) } catch {}
    }
    load()
    const t = setInterval(load, 8000)
    return () => clearInterval(t)
  }, [])

  if (!s) return null

  const items = [
    { label: 'RL Agent',  ok: s.agent_loaded, value: null },
    { label: 'SHAP',      ok: s.shap_ready,   value: null },
    { label: 'Decisions', ok: true,            value: s.decisions_served },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: -6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
      style={{
        display: 'flex', gap: 16, padding: '8px 16px',
        background: 'var(--surface)', border: '1px solid var(--border)',
        borderRadius: 8, fontSize: 11, alignItems: 'center',
        marginBottom: 20,
      }}
    >
      <span style={{ color: 'var(--muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
        System
      </span>

      {items.map(({ label, ok, value }, i) => (
        <motion.div
          key={label}
          initial={{ opacity: 0, x: -6 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.08, duration: 0.25 }}
          style={{ display: 'flex', alignItems: 'center', gap: 5 }}
        >
          {ok ? (
            <motion.div
              animate={{ opacity: [1, 0.3, 1], scale: [1, 1.2, 1] }}
              transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut', delay: i * 0.4 }}
              style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--green)', boxShadow: '0 0 6px var(--green)' }}
            />
          ) : (
            <motion.div
              animate={{ opacity: [1, 0.4, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
              style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--red)' }}
            />
          )}
          <span style={{ color: 'var(--text2)' }}>
            {label}
            {value != null && (
              <AnimatePresence mode="wait">
                <motion.span
                  key={value}
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 4 }}
                  transition={{ duration: 0.2 }}
                >
                  : {value}
                </motion.span>
              </AnimatePresence>
            )}
          </span>
        </motion.div>
      ))}
    </motion.div>
  )
}