import { useEffect, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle, XCircle, Info, X } from 'lucide-react'

const listeners = new Set()
let _id = 0

export const toast = {
  success: (msg, duration = 3500) => _emit('success', msg, duration),
  error:   (msg, duration = 5000) => _emit('error',   msg, duration),
  info:    (msg, duration = 3000) => _emit('info',    msg, duration),
}

function _emit(type, message, duration) {
  const id = ++_id
  listeners.forEach(fn => fn({ id, type, message, duration }))
}

const ICONS = {
  success: <CheckCircle size={15} />,
  error:   <XCircle size={15} />,
  info:    <Info size={15} />,
}

const COLORS = {
  success: { bar: 'var(--green)',   bg: 'rgba(16,185,129,0.15)',  border: 'rgba(16,185,129,0.35)',  text: 'var(--green2)' },
  error:   { bar: 'var(--red)',     bg: 'rgba(239,68,68,0.15)',   border: 'rgba(239,68,68,0.35)',   text: '#fca5a5'       },
  info:    { bar: 'var(--accent)',  bg: 'rgba(59,130,246,0.15)',  border: 'rgba(59,130,246,0.35)',  text: '#93c5fd'       },
}

function ToastItem({ id, type, message, duration, onRemove }) {
  const [progress, setProgress] = useState(100)
  const c = COLORS[type] || COLORS.info

  useEffect(() => {
    const start = Date.now()
    const tick = () => {
      const elapsed = Date.now() - start
      const remaining = Math.max(0, 100 - (elapsed / duration) * 100)
      setProgress(remaining)
      if (remaining > 0) requestAnimationFrame(tick)
    }
    const raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [duration])

  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: 80, scale: 0.95 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 80, scale: 0.95 }}
      transition={{ type: 'spring', stiffness: 400, damping: 30 }}
      style={{
        display: 'flex', flexDirection: 'column',
        background: c.bg, border: `1px solid ${c.border}`,
        borderRadius: 10, overflow: 'hidden',
        minWidth: 280, maxWidth: 360,
        boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        pointerEvents: 'all',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '12px 14px', color: c.text }}>
        <motion.div
          initial={{ scale: 0.5, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.1, type: 'spring', stiffness: 500 }}
          style={{ flexShrink: 0 }}
        >
          {ICONS[type]}
        </motion.div>
        <span style={{ flex: 1, lineHeight: 1.4, fontSize: 13, fontWeight: 500 }}>{message}</span>
        <motion.button
          whileHover={{ scale: 1.2, opacity: 1 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => onRemove(id)}
          style={{
            background: 'none', padding: 2,
            color: 'inherit', opacity: 0.5,
            display: 'flex', alignItems: 'center',
            border: 'none', cursor: 'pointer', borderRadius: 4,
          }}
        >
          <X size={13} />
        </motion.button>
      </div>

      {/* Progress bar */}
      <div style={{ height: 3, background: 'rgba(255,255,255,0.08)', flexShrink: 0 }}>
        <motion.div
          style={{ height: '100%', background: c.bar, borderRadius: 0, transformOrigin: 'left' }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.1, ease: 'linear' }}
        />
      </div>
    </motion.div>
  )
}

export default function ToastContainer() {
  const [toasts, setToasts] = useState([])

  const remove = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  useEffect(() => {
    const handler = (t) => {
      setToasts(prev => [...prev, t])
      setTimeout(() => remove(t.id), t.duration)
    }
    listeners.add(handler)
    return () => listeners.delete(handler)
  }, [remove])

  return (
    <div style={{
      position: 'fixed', bottom: 24, right: 24,
      zIndex: 9999, display: 'flex', flexDirection: 'column',
      gap: 10, pointerEvents: 'none',
    }}>
      <AnimatePresence mode="popLayout">
        {toasts.map(t => (
          <ToastItem key={t.id} {...t} onRemove={remove} />
        ))}
      </AnimatePresence>
    </div>
  )
}