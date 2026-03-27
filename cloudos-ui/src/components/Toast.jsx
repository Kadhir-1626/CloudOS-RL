import { useEffect, useState, useCallback } from 'react'
import { CheckCircle, XCircle, Info, X } from 'lucide-react'

// Singleton event bus — no extra lib needed
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
  error:   <XCircle    size={15} />,
  info:    <Info       size={15} />,
}

function ToastItem({ id, type, message, onRemove }) {
  useEffect(() => {
    // auto-remove handled by parent via duration
    return () => {}
  }, [])

  return (
    <div className={`toast toast-${type}`}>
      {ICONS[type]}
      <span style={{ flex: 1, lineHeight: 1.4 }}>{message}</span>
      <button
        onClick={() => onRemove(id)}
        style={{
          background: 'none', padding: 2,
          color: 'inherit', opacity: 0.6,
          display: 'flex', alignItems: 'center',
        }}
      >
        <X size={13} />
      </button>
    </div>
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

  if (toasts.length === 0) return null

  return (
    <div className="toast-container">
      {toasts.map(t => (
        <ToastItem key={t.id} {...t} onRemove={remove} />
      ))}
    </div>
  )
}