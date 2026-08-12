import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuth } from './AuthContext'
import { register as apiRegister } from '../api/client'
import { Zap, Loader, Eye, EyeOff, ArrowRight, UserPlus, LogIn } from 'lucide-react'

function ParticleCanvas() {
  const canvasRef = useRef(null)
  const mouseRef = useRef({ x: -999, y: -999 })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    let animationFrameId = 0

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resize()
    window.addEventListener('resize', resize)

    const onMouseMove = (e) => {
      mouseRef.current = { x: e.clientX, y: e.clientY }
    }
    window.addEventListener('mousemove', onMouseMove)

    const particleCount = 55
    const particles = Array.from({ length: particleCount }, () => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      r: Math.random() * 1.8 + 0.4,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      alpha: Math.random() * 0.5 + 0.15,
    }))

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      const mouse = mouseRef.current

      particles.forEach((p) => {
        const dx = p.x - mouse.x
        const dy = p.y - mouse.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < 120) {
          const force = (120 - dist) / 120
          p.vx += (dx / dist) * force * 0.04
          p.vy += (dy / dist) * force * 0.04
        }
        p.vx *= 0.98
        p.vy *= 0.98
        p.x += p.vx
        p.y += p.vy
        if (p.x < 0) p.x = canvas.width
        if (p.x > canvas.width) p.x = 0
        if (p.y < 0) p.y = canvas.height
        if (p.y > canvas.height) p.y = 0
      })

      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x
          const dy = particles[i].y - particles[j].y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < 130) {
            ctx.beginPath()
            ctx.strokeStyle = `rgba(99,102,241,${0.15 * (1 - dist / 130)})`
            ctx.lineWidth = 0.6
            ctx.moveTo(particles[i].x, particles[i].y)
            ctx.lineTo(particles[j].x, particles[j].y)
            ctx.stroke()
          }
        }
      }

      particles.forEach((p) => {
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(139,92,246,${p.alpha})`
        ctx.fill()
      })

      animationFrameId = requestAnimationFrame(draw)
    }

    draw()
    return () => {
      cancelAnimationFrame(animationFrameId)
      window.removeEventListener('resize', resize)
      window.removeEventListener('mousemove', onMouseMove)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      style={{ position: 'fixed', inset: 0, zIndex: 0, pointerEvents: 'none', opacity: 0.75 }}
    />
  )
}

function InputField({ label, type = 'text', value, onChange, onKeyDown, placeholder, autoFocus, autoComplete, rightSlot }) {
  const [focused, setFocused] = useState(false)
  return (
    <div style={{ marginBottom: 16 }}>
      <label>{label}</label>
      <div style={{ position: 'relative' }}>
        <input
          type={type}
          value={value}
          onChange={onChange}
          onKeyDown={onKeyDown}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder={placeholder}
          autoFocus={autoFocus}
          autoComplete={autoComplete}
          style={{ paddingRight: rightSlot ? 42 : 12 }}
        />
        <motion.div
          initial={false}
          animate={{ scaleX: focused ? 1 : 0, opacity: focused ? 1 : 0 }}
          transition={{ duration: 0.2 }}
          style={{
            position: 'absolute',
            bottom: 0,
            left: 8,
            right: 8,
            height: 2,
            borderRadius: 2,
            background: 'linear-gradient(90deg, var(--accent), var(--accent2))',
            transformOrigin: 'left',
          }}
        />
        {rightSlot && (
          <div style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)' }}>
            {rightSlot}
          </div>
        )}
      </div>
    </div>
  )
}

export default function LoginPage() {
  const { login } = useAuth()
  const [mode, setMode] = useState('signin')
  const [form, setForm] = useState({ username: '', password: '', confirm: '' })
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)
  const [loading, setLoading] = useState(false)

  const updateField = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }))
    setError(null)
    setSuccess(null)
  }

  const switchMode = (nextMode) => {
    if (loading) return
    setMode(nextMode)
    setForm({ username: '', password: '', confirm: '' })
    setError(null)
    setSuccess(null)
    setShowPassword(false)
    setShowConfirm(false)
  }

  const handleSignIn = async () => {
    const username = form.username.trim()
    const password = form.password
    if (!username || !password) { setError('Please enter your username and password.'); return }
    setLoading(true); setError(null); setSuccess(null)
    try { await login(username, password) }
    catch (e) { setError(e?.response?.data?.detail || 'Invalid username or password.') }
    finally { setLoading(false) }
  }

  const handleSignUp = async () => {
    const username = form.username.trim()
    const password = form.password
    const confirm = form.confirm
    if (!username) { setError('Username is required.'); return }
    if (username.length < 3) { setError('Username must be at least 3 characters.'); return }
    if (password.length < 6) { setError('Password must be at least 6 characters.'); return }
    if (password !== confirm) { setError('Passwords do not match.'); return }
    setLoading(true); setError(null); setSuccess(null)
    try {
      const result = await apiRegister(username, password, confirm)
      setMode('signin')
      setForm({ username: result?.username || username, password: '', confirm: '' })
      setShowPassword(false); setShowConfirm(false); setError(null)
      setSuccess(result?.message || `Account created. Sign in as "${result?.username || username}".`)
    }
    catch (e) { setError(e?.response?.data?.detail || 'Registration failed. Please try again.') }
    finally { setLoading(false) }
  }

  const handleSubmit = async () => {
    if (mode === 'signin') await handleSignIn()
    else await handleSignUp()
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') { e.preventDefault(); handleSubmit() }
  }

  return (
    <div style={{
      minHeight: '100vh', display: 'flex', alignItems: 'center',
      justifyContent: 'center', background: 'var(--bg)',
      position: 'relative', overflow: 'hidden', padding: 24,
    }}>
      <ParticleCanvas />

      {/* Ambient glow */}
      <div aria-hidden="true" style={{
        position: 'absolute', width: 700, height: 700, borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(99,102,241,0.06) 0%, transparent 70%)',
        pointerEvents: 'none', zIndex: 1,
      }} />

      <motion.div
        initial={{ opacity: 0, y: 32, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        style={{ width: '100%', maxWidth: 420, zIndex: 2, position: 'relative' }}
      >
        {/* Logo */}
        <div style={{ textAlign: 'center', marginBottom: 28 }}>
          <motion.div
            animate={{ boxShadow: ['0 8px 32px rgba(99,102,241,0.3)', '0 8px 48px rgba(99,102,241,0.6)', '0 8px 32px rgba(99,102,241,0.3)'] }}
            transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
            style={{
              width: 56, height: 56, borderRadius: 16,
              margin: '0 auto 14px',
              background: 'linear-gradient(135deg, var(--accent), var(--accent2))',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}
          >
            <Zap size={26} color="#fff" />
          </motion.div>
          <h1 style={{ fontSize: 22, fontWeight: 800, marginBottom: 4, letterSpacing: '-0.02em' }}>
            CloudOS-RL
          </h1>
          <p style={{ color: 'var(--muted)', fontSize: 13 }}>AI-Native Multi-Cloud Scheduler</p>
        </div>

        {/* Tab switcher */}
        <div style={{
          display: 'flex', background: 'var(--surface2)',
          border: '1px solid var(--border)', borderRadius: 10,
          padding: 4, marginBottom: 20, gap: 4, position: 'relative',
        }}>
          {[{ id: 'signin', label: 'Sign In', Icon: LogIn }, { id: 'signup', label: 'Sign Up', Icon: UserPlus }].map(({ id, label, Icon }) => (
            <button
              key={id}
              type="button"
              onClick={() => switchMode(id)}
              disabled={loading}
              style={{
                flex: 1, padding: '9px 0', borderRadius: 8,
                fontWeight: 600, fontSize: 13,
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                background: mode === id ? 'linear-gradient(135deg, var(--accent), var(--accent2))' : 'transparent',
                color: mode === id ? '#fff' : 'var(--muted)',
                border: 'none',
                transition: 'all 0.22s cubic-bezier(0.16,1,0.3,1)',
                transform: mode === id ? 'scale(1.02)' : 'scale(1)',
              }}
            >
              <Icon size={13} />
              {label}
            </button>
          ))}
        </div>

        {/* Form card */}
        <div className="card" style={{ padding: '28px 28px 24px', boxShadow: '0 24px 64px rgba(0,0,0,0.45), 0 0 0 1px rgba(99,102,241,0.08)', borderColor: 'rgba(99,102,241,0.12)' }}>
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 4 }}>
              {mode === 'signin' ? 'Welcome back' : 'Create account'}
            </div>
            <div style={{ color: 'var(--muted)', fontSize: 12 }}>
              {mode === 'signin' ? 'Sign in with your credentials' : 'Create a new account to access CloudOS-RL'}
            </div>
          </div>

          <AnimatePresence mode="wait">
            {success && (
              <motion.div
                key="success"
                initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
                style={{
                  background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)',
                  borderRadius: 8, padding: '10px 14px', color: 'var(--green)',
                  fontSize: 13, marginBottom: 18, display: 'flex', alignItems: 'center', gap: 8,
                }}
              >
                <span>✓</span><span>{success}</span>
              </motion.div>
            )}
          </AnimatePresence>

          <AnimatePresence mode="wait">
            <motion.div
              key={mode}
              initial={{ opacity: 0, x: mode === 'signin' ? -16 : 16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: mode === 'signin' ? 16 : -16 }}
              transition={{ duration: 0.22, ease: 'easeOut' }}
            >
              <InputField
                label="Username"
                value={form.username}
                onChange={(e) => updateField('username', e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter your username"
                autoFocus
                autoComplete="username"
              />

              <InputField
                label="Password"
                type={showPassword ? 'text' : 'password'}
                value={form.password}
                onChange={(e) => updateField('password', e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter your password"
                autoComplete={mode === 'signin' ? 'current-password' : 'new-password'}
                rightSlot={
                  <button
                    type="button"
                    onClick={() => setShowPassword((p) => !p)}
                    style={{ background: 'none', color: 'var(--muted)', padding: 0, border: 'none', borderRadius: 0, display: 'flex' }}
                  >
                    {showPassword ? <EyeOff size={15} /> : <Eye size={15} />}
                  </button>
                }
              />

              {mode === 'signup' && (
                <InputField
                  label="Confirm Password"
                  type={showConfirm ? 'text' : 'password'}
                  value={form.confirm}
                  onChange={(e) => updateField('confirm', e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Repeat your password"
                  autoComplete="new-password"
                  rightSlot={
                    <button
                      type="button"
                      onClick={() => setShowConfirm((p) => !p)}
                      style={{ background: 'none', color: 'var(--muted)', padding: 0, border: 'none', borderRadius: 0, display: 'flex' }}
                    >
                      {showConfirm ? <EyeOff size={15} /> : <Eye size={15} />}
                    </button>
                  }
                />
              )}
            </motion.div>
          </AnimatePresence>

          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                style={{
                  background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
                  borderRadius: 8, padding: '10px 14px', color: '#fca5a5',
                  fontSize: 12, marginBottom: 18, display: 'flex', alignItems: 'flex-start', gap: 8,
                }}
              >
                <span style={{ flexShrink: 0, marginTop: 1 }}>⚠</span>
                <span>{error}</span>
              </motion.div>
            )}
          </AnimatePresence>

          <motion.button
            type="button"
            onClick={handleSubmit}
            disabled={loading}
            whileHover={!loading ? { y: -2, boxShadow: '0 8px 28px rgba(99,102,241,0.45)' } : {}}
            whileTap={!loading ? { scale: 0.97 } : {}}
            style={{
              width: '100%', padding: '12px 0',
              background: loading ? 'var(--surface2)' : 'linear-gradient(135deg, var(--accent), var(--accent2))',
              color: loading ? 'var(--muted)' : '#fff',
              fontWeight: 700, fontSize: 14,
              border: loading ? '1px solid var(--border)' : 'none',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              letterSpacing: '0.02em', borderRadius: 10,
              position: 'relative', overflow: 'hidden',
            }}
          >
            {!loading && (
              <motion.div
                initial={{ x: '-100%', opacity: 0 }}
                whileHover={{ x: '100%', opacity: 0.15 }}
                transition={{ duration: 0.5 }}
                style={{
                  position: 'absolute', inset: 0,
                  background: 'linear-gradient(90deg, transparent, white, transparent)',
                  pointerEvents: 'none',
                }}
              />
            )}
            {loading ? (
              <>
                <motion.div animate={{ rotate: 360 }} transition={{ duration: 0.8, repeat: Infinity, ease: 'linear' }}>
                  <Loader size={15} />
                </motion.div>
                {mode === 'signin' ? 'Signing in…' : 'Creating account…'}
              </>
            ) : (
              <>
                {mode === 'signin' ? <LogIn size={15} /> : <UserPlus size={15} />}
                {mode === 'signin' ? 'Sign In' : 'Create Account'}
                <ArrowRight size={14} />
              </>
            )}
          </motion.button>
        </div>

        <div style={{ textAlign: 'center', marginTop: 20, fontSize: 11, color: 'var(--muted)', lineHeight: 1.6 }}>
          CloudOS-RL · AI-Native Multi-Cloud Scheduler
          <br />
          PPO · SHAP · Kafka · Kubernetes
        </div>
      </motion.div>
    </div>
  )
}