import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuth } from './AuthContext'
import { register as apiRegister } from '../api/client'
import { Zap, Loader, Eye, EyeOff, ArrowRight, UserPlus, LogIn, Chrome } from 'lucide-react'

function ParticleCanvas({ mousePos }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    let animationFrameId = 0

    const resize = () => {
      const rect = canvas.parentElement?.getBoundingClientRect()
      if (rect) {
        canvas.width = rect.width * window.devicePixelRatio
        canvas.height = rect.height * window.devicePixelRatio
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
      }
    }
    resize()
    window.addEventListener('resize', resize)

    const particles = Array.from({ length: 60 }, () => ({
      x: Math.random() * (canvas.parentElement?.clientWidth || window.innerWidth),
      y: Math.random() * (canvas.parentElement?.clientHeight || window.innerHeight),
      r: Math.random() * 1.5 + 0.3,
      vx: (Math.random() - 0.5) * 0.25,
      vy: (Math.random() - 0.5) * 0.25,
      alpha: Math.random() * 0.4 + 0.1,
    }))

    const draw = () => {
      const width = canvas.parentElement?.clientWidth || window.innerWidth
      const height = canvas.parentElement?.clientHeight || window.innerHeight
      ctx.clearRect(0, 0, width, height)

      const mouse = mousePos.current

      particles.forEach((p) => {
        const dx = p.x - mouse.x
        const dy = p.y - mouse.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < 100 && mouse.x > 0) {
          const force = (100 - dist) / 100
          p.vx += (dx / dist) * force * 0.03
          p.vy += (dy / dist) * force * 0.03
        }
        p.vx *= 0.985
        p.vy *= 0.985
        p.x += p.vx
        p.y += p.vy
        if (p.x < 0) p.x = width
        if (p.x > width) p.x = 0
        if (p.y < 0) p.y = height
        if (p.y > height) p.y = 0
      })

      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x
          const dy = particles[i].y - particles[j].y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < 100) {
            ctx.beginPath()
            ctx.strokeStyle = `rgba(99,102,241,${0.1 * (1 - dist / 100)})`
            ctx.lineWidth = 0.5
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
    }
  }, [mousePos])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      style={{ position: 'absolute', inset: 0, zIndex: 0, pointerEvents: 'none' }}
    />
  )
}

function StatCard({ value, label, icon: Icon, color, delay }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      style={{
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: 12,
        padding: '16px 20px',
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        minWidth: 180,
      }}
    >
      <div style={{ width: 40, height: 40, borderRadius: 10, background: `${color}20`, border: `1px solid ${color}40`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Icon size={18} color={color} />
      </div>
      <div>
        <div style={{ fontSize: 18, fontWeight: 800, color: '#fff', letterSpacing: '-0.02em' }}>{value}</div>
        <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
      </div>
    </motion.div>
  )
}

function ScrollingTicker({ items }) {
  const [offset, setOffset] = useState(0)
  const ref = useRef(null)

  useEffect(() => {
    let animationFrameId
    const width = ref.current?.scrollWidth || 0
    const animate = () => {
      setOffset((prev) => (prev >= width ? -window.innerWidth : prev + 0.5))
      animationFrameId = requestAnimationFrame(animate)
    }
    animationFrameId = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(animationFrameId)
  }, [])

  return (
    <div style={{ overflow: 'hidden', width: '100%', position: 'relative', maskImage: 'linear-gradient(90deg, transparent, black 10%, black 90%, transparent)' }}>
      <div ref={ref} style={{ display: 'flex', gap: 32, whiteSpace: 'nowrap', transform: `translateX(${offset}px)`, willChange: 'transform' }}>
        {items.map((item, i) => (
          <span key={i} style={{ color: 'rgba(255,255,255,0.35)', fontSize: 13, fontWeight: 500, letterSpacing: '0.02em' }}>{item}</span>
        ))}
        {items.map((item, i) => (
          <span key={i + items.length} style={{ color: 'rgba(255,255,255,0.35)', fontSize: 13, fontWeight: 500, letterSpacing: '0.02em' }}>{item}</span>
        ))}
      </div>
    </div>
  )
}

function InputField({ label, type = 'text', value, onChange, onKeyDown, placeholder, autoFocus, autoComplete, rightSlot }) {
  const [focused, setFocused] = useState(false)
  return (
    <div style={{ marginBottom: 16 }}>
      <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: 'var(--muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</label>
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
          style={{
            width: '100%', padding: '12px 14px', background: 'var(--surface2)', border: '1px solid var(--border)',
            borderRadius: 10, color: 'var(--text)', fontSize: 14, outline: 'none',
            paddingRight: rightSlot ? 42 : 14, transition: 'border-color 0.2s, box-shadow 0.2s',
            boxSizing: 'border-box',
          }}
          onMouseEnter={(e) => !focused && (e.target.style.borderColor = 'var(--accent)')}
          onMouseLeave={(e) => !focused && (e.target.style.borderColor = 'var(--border)')}
        />
        <motion.div
          initial={false}
          animate={{ scaleX: focused ? 1 : 0, opacity: focused ? 1 : 0 }}
          transition={{ duration: 0.2 }}
          style={{
            position: 'absolute', bottom: 0, left: 0, right: 0, height: 2,
            borderRadius: '0 0 10px 10px',
            background: 'linear-gradient(90deg, var(--accent), var(--accent2))',
            transformOrigin: 'left',
          }}
        />
        {rightSlot && (
          <div style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', zIndex: 1 }}>
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
  const [isMobile, setIsMobile] = useState(false)
  const mousePos = useRef({ x: -999, y: -999 })

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768)
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  useEffect(() => {
    const onMouseMove = (e) => {
      mousePos.current = { x: e.clientX, y: e.clientY }
    }
    window.addEventListener('mousemove', onMouseMove)
    return () => window.removeEventListener('mousemove', onMouseMove)
  }, [])

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

  const handleGoogleClick = () => {
    console.log('Google OAuth coming soon')
  }

  const techKeywords = ['PPO', 'SHAP', 'Kafka', 'Kubernetes', 'FastAPI', 'PyTorch', 'Reinforcement Learning', 'Carbon-Aware', 'Multi-Cloud']

  return (
    <div style={{
      minHeight: '100vh', display: 'flex', background: 'var(--bg)',
      position: 'relative', overflow: 'hidden',
    }}>
      {!isMobile && (
        <motion.aside
          initial={{ opacity: 0, x: -40 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          style={{
            width: '55%', height: '100vh', position: 'fixed', left: 0, top: 0, zIndex: 1,
            background: 'linear-gradient(135deg, #0a0f1e 0%, #0d1b2e 50%, #0a0f1e 100%)',
            display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center',
            padding: '60px 80px', overflow: 'hidden',
          }}
        >
          <ParticleCanvas mousePos={mousePos} />

          <motion.div
            style={{ position: 'relative', zIndex: 2, maxWidth: 520, width: '100%' }}
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
          >
            <motion.div
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12, marginBottom: 24 }}
              animate={{ boxShadow: ['0 0 30px rgba(99,102,241,0.2)', '0 0 60px rgba(99,102,241,0.4)', '0 0 30px rgba(99,102,241,0.2)'] }}
              transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
            >
              <motion.div
                animate={{ rotate: [0, 5, -5, 0] }}
                transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
                style={{
                  width: 56, height: 56, borderRadius: 14,
                  background: 'linear-gradient(135deg, var(--accent), var(--accent2))',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}
              >
                <Zap size={28} color="#fff" />
              </motion.div>
              <motion.h1
                style={{ fontSize: 32, fontWeight: 900, letterSpacing: '-0.03em', background: 'linear-gradient(135deg, #fff, #e0e7ff)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}
                animate={{ textShadow: ['0 0 0px transparent', '0 0 30px rgba(99,102,241,0.5)', '0 0 0px transparent'] }}
                transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
              >
                Velox
              </motion.h1>
            </motion.div>

            <motion.p
              style={{ textAlign: 'center', color: 'rgba(255,255,255,0.6)', fontSize: 16, fontWeight: 400, letterSpacing: '0.01em', marginBottom: 48 }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.35, duration: 0.5 }}
            >
              Multi-Cloud Workload Scheduler
            </motion.p>

            <motion.div
              style={{ display: 'flex', gap: 16, flexWrap: 'wrap', justifyContent: 'center', marginBottom: 48 }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5, duration: 0.5 }}
            >
              <StatCard value="2M+" label="Training Steps" icon={Zap} color="#6366f1" delay={0.55} />
              <StatCard value="3 Clouds" label="AWS · Azure · GCP" icon={Chrome} color="#10b981" delay={0.62} />
              <StatCard value="~35ms" label="Avg Inference" icon={Loader} color="#f59e0b" delay={0.69} />
            </motion.div>

            <motion.div
              style={{ marginBottom: 32 }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8, duration: 0.5 }}
            >
              <ScrollingTicker items={techKeywords} />
            </motion.div>

            <motion.p
              style={{ textAlign: 'center', color: 'rgba(255,255,255,0.25)', fontSize: 12, fontWeight: 500, letterSpacing: '0.1em', textTransform: 'uppercase' }}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1, duration: 0.5 }}
            >
              Built with Reinforcement Learning
            </motion.p>
          </motion.div>
        </motion.aside>
      )}

      <motion.main
        initial={{ opacity: 0, x: isMobile ? 0 : 40 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1], delay: isMobile ? 0 : 0.1 }}
        style={{
          width: isMobile ? '100%' : '45%', minHeight: '100vh',
          marginLeft: isMobile ? 0 : '55%', position: 'relative', zIndex: 2,
          background: 'var(--surface)', borderLeft: isMobile ? 'none' : '1px solid var(--border)',
          display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center',
          padding: '40px 60px', boxSizing: 'border-box',
        }}
      >
        <motion.div
          style={{ width: '100%', maxWidth: 400, position: 'relative' }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          <motion.div
            style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 32 }}
            whileHover={{ scale: 1.02 }}
          >
            <motion.div
              animate={{ boxShadow: ['0 4px 20px rgba(99,102,241,0.3)', '0 4px 30px rgba(99,102,241,0.5)', '0 4px 20px rgba(99,102,241,0.3)'] }}
              transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
              style={{
                width: 36, height: 36, borderRadius: 10,
                background: 'linear-gradient(135deg, var(--accent), var(--accent2))',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}
            >
              <Zap size={18} color="#fff" />
            </motion.div>
            <span style={{ fontWeight: 700, fontSize: 16, letterSpacing: '-0.02em' }}>Velox</span>
          </motion.div>

          {/* Tab switcher */}
          <div style={{
            display: 'flex', background: 'var(--surface2)',
            border: '1px solid var(--border)', borderRadius: 10,
            padding: 4, marginBottom: 24, gap: 4, position: 'relative',
          }}>
            {[{ id: 'signin', label: 'Sign In', Icon: LogIn }, { id: 'signup', label: 'Sign Up', Icon: UserPlus }].map(({ id, label, Icon }) => (
              <button
                key={id}
                type="button"
                onClick={() => switchMode(id)}
                disabled={loading}
                style={{
                  flex: 1, padding: '10px 0', borderRadius: 8,
                  fontWeight: 600, fontSize: 13,
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                  background: mode === id ? 'linear-gradient(135deg, var(--accent), var(--accent2))' : 'transparent',
                  color: mode === id ? '#fff' : 'var(--muted)',
                  border: 'none', cursor: loading ? 'not-allowed' : 'pointer',
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
          <div className="card" style={{ padding: '28px 28px 24px', boxShadow: '0 24px 64px rgba(0,0,0,0.35), 0 0 0 1px rgba(99,102,241,0.06)', borderColor: 'rgba(99,102,241,0.1)', borderRadius: 16 }}>
            <motion.div
              key={mode}
              initial={{ opacity: 0, x: mode === 'signin' ? -16 : 16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: mode === 'signin' ? 16 : -16 }}
              transition={{ duration: 0.22, ease: 'easeOut' }}
            >
              <div style={{ marginBottom: 24 }}>
                <div style={{ fontWeight: 700, fontSize: 17, marginBottom: 6 }}>
                  {mode === 'signin' ? 'Welcome back' : 'Create account'}
                </div>
                <div style={{ color: 'var(--muted)', fontSize: 13, lineHeight: 1.5 }}>
                  {mode === 'signin' ? 'Sign in with your credentials' : 'Create a new account to access Velox'}
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
                    style={{ background: 'none', color: 'var(--muted)', padding: 0, border: 'none', borderRadius: 0, display: 'flex', cursor: 'pointer' }}
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
                      style={{ background: 'none', color: 'var(--muted)', padding: 0, border: 'none', borderRadius: 0, display: 'flex', cursor: 'pointer' }}
                    >
                      {showConfirm ? <EyeOff size={15} /> : <Eye size={15} />}
                    </button>
                  }
                />
              )}

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
                  width: '100%', padding: '14px 0',
                  background: loading ? 'var(--surface2)' : 'linear-gradient(135deg, var(--accent), var(--accent2))',
                  color: loading ? 'var(--muted)' : '#fff',
                  fontWeight: 700, fontSize: 14,
                  border: loading ? '1px solid var(--border)' : 'none',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                  letterSpacing: '0.02em', borderRadius: 10,
                  position: 'relative', overflow: 'hidden', cursor: loading ? 'not-allowed' : 'pointer',
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
            </motion.div>
          </div>

          {/* Google OAuth button */}
          <motion.button
            type="button"
            onClick={handleGoogleClick}
            disabled={loading}
            whileHover={{ background: 'var(--surface2)', borderColor: 'var(--accent)' }}
            whileTap={{ scale: 0.98 }}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.4 }}
            style={{
              width: '100%', padding: '12px 0', marginTop: 16,
              background: 'var(--surface)', border: '1px solid var(--border)',
              color: 'var(--text)', fontWeight: 600, fontSize: 13,
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              borderRadius: 10, cursor: loading ? 'not-allowed' : 'pointer',
            }}
          >
            <Chrome size={15} />
            Continue with Google
          </motion.button>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6, duration: 0.4 }}
            style={{ textAlign: 'center', marginTop: 28, fontSize: 11, color: 'var(--muted)', lineHeight: 1.6 }}
          >
            Velox · Multi-Cloud Workload Scheduler
            <br />
            PPO · SHAP · Kafka · Kubernetes
          </motion.div>
        </motion.div>
      </motion.main>
    </div>
  )
}