import { useMemo, useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Activity, Zap, LogOut, User, Bell,
  LayoutDashboard, BarChart3, Cpu, ShieldCheck, Cloud,
} from 'lucide-react'

const ROLE_COLORS = {
  viewer: 'var(--muted)', user: 'var(--accent)',
  engineer: 'var(--green)', admin: 'var(--red)', executive: 'var(--accent2)',
}

function InlineSidebar({ activeSection, onNavigate, userInfo }) {
  const items = useMemo(() => {
    const base = [
      { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
      { id: 'metrics',   label: 'Metrics',   icon: BarChart3 },
      { id: 'decisions', label: 'Decisions', icon: Cpu },
    ]
    if (userInfo?.role === 'admin')     base.push({ id: 'admin',    label: 'Admin',    icon: ShieldCheck })
    if (userInfo?.role === 'executive') base.push({ id: 'snapshot', label: 'Snapshot', icon: Cloud })
    return base
  }, [userInfo])

  return (
    <motion.aside
      initial={{ x: -20, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
      style={{
        width: 220, borderRight: '1px solid var(--border)',
        background: 'var(--surface)', padding: '18px 14px', flexShrink: 0,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 22, padding: '0 6px' }}>
        <motion.div
          animate={{ boxShadow: ['0 0 8px rgba(99,102,241,0.3)', '0 0 20px rgba(99,102,241,0.6)', '0 0 8px rgba(99,102,241,0.3)'] }}
          transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
          style={{
            width: 34, height: 34, borderRadius: 10,
            background: 'linear-gradient(135deg, var(--accent), var(--accent2))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
        >
          <Zap size={16} color="#fff" />
        </motion.div>
        <div>
          <div style={{ fontWeight: 800, fontSize: 14, lineHeight: 1.1 }}>CloudOS</div>
          <div style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 700, marginTop: 2 }}>
            RL Platform
          </div>
        </div>
      </div>

      <div style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 700, margin: '0 6px 10px' }}>
        Navigation
      </div>

      <nav style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {items.map(({ id, label, icon: Icon }, i) => {
          const active = activeSection === id
          return (
            <motion.button
              key={id}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
              whileHover={{ backgroundColor: 'var(--surface2)' }}
              whileTap={{ scale: 0.97 }}
              onClick={() => onNavigate(id)}
              style={{
                display: 'flex', alignItems: 'center', gap: 10,
                width: '100%', padding: '10px 12px', borderRadius: 10,
                border: `1px solid ${active ? 'var(--accent2)' : 'var(--border)'}`,
                background: active ? 'var(--surface2)' : 'transparent',
                color: active ? 'var(--text)' : 'var(--muted)',
                fontSize: 13, fontWeight: active ? 700 : 600,
                cursor: 'pointer', textAlign: 'left', position: 'relative',
              }}
            >
              {active && (
                <motion.div
                  layoutId="sidebarActive"
                  style={{
                    position: 'absolute', left: 0, top: 4, bottom: 4,
                    width: 3, borderRadius: 2,
                    background: 'linear-gradient(180deg, var(--accent), var(--accent2))',
                  }}
                />
              )}
              <Icon size={14} /><span>{label}</span>
            </motion.button>
          )
        })}
      </nav>

      <motion.div
        animate={{ opacity: [0.7, 1, 0.7] }}
        transition={{ duration: 2, repeat: Infinity }}
        style={{ marginTop: 20, padding: '12px 10px', borderRadius: 10, background: 'var(--surface2)', border: '1px solid var(--border)' }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--muted)' }}>
          <motion.span
            animate={{ opacity: [1, 0.3, 1], scale: [1, 1.2, 1] }}
            transition={{ duration: 1.8, repeat: Infinity }}
            style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--green)', boxShadow: '0 0 8px var(--green)', display: 'inline-block' }}
          />
          <span>AI Scheduler Live</span>
        </div>
      </motion.div>
    </motion.aside>
  )
}

export default function Layout({ children, userInfo, onLogout, headerExtra, showSidebar = false }) {
  const [activeSection, setActiveSection] = useState('dashboard')
  const [scrolled, setScrolled] = useState(false)
  const [bellShake, setBellShake] = useState(false)
  const bellTimer = useRef(null)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  // Bell shakes every 30s subtly
  useEffect(() => {
    bellTimer.current = setInterval(() => {
      setBellShake(true)
      setTimeout(() => setBellShake(false), 600)
    }, 30000)
    return () => clearInterval(bellTimer.current)
  }, [])

  const handleNavigate = (sectionId) => {
    setActiveSection(sectionId)
    const el = document.getElementById(`section-${sectionId}`)
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <motion.header
        animate={{
          boxShadow: scrolled ? '0 1px 24px rgba(0,0,0,0.3)' : '0 1px 0px rgba(0,0,0,0)',
          borderBottomColor: scrolled ? 'var(--border2)' : 'var(--border)',
        }}
        transition={{ duration: 0.2 }}
        style={{
          background: 'var(--surface)', borderBottom: '1px solid var(--border)',
          padding: '0 24px 0 32px', height: 56,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          position: 'sticky', top: 0, zIndex: 100,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {!showSidebar ? (
            <>
              <motion.div
                whileHover={{ scale: 1.05 }}
                style={{
                  width: 32, height: 32, borderRadius: 8,
                  background: 'linear-gradient(135deg, var(--accent), var(--accent2))',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}
              >
                <Zap size={16} color="#fff" />
              </motion.div>
              <span style={{ fontWeight: 700, fontSize: 15 }}>CloudOS</span>
              <span style={{
                background: 'var(--surface2)', border: '1px solid var(--border)',
                padding: '1px 7px', borderRadius: 4, fontSize: 10,
                color: 'var(--accent)', fontWeight: 700, letterSpacing: '0.05em',
              }}>RL</span>
            </>
          ) : (
            <div style={{ fontSize: 12, color: 'var(--muted)', display: 'flex', alignItems: 'center', gap: 6 }}>
              <motion.span
                animate={{ opacity: [1, 0.3, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
                style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--green)', boxShadow: '0 0 6px var(--green)', display: 'inline-block' }}
              />
              <span>AI-Native Multi-Cloud Scheduler</span>
            </div>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {headerExtra}

          <motion.button
            animate={bellShake ? { rotate: [-8, 8, -6, 6, 0] } : { rotate: 0 }}
            transition={{ duration: 0.4 }}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            type="button"
            title="Notifications"
            style={{
              background: 'var(--surface2)', border: '1px solid var(--border)',
              color: 'var(--muted)', padding: '6px 8px', borderRadius: 8,
              display: 'flex', alignItems: 'center', cursor: 'pointer',
            }}
          >
            <Bell size={13} />
          </motion.button>

          {userInfo ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  background: 'var(--surface2)', border: '1px solid var(--border)',
                  borderRadius: 8, padding: '4px 10px', fontSize: 12,
                }}
              >
                <User size={11} color="var(--muted)" />
                <span style={{ color: 'var(--text2)' }}>{userInfo.username}</span>
                <span style={{
                  fontWeight: 700, fontSize: 10, textTransform: 'uppercase',
                  color: ROLE_COLORS[userInfo.role] || 'var(--muted)', letterSpacing: '0.06em',
                }}>
                  {userInfo.role}
                </span>
              </motion.div>

              <motion.button
                whileHover={{ scale: 1.04, color: 'var(--red)' }}
                whileTap={{ scale: 0.96 }}
                type="button"
                onClick={onLogout}
                title="Sign out"
                style={{
                  display: 'flex', alignItems: 'center', gap: 5,
                  padding: '5px 10px', background: 'var(--surface2)',
                  border: '1px solid var(--border)', color: 'var(--muted)',
                  fontSize: 11, fontWeight: 600, borderRadius: 8, cursor: 'pointer',
                }}
              >
                <LogOut size={11} />Sign out
              </motion.button>
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--muted)', fontSize: 12 }}>
              <Activity size={13} />
              <span>AI-Native Multi-Cloud Scheduler</span>
            </div>
          )}
        </div>
      </motion.header>

      <div style={{ display: 'flex', flex: 1 }}>
        {showSidebar && userInfo && (
          <InlineSidebar activeSection={activeSection} onNavigate={handleNavigate} userInfo={userInfo} />
        )}

        <main style={{
          flex: 1, padding: '28px 32px',
          maxWidth: showSidebar ? '100%' : 1280,
          margin: showSidebar ? 0 : '0 auto',
          width: '100%', overflowX: 'hidden',
        }}>
          <AnimatePresence mode="wait">
            <motion.div
              key={userInfo?.role || 'default'}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.25, ease: 'easeOut' }}
            >
              {children}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  )
}