import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LayoutDashboard, Send, ListChecks, Zap,
  BarChart2, Activity, DollarSign, Leaf,
  ChevronLeft, ChevronRight, Settings,
} from 'lucide-react'
import { useAuth } from '../auth/AuthContext'

const NAV_ITEMS = [
  { id: 'hero',        label: 'Overview',      icon: LayoutDashboard, roles: ['engineer','admin','executive','viewer','user'] },
  { id: 'metrics',     label: 'Performance',   icon: BarChart2,       roles: ['engineer','admin','executive'] },
  { id: 'schedule',    label: 'New Workload',  icon: Send,            roles: ['engineer','admin','user'] },
  { id: 'decisions',   label: 'Placements',    icon: ListChecks,      roles: ['engineer','admin','executive','viewer','user'] },
  { id: 'intelligence',label: 'Intelligence',  icon: Zap,             roles: ['engineer','admin'] },
  { id: 'kafka',       label: 'Stream',        icon: Activity,        roles: ['engineer','admin'] },
  { id: 'cost',        label: 'Cost',          icon: DollarSign,      roles: ['engineer','admin','executive'] },
  { id: 'carbon',      label: 'Carbon',        icon: Leaf,            roles: ['engineer','admin','executive'] },
  { id: 'control',     label: 'Control',       icon: Settings,        roles: ['admin'] },
]

const SECTION_MAP = {
  hero:         'section-hero',
  metrics:      'section-metrics',
  schedule:     'section-schedule',
  decisions:    'section-decisions',
  intelligence: 'section-intelligence',
  kafka:        'section-kafka',
  cost:         'section-cost',
  carbon:       'section-carbon',
  control:      'section-control',
}

function AIHeartbeat({ user }) {
  const role = (user?.role || 'viewer').toUpperCase()

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 6, padding: '10px 16px 14px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <motion.div
          animate={{ opacity: [1, 0.3, 1] }}
          transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
          style={{
            width: 8, height: 8, borderRadius: '50%',
            background: 'var(--green)', boxShadow: '0 0 8px var(--green)',
          }}
        />
        <span style={{ fontSize: 11, color: 'var(--green)', fontWeight: 700, letterSpacing: '0.05em' }}>
          AI Ready
        </span>
      </div>
      <div style={{ marginLeft: 16 }}>
        <span style={{ fontSize: 9, color: 'var(--accent2)', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
          {role}
        </span>
      </div>
    </div>
  )
}

export default function Sidebar({ activeSection, onNavigate }) {
  const { user } = useAuth()
  const [collapsed, setCollapsed] = useState(false)
  const role = user?.role || 'viewer'
  const visible = NAV_ITEMS.filter(item => item.roles.includes(role))

  const handleClick = (itemId) => {
    onNavigate?.(itemId)
    const sectionId = SECTION_MAP[itemId]
    if (sectionId) {
      const el = document.getElementById(sectionId)
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  return (
    <motion.aside
      animate={{ width: collapsed ? 56 : 220 }}
      transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
      style={{
        minHeight: '100vh',
        background: 'var(--surface)',
        borderRight: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        position: 'sticky',
        top: 0,
        flexShrink: 0,
        zIndex: 50,
        overflow: 'hidden',
      }}
    >
      {/* Logo area */}
      <div style={{
        height: 56, display: 'flex', alignItems: 'center',
        padding: collapsed ? '0 12px' : '0 16px',
        borderBottom: '1px solid var(--border)',
        justifyContent: collapsed ? 'center' : 'space-between',
        overflow: 'hidden',
      }}>
        <AnimatePresence mode="wait">
          {!collapsed && (
            <motion.div
              key="logo"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              transition={{ duration: 0.18 }}
              style={{ display: 'flex', alignItems: 'center', gap: 8 }}
            >
              <motion.div
                animate={{ boxShadow: ['0 0 8px rgba(99,102,241,0.3)', '0 0 18px rgba(99,102,241,0.6)', '0 0 8px rgba(99,102,241,0.3)'] }}
                transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
                style={{
                  width: 28, height: 28, borderRadius: 7,
                  background: 'linear-gradient(135deg, var(--accent), var(--accent2))',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}
              >
                <Zap size={14} color="#fff" />
              </motion.div>
              <div>
                <div style={{ fontWeight: 800, fontSize: 13, lineHeight: 1 }}>Velox</div>
                <div style={{ fontSize: 9, color: 'var(--accent)', fontWeight: 700, letterSpacing: '0.08em' }}>SCHEDULER</div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.93 }}
          onClick={() => setCollapsed(c => !c)}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          style={{
            background: 'var(--surface2)', border: '1px solid var(--border)',
            color: 'var(--muted)', padding: 4, borderRadius: 6,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          {collapsed ? <ChevronRight size={12} /> : <ChevronLeft size={12} />}
        </motion.button>
      </div>

      {/* Nav items */}
      <nav style={{ flex: 1, padding: '12px 8px', display: 'flex', flexDirection: 'column', gap: 2 }}>
        {visible.map((item, i) => {
          const isActive = activeSection === item.id
          return (
            <motion.button
              key={item.id}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.04, duration: 0.22 }}
              whileHover={{ backgroundColor: isActive ? undefined : 'var(--surface2)' }}
              whileTap={{ scale: 0.97 }}
              onClick={() => handleClick(item.id)}
              title={collapsed ? item.label : undefined}
              style={{
                display: 'flex', alignItems: 'center',
                gap: 10,
                padding: collapsed ? '9px 0' : '9px 12px',
                justifyContent: collapsed ? 'center' : 'flex-start',
                borderRadius: 8,
                fontWeight: isActive ? 700 : 500,
                fontSize: 13,
                color: isActive ? 'var(--accent)' : 'var(--text2)',
                background: isActive ? 'rgba(59,130,246,0.1)' : 'transparent',
                border: `1px solid ${isActive ? 'rgba(59,130,246,0.2)' : 'transparent'}`,
                width: '100%',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                position: 'relative',
                cursor: 'pointer',
              }}
            >
              {/* Active left accent bar */}
              {isActive && (
                <motion.div
                  layoutId="activeBar"
                  style={{
                    position: 'absolute', left: 0, top: 6, bottom: 6,
                    width: 3, borderRadius: 2,
                    background: 'linear-gradient(180deg, var(--accent), var(--accent2))',
                  }}
                  transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                />
              )}
              <item.icon size={15} style={{ flexShrink: 0 }} />
              <AnimatePresence>
                {!collapsed && (
                  <motion.span
                    initial={{ opacity: 0, width: 0 }}
                    animate={{ opacity: 1, width: 'auto' }}
                    exit={{ opacity: 0, width: 0 }}
                    transition={{ duration: 0.18 }}
                    style={{ overflow: 'hidden' }}
                  >
                    {item.label}
                  </motion.span>
                )}
              </AnimatePresence>
            </motion.button>
          )
        })}
      </nav>

      {/* AI heartbeat */}
      <AnimatePresence>
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            style={{ borderTop: '1px solid var(--border)' }}
          >
            <AIHeartbeat user={user} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* User info */}
      <AnimatePresence>
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            style={{ padding: '12px 16px', borderTop: '1px solid var(--border)', fontSize: 11 }}
          >
            <div style={{ color: 'var(--muted)', marginBottom: 3 }}>Signed in as</div>
            <div style={{ fontWeight: 700, color: 'var(--text)', fontSize: 12 }}>{user?.username}</div>
            <div style={{
              display: 'inline-block', marginTop: 4,
              background: 'rgba(99,102,241,0.12)', color: 'var(--accent2)',
              border: '1px solid rgba(99,102,241,0.25)',
              borderRadius: 12, padding: '1px 8px',
              fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em',
            }}>
              {user?.role}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.aside>
  )
}