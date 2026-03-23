import { useState } from 'react'
import Layout      from './components/Layout'
import MetricsBar  from './components/MetricsBar'
import ScheduleForm from './components/ScheduleForm'
import DecisionCard from './components/DecisionCard'
import DecisionTable from './components/DecisionTable'

export default function App() {
  const [lastDecision, setLastDecision] = useState(null)

  return (
    <Layout>
      {/* Page title */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>
          AI Cloud Scheduler
        </h1>
        <p style={{ color: 'var(--muted)', fontSize: 13 }}>
          Real-time multi-cloud workload placement powered by Proximal Policy Optimization
        </p>
      </div>

      {/* Live metrics */}
      <MetricsBar />

      {/* Two-column: form + result */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 24 }}>
        <ScheduleForm onResult={setLastDecision} />

        {lastDecision ? (
          <DecisionCard key={lastDecision.decision_id} decision={lastDecision} />
        ) : (
          <div className="card" style={{
            display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            color: 'var(--muted)', gap: 12, minHeight: 320,
          }}>
            <div style={{
              width: 48, height: 48, borderRadius: '50%',
              background: 'var(--surface2)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 22,
            }}>
              ⚡
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontWeight: 600, color: 'var(--text)', marginBottom: 4 }}>
                No decision yet
              </div>
              <div style={{ fontSize: 12 }}>
                Submit a workload to see the RL placement decision
              </div>
            </div>
          </div>
        )}
      </div>

      {/* History table */}
      <DecisionTable />
    </Layout>
  )
}