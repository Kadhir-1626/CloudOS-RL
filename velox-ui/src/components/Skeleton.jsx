export function SkeletonLine({ width = '100%', height = 14, style = {} }) {
  return (
    <div
      className="skeleton"
      style={{ width, height, borderRadius: 6, ...style }}
    />
  )
}

export function SkeletonMetricsBar() {
  return (
    <div style={{ display: 'flex', gap: 16, marginBottom: 24 }}>
      {[1, 2, 3, 4].map(i => (
        <div key={i} className="card" style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 14 }}>
          <div className="skeleton" style={{ width: 42, height: 42, borderRadius: 10, flexShrink: 0 }} />
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <SkeletonLine width="60%" height={11} />
            <SkeletonLine width="40%" height={22} />
            <SkeletonLine width="70%" height={10} />
          </div>
        </div>
      ))}
    </div>
  )
}

export function SkeletonDecisionCard() {
  return (
    <div className="card" style={{ borderLeft: '3px solid var(--border2)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 20 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <SkeletonLine width={160} height={16} />
          <SkeletonLine width={120} height={11} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'flex-end' }}>
          <SkeletonLine width={60} height={11} />
          <SkeletonLine width={70} height={22} />
        </div>
      </div>
      {[1, 2, 3, 4].map(i => (
        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 0', borderBottom: '1px solid var(--border)' }}>
          <SkeletonLine width={100} height={12} />
          <SkeletonLine width={80}  height={12} />
        </div>
      ))}
      <div style={{ display: 'flex', gap: 12, marginTop: 16 }}>
        <div className="skeleton" style={{ flex: 1, height: 70, borderRadius: 8 }} />
        <div className="skeleton" style={{ flex: 1, height: 70, borderRadius: 8 }} />
      </div>
      <div className="skeleton" style={{ width: '100%', height: 38, borderRadius: 8, marginTop: 14 }} />
    </div>
  )
}

export function SkeletonTableRows({ count = 5 }) {
  return Array.from({ length: count }).map((_, i) => (
    <tr key={i}>
      {[80, 50, 90, 90, 80, 70, 60, 60, 60].map((w, j) => (
        <td key={j} style={{ padding: '11px 12px', borderBottom: '1px solid var(--border)' }}>
          <div className="skeleton" style={{ width: w, height: 12, borderRadius: 4 }} />
        </td>
      ))}
    </tr>
  ))
}