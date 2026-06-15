'use client'

// Static figure: hinge vs logistic vs 0–1 loss against the functional margin.
// No state — a labelled comparison of the three loss curves.

const W = 460
const H = 280
const PAD_L = 46
const PAD_R = 16
const PAD_T = 20
const PAD_B = 44
const Z_MIN = -2
const Z_MAX = 3
const L_MAX = 3

const ZX = (z: number) => PAD_L + ((z - Z_MIN) / (Z_MAX - Z_MIN)) * (W - PAD_L - PAD_R)
const LY = (l: number) => H - PAD_B - (Math.min(l, L_MAX) / L_MAX) * (H - PAD_T - PAD_B)

function curve(f: (z: number) => number): string {
  const pts: string[] = []
  for (let k = 0; k <= 120; k++) {
    const z = Z_MIN + (k / 120) * (Z_MAX - Z_MIN)
    pts.push(`${k === 0 ? 'M' : 'L'} ${ZX(z).toFixed(1)} ${LY(f(z)).toFixed(1)}`)
  }
  return pts.join(' ')
}

const hinge = (z: number) => Math.max(0, 1 - z)
const logistic = (z: number) => Math.log(1 + Math.exp(-z)) / Math.LN2 // bits, so loss(0) = 1

export default function HingeLoss() {
  return (
    <figure className="my-10">
      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* axes */}
          <line x1={PAD_L} y1={LY(0)} x2={W - PAD_R} y2={LY(0)} stroke="var(--rule)" strokeWidth={1} />
          <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={LY(0)} stroke="var(--rule)" strokeWidth={1} />
          {/* margin = 1 reference */}
          <line x1={ZX(1)} y1={PAD_T} x2={ZX(1)} y2={LY(0)} stroke="var(--rule)" strokeWidth={1} strokeDasharray="2,3" opacity={0.7} />
          <text x={ZX(1)} y={PAD_T - 6} fontSize={9.5} fill="var(--ink-faint)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">z = 1</text>

          {/* 0–1 loss (step at z = 0) */}
          <path d={`M ${ZX(Z_MIN)} ${LY(1)} L ${ZX(0)} ${LY(1)} L ${ZX(0)} ${LY(0)} L ${ZX(Z_MAX)} ${LY(0)}`} fill="none" stroke="var(--ink-muted)" strokeWidth={1.5} strokeDasharray="4,3" />
          {/* logistic (bits) */}
          <path d={curve(logistic)} fill="none" stroke="#3c5a8c" strokeWidth={2} />
          {/* hinge */}
          <path d={curve(hinge)} fill="none" stroke="var(--accent)" strokeWidth={2.25} />

          {/* ticks */}
          {[-2, -1, 0, 1, 2, 3].map((z) => (
            <text key={z} x={ZX(z)} y={LY(0) + 15} fontSize={10} fill="var(--ink-muted)" textAnchor="middle" fontFamily="var(--font-mono, monospace)">{z}</text>
          ))}
          {[0, 1, 2, 3].map((l) => (
            <text key={l} x={PAD_L - 7} y={LY(l) + 3} fontSize={10} fill="var(--ink-muted)" textAnchor="end" fontFamily="var(--font-mono, monospace)">{l}</text>
          ))}
          <text x={(PAD_L + W - PAD_R) / 2} y={H - 8} fontSize={11} fill="var(--ink-muted)" textAnchor="middle" fontStyle="italic">functional margin z = y · (w · x + b)</text>
          <text x={PAD_L} y={PAD_T - 7} fontSize={11} fill="var(--ink-muted)" fontStyle="italic">loss</text>

          {/* legend */}
          <g transform={`translate(${W - PAD_R - 132}, ${PAD_T + 4})`}>
            <line x1={0} y1={0} x2={16} y2={0} stroke="var(--accent)" strokeWidth={2.25} />
            <text x={21} y={3.5} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">hinge (SVM)</text>
            <line x1={0} y1={14} x2={16} y2={14} stroke="#3c5a8c" strokeWidth={2} />
            <text x={21} y={17.5} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">logistic</text>
            <line x1={0} y1={28} x2={16} y2={28} stroke="var(--ink-muted)" strokeWidth={1.5} strokeDasharray="4,3" />
            <text x={21} y={31.5} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">0–1 (true error)</text>
          </g>
        </svg>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 15.3 — Three losses against the functional margin{' '}
        <span className="font-mono">z = y(w·x + b)</span>. Both the hinge loss
        (teal) and the logistic loss (blue, in bits) sit above the 0&ndash;1
        error they stand in for, so minimising either pushes the true error
        down. But the hinge has a <em>kink</em>: once a point is past the margin
        (<span className="font-mono">z ≥ 1</span>) its loss — and its gradient —
        is exactly zero, so it stops influencing the boundary entirely. That
        flat region is why the SVM&rsquo;s solution depends on only a few
        support vectors, while logistic regression&rsquo;s ever-decreasing tail
        keeps tugging on every point forever.
      </figcaption>
    </figure>
  )
}
