'use client'

import { useMemo, useState } from 'react'
import type { Point } from '@/lib/tree'
import { makeCircles2, trainKernelSVM, linearWeights } from '@/lib/svm'

const W = 660
const H = 360

// Left panel (2-D)
const LCX = 150
const LCY = 180
const LS = 46 // px per data unit

// Right panel (pseudo-3-D isometric)
const OX = 478
const OY = 250
const SXY = 27 // px per unit (horizontal)
const SZ = 25 // px per unit of height (z)
const EXT = 2.7 // XY half-extent for the plane / ground

const R2_BOUNDARY = 2.3 // lifted threshold; sqrt is the circle radius

const CLASS = ['#3c5a8c', '#c7522a'] as const

export default function KernelLift() {
  const [lift, setLift] = useState(1)

  const data = useMemo<Point[]>(() => makeCircles2(120, 5, 0.13), [])

  const failLine = useMemo(() => {
    const m = trainKernelSVM(data, { kind: 'linear' }, { C: 1, seed: 3 })
    const { w, b } = linearWeights(m)
    let ok = 0
    for (const p of data) if ((w[0] * p.x[0] + w[1] * p.x[1] + b >= 0 ? 1 : 0) === p.y) ok++
    return { w, b, acc: ok / data.length }
  }, [data])

  // 2-D left panel maps
  const lsx = (x: number) => LCX + x * LS
  const lsy = (y: number) => LCY - y * LS

  // isometric projection for the right panel
  const iso = (X: number, Y: number, Z: number): [number, number] => [
    OX + (X - Y) * SXY,
    OY + (X + Y) * SXY * 0.5 - Z * SZ,
  ]

  const planeZ = lift * R2_BOUNDARY
  const R = Math.sqrt(R2_BOUNDARY)

  // painter's order: back (small X+Y) first
  const order = useMemo(
    () => data.map((_, i) => i).sort((a, b) => data[a].x[0] + data[a].x[1] - (data[b].x[0] + data[b].x[1])),
    [data],
  )

  const planeCorners = [
    iso(EXT, EXT, planeZ),
    iso(EXT, -EXT, planeZ),
    iso(-EXT, -EXT, planeZ),
    iso(-EXT, EXT, planeZ),
  ]
  const groundCorners = [
    iso(EXT, EXT, 0),
    iso(EXT, -EXT, 0),
    iso(-EXT, -EXT, 0),
    iso(-EXT, EXT, 0),
  ]
  // boundary circle on the ground (the plane's shadow)
  const shadow = Array.from({ length: 48 }, (_, k) => {
    const a = (k / 48) * 2 * Math.PI
    return iso(R * Math.cos(a), R * Math.sin(a), 0)
  })

  const w = failLine.w
  const b = failLine.b
  const lineY = (x: number) => -(w[0] * x + b) / w[1]

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Lift z = <span className="font-mono">{lift.toFixed(2)}</span> · (x₁² + x₂²)</span>
          <input type="range" min={0} max={1} step={0.01} value={lift} onChange={(e) => setLift(parseFloat(e.target.value))} className="w-52" />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <defs>
            <clipPath id="kl-left">
              <rect x={4} y={28} width={296} height={H - 36} />
            </clipPath>
          </defs>

          {/* ── Left: 2-D rings, no line separates ── */}
          <text x={16} y={20} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">
            original 2-D: no line works
          </text>
          <g clipPath="url(#kl-left)">
            <line
              x1={lsx(-3)} y1={lsy(lineY(-3))} x2={lsx(3)} y2={lsy(lineY(3))}
              stroke="var(--ink-muted)" strokeWidth={1.6} strokeDasharray="3,3"
            />
            {/* the kernel boundary (plane's shadow) = a circle */}
            <circle cx={lsx(0)} cy={lsy(0)} r={R * LS} fill="none" stroke="var(--accent)" strokeWidth={2} opacity={lift} />
          </g>
          {data.map((p, i) => (
            <circle key={`l${i}`} cx={lsx(p.x[0])} cy={lsy(p.x[1])} r={3.2} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.6} />
          ))}
          <text x={16} y={H - 10} fontSize={10} fill="var(--ink-faint)" fontFamily="var(--font-mono, monospace)">
            best line: {(failLine.acc * 100).toFixed(0)}% correct
          </text>

          {/* divider */}
          <line x1={308} y1={20} x2={308} y2={H - 20} stroke="var(--rule)" strokeWidth={1} />

          {/* ── Right: lifted to 3-D, a plane separates ── */}
          <text x={324} y={20} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">
            lifted to 3-D: a plane separates
          </text>
          {/* ground reference */}
          <polygon points={groundCorners.map((c) => c.join(',')).join(' ')} fill="var(--ink-faint)" opacity={0.06} stroke="var(--rule)" strokeWidth={0.75} />
          <polygon points={shadow.map((c) => c.join(',')).join(' ')} fill="none" stroke="var(--accent)" strokeWidth={1.5} opacity={0.5 * lift} />

          {/* stems + points, back to front */}
          {order.map((i) => {
            const p = data[i]
            const z = lift * (p.x[0] ** 2 + p.x[1] ** 2)
            const [gx, gy] = iso(p.x[0], p.x[1], 0)
            const [px, py] = iso(p.x[0], p.x[1], z)
            return (
              <g key={`r${i}`}>
                <line x1={gx} y1={gy} x2={px} y2={py} stroke={CLASS[p.y]} strokeWidth={0.6} opacity={0.22} />
                <circle cx={px} cy={py} r={3.2} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.6} />
              </g>
            )
          })}

          {/* separating plane (translucent, on top so above/below reads) */}
          <polygon points={planeCorners.map((c) => c.join(',')).join(' ')} fill="var(--accent)" opacity={0.16} stroke="var(--accent)" strokeWidth={1} strokeOpacity={0.5} />
        </svg>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 15.4 — The kernel trick, made literal. On the left, one class
        (blue) sits inside the other (orange): no straight line can separate
        them — the best one barely beats a coin. Lift each point to a third
        dimension <span className="font-mono">z = x₁² + x₂²</span> and the inner
        ring sinks into a bowl while the outer ring rides up its sides, until a
        single flat <span className="text-accent">plane</span> slices cleanly
        between them. Drag the lift down and watch them collapse back into the
        inseparable rings. The plane&rsquo;s shadow on the floor — and back in
        the 2-D panel — is a <span className="text-accent">circle</span>: a
        non-linear boundary that is merely a linear one seen from a higher
        dimension.
      </figcaption>
    </figure>
  )
}
