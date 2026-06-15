'use client'

import { useMemo, useState } from 'react'
import { expand, fitLogistic, ALL_TERMS, type PolyTerm } from '@/lib/features'
import { makeXOR, type Point } from '@/lib/tree'
import { makeCircles2 } from '@/lib/svm'

const W = 420
const H = 380
const PAD = 24
const GX = 48
const GY = 44

const CLASS = ['#3c5a8c', '#c7522a'] as const

function blend(p1: number): string {
  const r = 60 + (199 - 60) * p1
  const g = 90 + (82 - 90) * p1
  const b = 140 + (42 - 140) * p1
  const alpha = Math.abs(p1 - 0.5) * 2 * 0.45
  return `rgba(${r.toFixed(0)}, ${g.toFixed(0)}, ${b.toFixed(0)}, ${alpha.toFixed(3)})`
}

const LABELS: Record<PolyTerm, string> = {
  x1: 'x₁',
  x2: 'x₂',
  'x1^2': 'x₁²',
  'x2^2': 'x₂²',
  'x1*x2': 'x₁·x₂',
}

export default function PolynomialFeatures() {
  const [dataset, setDataset] = useState<'xor' | 'circles'>('xor')
  const [terms, setTerms] = useState<Record<PolyTerm, boolean>>({
    x1: true,
    x2: true,
    'x1^2': false,
    'x2^2': false,
    'x1*x2': false,
  })

  const data = useMemo<Point[]>(
    () => (dataset === 'xor' ? makeXOR(180, 3) : makeCircles2(180, 5, 0.12)),
    [dataset],
  )

  const RANGE = dataset === 'xor' ? 2.4 : 2.8
  const sx = (x: number) => PAD + ((x + RANGE) / (2 * RANGE)) * (W - 2 * PAD)
  const sy = (y: number) => H - PAD - ((y + RANGE) / (2 * RANGE)) * (H - 2 * PAD)

  const activeTerms = ALL_TERMS.filter((t) => terms[t])

  const { predict, acc } = useMemo(() => {
    if (activeTerms.length === 0) {
      return { predict: () => 0.5, acc: 0.5 }
    }
    const X = data.map((p) => expand([p.x[0], p.x[1]], activeTerms))
    const y = data.map((p) => p.y)
    const f = fitLogistic(X, y, 500, 0.6)
    const predict = (px: number, py: number) => f(expand([px, py], activeTerms))
    let ok = 0
    for (const p of data) if ((predict(p.x[0], p.x[1]) >= 0.5 ? 1 : 0) === p.y) ok++
    return { predict, acc: ok / data.length }
  }, [data, activeTerms.join(',')])

  const heatmap = useMemo(() => {
    const cells: { x: number; y: number; w: number; h: number; fill: string }[] = []
    const cw = (W - 2 * PAD) / GX
    const ch = (H - 2 * PAD) / GY
    for (let i = 0; i < GX; i++) {
      for (let j = 0; j < GY; j++) {
        const xD = -RANGE + ((i + 0.5) / GX) * 2 * RANGE
        const yD = RANGE - ((j + 0.5) / GY) * 2 * RANGE
        cells.push({ x: PAD + i * cw, y: PAD + j * ch, w: cw, h: ch, fill: blend(predict(xD, yD)) })
      }
    }
    return cells
  }, [predict, RANGE])

  const toggle = (t: PolyTerm) => setTerms((s) => ({ ...s, [t]: !s[t] }))

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-4 gap-y-3 items-center mb-3 font-sans text-sm">
        <div className="flex rounded border border-rule overflow-hidden">
          {(['xor', 'circles'] as const).map((d) => (
            <button key={d} onClick={() => setDataset(d)} className="px-3 py-1 transition-colors" style={{ background: dataset === d ? 'var(--accent)' : 'transparent', color: dataset === d ? 'var(--paper)' : 'var(--ink-muted)' }}>
              {d === 'xor' ? 'XOR' : 'circles'}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap gap-2">
          {ALL_TERMS.map((t) => (
            <button key={t} onClick={() => toggle(t)} className="px-2.5 py-1 border rounded font-mono transition-colors" style={{ borderColor: terms[t] ? 'var(--accent)' : 'var(--rule)', background: terms[t] ? 'var(--accent)' : 'transparent', color: terms[t] ? 'var(--paper)' : 'var(--ink-muted)' }}>
              {LABELS[t]}
            </button>
          ))}
        </div>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {heatmap.map((c, i) => (
            <rect key={i} x={c.x} y={c.y} width={c.w} height={c.h} fill={c.fill} shapeRendering="crispEdges" />
          ))}
          {data.map((p, i) => (
            <circle key={i} cx={sx(p.x[0])} cy={sy(p.x[1])} r={3} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={0.7} />
          ))}
          <text x={PAD} y={PAD - 8} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">
            logistic regression on {activeTerms.length === 0 ? 'no features' : activeTerms.map((t) => LABELS[t]).join(', ')}
          </text>
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>features = <span className="text-ink">{activeTerms.length}</span> column{activeTerms.length === 1 ? '' : 's'} + bias</div>
        <div>training accuracy = <span className="text-accent">{(acc * 100).toFixed(1)}%</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 17.2 — Making a linear model curve. Logistic regression draws
        only straight boundaries <em>in the columns it is given</em>, so on the
        raw features x₁, x₂ it is stuck near chance — no line separates the XOR
        checkerboard or the nested circles. Hand it the right manufactured
        column and the very same linear model bends: tick{' '}
        <span className="font-mono">x₁·x₂</span> for XOR, or{' '}
        <span className="font-mono">x₁²</span> and{' '}
        <span className="font-mono">x₂²</span> for the circles, and the boundary
        snaps into shape. This is the kernel trick of Chapter 15, done by hand —
        you build the lift yourself, one column at a time.
      </figcaption>
    </figure>
  )
}
