'use client'

import { useMemo, useRef, useState } from 'react'
import { makeBlobs2, gini, type Point } from '@/lib/tree'

const W = 560
const SH = 300 // scatter height
const GH = 110 // gain-strip height
const H = SH + GH + 8
const PAD = 26
const X_MIN = -3.2
const X_MAX = 3.2
const Y_MIN = -2.6
const Y_MAX = 2.6

const CLASS = ['#3c5a8c', '#c7522a'] as const
const CLASS_RGB = [
  [60, 90, 140],
  [199, 82, 42],
] as const

function countsOf(pts: Point[]): [number, number] {
  let a = 0
  let b = 0
  for (const p of pts) {
    if (p.y === 0) a++
    else b++
  }
  return [a, b]
}

export default function SplitFinder() {
  const [feature, setFeature] = useState<0 | 1>(0)
  const [threshold, setThreshold] = useState(0)
  const [dragging, setDragging] = useState(false)
  const svgRef = useRef<SVGSVGElement>(null)

  const data = useMemo(() => makeBlobs2(50, 5, 2), [])

  const lo = feature === 0 ? X_MIN : Y_MIN
  const hi = feature === 0 ? X_MAX : Y_MAX

  // scatter transforms
  const sx = (x: number) => PAD + ((x - X_MIN) / (X_MAX - X_MIN)) * (W - 2 * PAD)
  const sy = (y: number) => SH - PAD - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (SH - 2 * PAD)

  // split position in scatter
  const splitPos =
    feature === 0
      ? { x1: sx(threshold), y1: PAD, x2: sx(threshold), y2: SH - PAD }
      : { x1: PAD, y1: sy(threshold), x2: W - PAD, y2: sy(threshold) }

  const left = data.filter((p) => p.x[feature] < threshold)
  const right = data.filter((p) => p.x[feature] >= threshold)
  const cL = countsOf(left)
  const cR = countsOf(right)
  const cP = countsOf(data)
  const n = data.length
  const gL = gini(cL)
  const gR = gini(cR)
  const gP = gini(cP)
  const weighted = (left.length / n) * gL + (right.length / n) * gR
  const gain = gP - weighted

  // gain-vs-threshold curve
  const { gainPath, bestT, maxGain } = useMemo(() => {
    const N = 120
    const pts: [number, number][] = []
    let bestT = lo
    let maxGain = -1
    for (let i = 0; i <= N; i++) {
      const t = lo + (i / N) * (hi - lo)
      const l = data.filter((p) => p.x[feature] < t)
      const r = data.filter((p) => p.x[feature] >= t)
      const w =
        (l.length / n) * gini(countsOf(l)) + (r.length / n) * gini(countsOf(r))
      const g = gP - w
      pts.push([t, g])
      if (g > maxGain) {
        maxGain = g
        bestT = t
      }
    }
    const gMaxLocal = Math.max(0.02, maxGain * 1.15)
    const gxL = (t: number) => PAD + ((t - lo) / (hi - lo)) * (W - 2 * PAD)
    const gyL = (g: number) => H - PAD - (Math.max(0, g) / gMaxLocal) * (GH - 2 * PAD)
    const gainPath = pts
      .map(([t, g], i) => `${i === 0 ? 'M' : 'L'} ${gxL(t).toFixed(1)} ${gyL(g).toFixed(1)}`)
      .join(' ')
    return { gainPath, bestT, maxGain }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, feature])

  const gMax = Math.max(0.02, maxGain * 1.15)
  const gx = (t: number) => PAD + ((t - lo) / (hi - lo)) * (W - 2 * PAD)
  const gyBase = H - PAD

  // region shading: majority class per side
  const majL = cL[1] > cL[0] ? 1 : 0
  const majR = cR[1] > cR[0] ? 1 : 0

  function onPointerMove(e: React.PointerEvent<SVGSVGElement>) {
    if (!dragging) return
    const svg = svgRef.current
    if (!svg) return
    const pt = svg.createSVGPoint()
    pt.x = e.clientX
    pt.y = e.clientY
    const { x, y } = pt.matrixTransform(svg.getScreenCTM()!.inverse())
    if (feature === 0) {
      const t = X_MIN + ((x - PAD) / (W - 2 * PAD)) * (X_MAX - X_MIN)
      setThreshold(Math.max(X_MIN, Math.min(X_MAX, t)))
    } else {
      const t = Y_MIN + ((SH - PAD - y) / (SH - 2 * PAD)) * (Y_MAX - Y_MIN)
      setThreshold(Math.max(Y_MIN, Math.min(Y_MAX, t)))
    }
  }

  function switchFeature(f: 0 | 1) {
    setFeature(f)
    setThreshold(0)
  }

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <div className="flex items-center gap-2">
          <span className="text-ink-muted">Split on</span>
          <div className="inline-flex rounded border border-rule overflow-hidden">
            {([0, 1] as const).map((f) => (
              <button
                key={f}
                onClick={() => switchFeature(f)}
                className="px-3 py-1 transition-colors border-l border-rule first:border-l-0"
                style={{
                  background: feature === f ? 'var(--accent)' : 'transparent',
                  color: feature === f ? 'var(--paper)' : 'var(--ink)',
                }}
              >
                x{f === 0 ? '₁' : '₂'}
              </button>
            ))}
          </div>
        </div>
        <button
          onClick={() => setThreshold(bestT)}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors"
        >
          Snap to best split
        </button>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="w-full h-auto block select-none"
          style={{ touchAction: 'none' }}
          onPointerMove={onPointerMove}
          onPointerUp={(e) => {
            setDragging(false)
            e.currentTarget.releasePointerCapture(e.pointerId)
          }}
        >
          {/* region shading */}
          {feature === 0 ? (
            <>
              <rect x={PAD} y={PAD} width={sx(threshold) - PAD} height={SH - 2 * PAD}
                fill={`rgba(${CLASS_RGB[majL].join(',')},0.10)`} />
              <rect x={sx(threshold)} y={PAD} width={W - PAD - sx(threshold)} height={SH - 2 * PAD}
                fill={`rgba(${CLASS_RGB[majR].join(',')},0.10)`} />
            </>
          ) : (
            <>
              <rect x={PAD} y={sy(threshold)} width={W - 2 * PAD} height={SH - PAD - sy(threshold)}
                fill={`rgba(${CLASS_RGB[majL].join(',')},0.10)`} />
              <rect x={PAD} y={PAD} width={W - 2 * PAD} height={sy(threshold) - PAD}
                fill={`rgba(${CLASS_RGB[majR].join(',')},0.10)`} />
            </>
          )}

          {/* scatter frame */}
          <rect x={PAD} y={PAD} width={W - 2 * PAD} height={SH - 2 * PAD} fill="none" stroke="var(--rule)" strokeWidth={1} />

          {/* data points */}
          {data.map((p, i) => (
            <circle key={i} cx={sx(p.x[0])} cy={sy(p.x[1])} r={4} fill={CLASS[p.y]} stroke="var(--paper)" strokeWidth={1} />
          ))}

          {/* split line + grab handle */}
          <line x1={splitPos.x1} y1={splitPos.y1} x2={splitPos.x2} y2={splitPos.y2} stroke="var(--ink)" strokeWidth={2} />
          <rect
            x={feature === 0 ? sx(threshold) - 6 : PAD}
            y={feature === 0 ? PAD : sy(threshold) - 6}
            width={feature === 0 ? 12 : W - 2 * PAD}
            height={feature === 0 ? SH - 2 * PAD : 12}
            fill="transparent"
            className={feature === 0 ? 'cursor-ew-resize' : 'cursor-ns-resize'}
            onPointerDown={(e) => {
              setDragging(true)
              e.currentTarget.setPointerCapture(e.pointerId)
              e.stopPropagation()
            }}
          />

          {/* axis labels */}
          <text x={PAD + 2} y={PAD + 12} fontSize={10} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">
            drag the split — region tints show each side's majority class
          </text>

          {/* gain strip */}
          <line x1={PAD} y1={gyBase} x2={W - PAD} y2={gyBase} stroke="var(--rule)" strokeWidth={1} />
          <text x={PAD} y={SH + 14} fontSize={10} fill="var(--ink-muted)" fontStyle="italic" fontFamily="var(--font-sans, sans-serif)">
            information gain vs split position (x{feature === 0 ? '₁' : '₂'})
          </text>
          <path d={gainPath} fill="none" stroke="var(--accent)" strokeWidth={1.75} />
          {/* best-split marker */}
          <line x1={gx(bestT)} y1={SH + 18} x2={gx(bestT)} y2={gyBase} stroke="var(--accent)" strokeWidth={1} strokeDasharray="2,3" opacity={0.6} />
          <circle cx={gx(bestT)} cy={H - PAD - (Math.max(0, maxGain) / gMax) * (GH - 2 * PAD)} r={3.5} fill="var(--accent)" stroke="var(--paper)" strokeWidth={1} />
          {/* current threshold marker */}
          <line x1={gx(threshold)} y1={SH + 18} x2={gx(threshold)} y2={gyBase} stroke="var(--ink)" strokeWidth={1} opacity={0.4} />
        </svg>
      </div>

      <div className="grid grid-cols-4 gap-3 mt-3 font-mono text-xs text-ink-muted">
        <div>Gini left = <span className="text-ink">{gL.toFixed(3)}</span></div>
        <div>Gini right = <span className="text-ink">{gR.toFixed(3)}</span></div>
        <div>parent Gini = {gP.toFixed(3)}</div>
        <div>info gain = <span className="text-accent">{gain.toFixed(3)}</span></div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 12.1 — A single split. Drag the line, or switch which feature it
        cuts on, and watch the Gini impurity of each side and the resulting
        information gain. The curve below plots the gain for every threshold;
        the tree-building algorithm simply picks its peak — the split that
        makes the two sides as pure as possible.
      </figcaption>
    </figure>
  )
}
