'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { createRng, gauss } from '@/lib/rng'
import type { Point } from '@/lib/datasets'
import { Tex } from './Tex'

const ACCENT = '#1d6d5e'
const MEAN_COLOR = '#c7522a'

const CX = 340
const CY = 180
const STD = 38
const N_POINTS = 60
const DRAG_RADIUS = 165

export default function UpdateStepDemo() {
  const [seed, setSeed] = useState(7)

  const points: Point[] = useMemo(() => {
    const rng = createRng(seed)
    const out: Point[] = []
    for (let i = 0; i < N_POINTS; i++) {
      out.push([gauss(rng, CX, STD), gauss(rng, CY, STD)])
    }
    return out
  }, [seed])

  const mean = useMemo<Point>(() => {
    let sx = 0
    let sy = 0
    for (const p of points) {
      sx += p[0]
      sy += p[1]
    }
    return [sx / points.length, sy / points.length]
  }, [points])

  const wcssMin = useMemo(() => {
    let total = 0
    for (const p of points) {
      const dx = p[0] - mean[0]
      const dy = p[1] - mean[1]
      total += dx * dx + dy * dy
    }
    return total
  }, [points, mean])

  // Concentric level rings at 1.5×, 2.5×, 4× the minimum WCSS.
  // From the algebra in §4: WCSS(r) = WCSS_min + n·r², so the radius at a
  // given WCSS multiplier c is r = sqrt((c - 1) · WCSS_min / n).
  const levelRings = useMemo(() => {
    const multipliers = [1.5, 2.5, 4]
    return multipliers.map((c) => ({
      multiplier: c,
      radius: Math.sqrt(((c - 1) * wcssMin) / points.length),
    }))
  }, [wcssMin, points.length])

  const [centroid, setCentroid] = useState<Point>([CX + 70, CY + 50])
  useEffect(() => {
    setCentroid([mean[0] + 75, mean[1] + 55])
  }, [mean])

  const svgRef = useRef<SVGSVGElement>(null)
  const dragging = useRef(false)

  const wcss = useMemo(() => {
    let total = 0
    for (const p of points) {
      const dx = p[0] - centroid[0]
      const dy = p[1] - centroid[1]
      total += dx * dx + dy * dy
    }
    return total
  }, [points, centroid])

  const wcssRatio = wcss / Math.max(wcssMin, 1)
  const atMinimum = wcssRatio < 1.02

  const handlePointer = (e: React.PointerEvent) => {
    if (!svgRef.current) return
    const svg = svgRef.current
    const pt = svg.createSVGPoint()
    pt.x = e.clientX
    pt.y = e.clientY
    const ctm = svg.getScreenCTM()
    if (!ctm) return
    const local = pt.matrixTransform(ctm.inverse())

    // Constrain to a disk around the cluster mean. This keeps the WCSS
    // numbers human-readable while still letting the reader explore the bowl.
    const dx = local.x - mean[0]
    const dy = local.y - mean[1]
    const dist = Math.hypot(dx, dy)
    let x = local.x
    let y = local.y
    if (dist > DRAG_RADIUS) {
      x = mean[0] + (dx / dist) * DRAG_RADIUS
      y = mean[1] + (dy / dist) * DRAG_RADIUS
    }
    setCentroid([x, y])
  }

  return (
    <figure className="not-prose my-10">
      <div className="rounded-lg border border-rule bg-paper p-4 sm:p-5">
        <svg
          ref={svgRef}
          viewBox="0 0 680 360"
          xmlns="http://www.w3.org/2000/svg"
          className="w-full block rounded-md cursor-grab active:cursor-grabbing"
          style={{
            background: 'color-mix(in srgb, var(--ink) 3%, var(--paper))',
            touchAction: 'none',
            aspectRatio: '680 / 360',
          }}
          onPointerDown={(e) => {
            dragging.current = true
            ;(e.target as Element).setPointerCapture?.(e.pointerId)
            handlePointer(e)
          }}
          onPointerMove={(e) => {
            if (dragging.current) handlePointer(e)
          }}
          onPointerUp={() => (dragging.current = false)}
          onPointerCancel={() => (dragging.current = false)}
        >
          {levelRings.map((r) => (
            <circle
              key={r.multiplier}
              cx={mean[0]}
              cy={mean[1]}
              r={r.radius}
              fill="none"
              stroke="var(--ink-faint)"
              strokeWidth={0.5}
              strokeDasharray="2 4"
              opacity={0.55}
            />
          ))}
          {levelRings.map((r) => (
            <text
              key={`label-${r.multiplier}`}
              x={mean[0]}
              y={mean[1] - r.radius - 2}
              fontFamily="var(--font-sans)"
              fontSize={9}
              fill="var(--ink-muted)"
              textAnchor="middle"
              opacity={0.6}
            >
              {r.multiplier}× min
            </text>
          ))}

          {points.map((p, i) => (
            <line
              key={`l-${i}`}
              x1={p[0]}
              y1={p[1]}
              x2={centroid[0]}
              y2={centroid[1]}
              stroke="var(--ink-faint)"
              strokeWidth={0.5}
              opacity={0.3}
            />
          ))}

          {points.map((p, i) => (
            <circle
              key={`p-${i}`}
              cx={p[0]}
              cy={p[1]}
              r={4.5}
              fill={ACCENT}
              opacity={0.8}
            />
          ))}

          <circle
            cx={mean[0]}
            cy={mean[1]}
            r={11}
            fill="none"
            stroke={MEAN_COLOR}
            strokeWidth={1.5}
            strokeDasharray="3 3"
            opacity={0.75}
          />
          <text
            x={mean[0] + 16}
            y={mean[1] + 4}
            fontFamily="var(--font-sans)"
            fontSize={11}
            fill={MEAN_COLOR}
            opacity={0.85}
          >
            mean
          </text>

          <g transform={`translate(${centroid[0]}, ${centroid[1]})`}>
            <circle
              r={13}
              fill={ACCENT}
              stroke="var(--paper)"
              strokeWidth={3}
            />
            <path
              d="M-5,0 L5,0 M0,-5 L0,5"
              stroke="var(--paper)"
              strokeWidth={2.5}
              strokeLinecap="round"
            />
          </g>
        </svg>

        <div className="grid grid-cols-3 gap-3 mt-3">
          <Stat
            label="WCSS at centroid"
            value={Math.round(wcss).toLocaleString()}
            highlight={atMinimum}
          />
          <Stat
            label="× minimum"
            value={wcssRatio < 100 ? `${wcssRatio.toFixed(2)}×` : `${wcssRatio.toFixed(0)}×`}
            highlight={atMinimum}
          />
          <Stat
            label="WCSS at the mean"
            value={Math.round(wcssMin).toLocaleString()}
            faint
          />
        </div>

        <div className="mt-3 flex items-center gap-3">
          <button
            onClick={() => setSeed(Math.floor(Math.random() * 100000))}
            className="font-sans text-xs uppercase tracking-wider px-3 py-1.5 rounded-md border border-rule text-ink-muted hover:text-ink hover:border-ink-faint transition-colors"
          >
            New cluster
          </button>
          <button
            onClick={() => setCentroid([mean[0], mean[1]])}
            className="font-sans text-xs uppercase tracking-wider px-3 py-1.5 rounded-md transition-colors"
            style={{ background: 'var(--accent)', color: 'var(--paper)' }}
          >
            Snap to mean
          </button>
          <span className="font-sans text-xs text-ink-faint ml-auto">
            Drag the centroid · dashed rings are WCSS contours
          </span>
        </div>
      </div>
      <figcaption className="font-sans text-sm text-ink-muted mt-3 text-center">
        Figure 18.2 — WCSS as a function of centroid position for a single
        cluster. The level rings are contours of the WCSS surface — concentric
        circles centred on the mean, where WCSS doubles roughly every{' '}
        <Tex>{String.raw`\sqrt{2}`}</Tex> in radius.
      </figcaption>
    </figure>
  )
}

function Stat({
  label,
  value,
  highlight,
  faint,
}: {
  label: string
  value: string
  highlight?: boolean
  faint?: boolean
}) {
  return (
    <div
      className="rounded-md px-3 py-2"
      style={{
        background: 'color-mix(in srgb, var(--ink) 4%, var(--paper))',
        outline: highlight ? '1.5px solid var(--accent)' : 'none',
        outlineOffset: '-1.5px',
        transition: 'outline-color 0.2s',
      }}
    >
      <div className="font-sans text-xs uppercase tracking-wider text-ink-muted">
        {label}
      </div>
      <div
        className={`font-sans tabular-nums text-lg ${
          faint ? 'text-ink-muted' : 'text-ink'
        }`}
        style={{ fontWeight: 500 }}
      >
        {value}
      </div>
    </div>
  )
}
