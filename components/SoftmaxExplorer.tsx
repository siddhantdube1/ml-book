'use client'

import { useMemo, useRef, useState } from 'react'
import { softmax } from '@/lib/multinomial'

const W = 640
const H = 380
const PAD = 30
const Z_MIN = -4
const Z_MAX = 4

const CLASS_COLORS = ['#3c5a8c', '#c7522a', '#5d8a3a'] as const

// Layout
const TITLE_Y = 22
const BAR_AREA_TOP = 40
const BAR_AREA_HEIGHT = 180
const BAR_CENTER_Y = BAR_AREA_TOP + BAR_AREA_HEIGHT / 2 // 130, z = 0
const BAR_AREA_BOTTOM = BAR_AREA_TOP + BAR_AREA_HEIGHT // 220
const Z_LABEL_Y = BAR_AREA_BOTTOM + 22 // 242
const ARROW_Y = 272
const PROB_LABEL_Y = 298
const PROB_BAR_Y = 308
const PROB_BAR_HEIGHT = 36

export default function SoftmaxExplorer() {
  const [z, setZ] = useState<[number, number, number]>([1.4, 0.4, -0.8])
  const [dragging, setDragging] = useState<number | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  const probs = useMemo(() => softmax(z), [z])

  const barX = (i: number) => PAD + ((i + 0.5) / 3) * (W - 2 * PAD)
  const zToY = (zVal: number) =>
    BAR_CENTER_Y - (zVal / Z_MAX) * (BAR_AREA_HEIGHT / 2)

  function setZi(i: number, val: number) {
    const clamped = Math.max(Z_MIN, Math.min(Z_MAX, val))
    setZ((prev) => {
      const next: [number, number, number] = [prev[0], prev[1], prev[2]]
      next[i] = clamped
      return next
    })
  }

  function onPointerMove(e: React.PointerEvent<SVGSVGElement>) {
    if (dragging === null) return
    const svg = svgRef.current
    if (!svg) return
    const pt = svg.createSVGPoint()
    pt.x = e.clientX
    pt.y = e.clientY
    const { y } = pt.matrixTransform(svg.getScreenCTM()!.inverse())
    const zVal = ((BAR_CENTER_Y - y) * Z_MAX) / (BAR_AREA_HEIGHT / 2)
    setZi(dragging, zVal)
  }

  const probSegments = useMemo(() => {
    const left = PAD
    const totalW = W - 2 * PAD
    let x = left
    return probs.map((p, i) => {
      const segW = p * totalW
      const seg = { x, w: segW, color: CLASS_COLORS[i] }
      x += segW
      return seg
    })
  }, [probs])

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        {z.map((zi, i) => (
          <label key={i} className="flex items-center gap-2">
            <span className="text-ink-muted">
              <span style={{ color: CLASS_COLORS[i] }}>●</span> z
              <sub>{i + 1}</sub> ={' '}
              <span className="font-mono">{zi.toFixed(2)}</span>
            </span>
            <input
              type="range"
              min={Z_MIN}
              max={Z_MAX}
              step={0.05}
              value={zi}
              onChange={(e) => setZi(i, parseFloat(e.target.value))}
              className="w-28"
            />
          </label>
        ))}
        <button
          onClick={() => setZ([0, 0, 0])}
          className="px-3 py-1 border border-rule rounded hover:bg-paper-soft transition-colors"
        >
          All zero
        </button>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="w-full h-auto block select-none"
          onPointerMove={onPointerMove}
          onPointerUp={(e) => {
            setDragging(null)
            e.currentTarget.releasePointerCapture(e.pointerId)
          }}
        >
          {/* Top label */}
          <text
            x={PAD}
            y={TITLE_Y}
            fontSize={12}
            fill="var(--ink-muted)"
            fontFamily="var(--font-sans, sans-serif)"
          >
            input scores (logits)
          </text>

          {/* Z baseline */}
          <line
            x1={PAD}
            y1={BAR_CENTER_Y}
            x2={W - PAD}
            y2={BAR_CENTER_Y}
            stroke="var(--rule)"
            strokeWidth={1}
            strokeDasharray="3,3"
          />
          <text
            x={W - PAD - 4}
            y={BAR_CENTER_Y - 4}
            fontSize={10}
            fill="var(--ink-muted)"
            textAnchor="end"
            fontFamily="var(--font-mono, monospace)"
          >
            z = 0
          </text>

          {/* Z-axis tick marks */}
          {[-4, -2, 2, 4].map((tick) => (
            <g key={tick}>
              <line
                x1={PAD}
                y1={zToY(tick)}
                x2={PAD + 4}
                y2={zToY(tick)}
                stroke="var(--ink-muted)"
                strokeWidth={1}
              />
              <text
                x={PAD - 4}
                y={zToY(tick) + 3}
                fontSize={9}
                fill="var(--ink-muted)"
                textAnchor="end"
                fontFamily="var(--font-mono, monospace)"
              >
                {tick}
              </text>
            </g>
          ))}

          {/* Three z-bars */}
          {z.map((zi, i) => {
            const x = barX(i)
            const yHandle = zToY(zi)
            const yTop = Math.min(yHandle, BAR_CENTER_Y)
            const yBot = Math.max(yHandle, BAR_CENTER_Y)
            return (
              <g key={i}>
                <rect
                  x={x - 22}
                  y={yTop}
                  width={44}
                  height={yBot - yTop}
                  fill={CLASS_COLORS[i]}
                  opacity={0.55}
                />
                <line
                  x1={x - 24}
                  y1={yHandle}
                  x2={x + 24}
                  y2={yHandle}
                  stroke={CLASS_COLORS[i]}
                  strokeWidth={2}
                />
                <circle
                  cx={x}
                  cy={yHandle}
                  r={8}
                  fill="var(--paper)"
                  stroke={CLASS_COLORS[i]}
                  strokeWidth={2}
                  className="cursor-ns-resize"
                  onPointerDown={(e) => {
                    setDragging(i)
                    e.currentTarget.setPointerCapture(e.pointerId)
                    e.stopPropagation()
                  }}
                />
                <text
                  x={x}
                  y={Z_LABEL_Y}
                  fontSize={12}
                  fill="var(--ink)"
                  textAnchor="middle"
                  fontFamily="var(--font-mono, monospace)"
                >
                  z{i + 1} = {zi.toFixed(2)}
                </text>
              </g>
            )
          })}

          {/* Softmax arrow */}
          <text
            x={W / 2}
            y={ARROW_Y}
            fontSize={13}
            fill="var(--ink-muted)"
            textAnchor="middle"
            fontStyle="italic"
            fontFamily="var(--font-sans, sans-serif)"
          >
            ↓ softmax ↓
          </text>

          {/* Probability bar label */}
          <text
            x={PAD}
            y={PROB_LABEL_Y}
            fontSize={12}
            fill="var(--ink-muted)"
            fontFamily="var(--font-sans, sans-serif)"
          >
            output probabilities (sum = 1)
          </text>

          {/* Probability bar segments */}
          {probSegments.map((seg, i) => (
            <g key={i}>
              <rect
                x={seg.x}
                y={PROB_BAR_Y}
                width={seg.w}
                height={PROB_BAR_HEIGHT}
                fill={seg.color}
                opacity={0.75}
              />
              {seg.w > 38 && (
                <text
                  x={seg.x + seg.w / 2}
                  y={PROB_BAR_Y + PROB_BAR_HEIGHT / 2 + 4}
                  fontSize={12}
                  fill="var(--paper)"
                  textAnchor="middle"
                  fontFamily="var(--font-mono, monospace)"
                >
                  {probs[i].toFixed(3)}
                </text>
              )}
            </g>
          ))}
          <rect
            x={PAD}
            y={PROB_BAR_Y}
            width={W - 2 * PAD}
            height={PROB_BAR_HEIGHT}
            fill="none"
            stroke="var(--rule)"
            strokeWidth={1}
          />
        </svg>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-3 font-mono text-xs text-ink-muted">
        {probs.map((p, i) => (
          <div key={i}>
            <span style={{ color: CLASS_COLORS[i] }}>●</span> p
            <sub>{i + 1}</sub> ={' '}
            <span className="text-ink">{p.toFixed(4)}</span>
          </div>
        ))}
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 8.1 — The softmax function turns any three real numbers — the
        "logits" — into three probabilities that always sum to one. Drag a
        bar (or use a slider) to change a logit, and watch the probabilities
        redistribute among the three classes.
      </figcaption>
    </figure>
  )
}
