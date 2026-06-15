'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { sigmoid } from '@/lib/logistic'

const W = 640
const H = 320
const PAD_X = 40
const PAD_Y = 28
const Z_MIN = -8
const Z_MAX = 8

export default function SigmoidExplorer() {
  const [z, setZ] = useState(0)
  const [showStep, setShowStep] = useState(false)

  const sx = (zVal: number) =>
    PAD_X + ((zVal - Z_MIN) / (Z_MAX - Z_MIN)) * (W - 2 * PAD_X)
  const sy = (yVal: number) =>
    H - PAD_Y - yVal * (H - 2 * PAD_Y)

  const sigmoidPath = useMemo(() => {
    const N = 200
    const pts: string[] = []
    for (let i = 0; i <= N; i++) {
      const zVal = Z_MIN + (i / N) * (Z_MAX - Z_MIN)
      pts.push(
        `${i === 0 ? 'M' : 'L'} ${sx(zVal).toFixed(2)} ${sy(sigmoid(zVal)).toFixed(2)}`,
      )
    }
    return pts.join(' ')
  }, [])

  const svgRef = useRef<SVGSVGElement>(null)
  const [dragging, setDragging] = useState(false)

  function handlePointerMove(e: React.PointerEvent<SVGSVGElement>) {
    if (!dragging) return
    const svg = svgRef.current
    if (!svg) return
    const pt = svg.createSVGPoint()
    pt.x = e.clientX
    pt.y = e.clientY
    const { x } = pt.matrixTransform(svg.getScreenCTM()!.inverse())
    const newZ =
      Z_MIN + Math.max(0, Math.min(1, (x - PAD_X) / (W - 2 * PAD_X))) *
        (Z_MAX - Z_MIN)
    setZ(newZ)
  }

  const sigZ = sigmoid(z)
  const stepZ = z > 0 ? 1 : z < 0 ? 0 : 0.5

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            z = <span className="font-mono">{z.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={Z_MIN}
            max={Z_MAX}
            step={0.05}
            value={z}
            onChange={(e) => setZ(parseFloat(e.target.value))}
            className="w-48"
          />
        </label>
        <label className="flex items-center gap-2 text-ink-muted">
          <input
            type="checkbox"
            checked={showStep}
            onChange={(e) => setShowStep(e.target.checked)}
          />
          Overlay step function
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="w-full h-auto block cursor-ew-resize select-none"
          onPointerDown={(e) => {
            setDragging(true)
            e.currentTarget.setPointerCapture(e.pointerId)
          }}
          onPointerUp={(e) => {
            setDragging(false)
            e.currentTarget.releasePointerCapture(e.pointerId)
          }}
          onPointerMove={handlePointerMove}
        >
          {/* axes */}
          <line
            x1={sx(Z_MIN)}
            y1={sy(0)}
            x2={sx(Z_MAX)}
            y2={sy(0)}
            stroke="var(--rule)"
            strokeWidth={1}
          />
          <line
            x1={sx(0)}
            y1={sy(0)}
            x2={sx(0)}
            y2={sy(1)}
            stroke="var(--rule)"
            strokeWidth={1}
            strokeDasharray="2,3"
          />

          {/* threshold line at y = 0.5 */}
          <line
            x1={sx(Z_MIN)}
            y1={sy(0.5)}
            x2={sx(Z_MAX)}
            y2={sy(0.5)}
            stroke="var(--rule)"
            strokeWidth={1}
            strokeDasharray="3,3"
          />
          <text
            x={sx(Z_MIN) + 4}
            y={sy(0.5) - 4}
            fontSize={10}
            fill="var(--ink-muted)"
            fontFamily="var(--font-sans, sans-serif)"
          >
            threshold = 0.5
          </text>

          {/* y = 0 and y = 1 reference */}
          <text
            x={sx(Z_MAX) - 4}
            y={sy(0) + 14}
            fontSize={10}
            fill="var(--ink-muted)"
            textAnchor="end"
          >
            σ → 0
          </text>
          <text
            x={sx(Z_MAX) - 4}
            y={sy(1) + 12}
            fontSize={10}
            fill="var(--ink-muted)"
            textAnchor="end"
          >
            σ → 1
          </text>

          {/* x-axis label */}
          <text
            x={sx(Z_MAX) - 4}
            y={sy(0) - 6}
            fontSize={11}
            fill="var(--ink-muted)"
            textAnchor="end"
            fontStyle="italic"
          >
            z
          </text>

          {/* step function overlay */}
          {showStep && (
            <g
              stroke="var(--ink-muted)"
              strokeWidth={1.5}
              strokeDasharray="4,3"
              fill="none"
            >
              <line x1={sx(Z_MIN)} y1={sy(0)} x2={sx(0)} y2={sy(0)} />
              <line x1={sx(0)} y1={sy(0)} x2={sx(0)} y2={sy(1)} />
              <line x1={sx(0)} y1={sy(1)} x2={sx(Z_MAX)} y2={sy(1)} />
            </g>
          )}

          {/* sigmoid curve */}
          <path
            d={sigmoidPath}
            fill="none"
            stroke="var(--ink)"
            strokeWidth={2}
          />

          {/* current z vertical line */}
          <line
            x1={sx(z)}
            y1={sy(0)}
            x2={sx(z)}
            y2={sy(sigZ)}
            stroke="var(--accent)"
            strokeWidth={1.5}
            strokeDasharray="2,2"
            opacity={0.6}
          />
          {/* horizontal line from sigmoid value to y-axis */}
          <line
            x1={sx(Z_MIN)}
            y1={sy(sigZ)}
            x2={sx(z)}
            y2={sy(sigZ)}
            stroke="var(--accent)"
            strokeWidth={1.5}
            strokeDasharray="2,2"
            opacity={0.6}
          />
          <text
            x={sx(Z_MIN) - 6}
            y={sy(sigZ) + 4}
            fontSize={11}
            fill="var(--accent)"
            textAnchor="end"
            fontFamily="var(--font-mono, monospace)"
          >
            {sigZ.toFixed(3)}
          </text>

          {/* current point on sigmoid */}
          <circle
            cx={sx(z)}
            cy={sy(sigZ)}
            r={5}
            fill="var(--accent)"
            stroke="var(--paper)"
            strokeWidth={1.5}
          />

          {/* tick for current z on x-axis */}
          <text
            x={sx(z)}
            y={sy(0) + 14}
            fontSize={10}
            fill="var(--accent)"
            textAnchor="middle"
            fontFamily="var(--font-mono, monospace)"
          >
            z = {z.toFixed(2)}
          </text>
        </svg>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>z = {z.toFixed(3)}</div>
        <div>σ(z) = {sigZ.toFixed(4)}</div>
        <div>
          predicted class ={' '}
          <span className="text-ink">{sigZ >= 0.5 ? '1' : '0'}</span>
          {showStep && (
            <span className="ml-2">(step = {stepZ})</span>
          )}
        </div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 7.1 — The sigmoid function σ(z) = 1 / (1 + e<sup>−z</sup>).
        Drag the line or use the slider to see how z maps to a probability.
        At z = 0, σ = 0.5 — exactly the decision threshold.
      </figcaption>
    </figure>
  )
}
