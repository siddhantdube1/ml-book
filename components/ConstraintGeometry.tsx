'use client'

import { useMemo, useRef, useState } from 'react'
import {
  LOSS_EIGS,
  LOSS_THETA,
  constrainedOptL1,
  constrainedOptL2,
  quadLoss,
} from '@/lib/regularisation'

const W = 600
const H = 420
const PAD = 30
const SCALE = 100 // pixels per math unit — equal for x and y so the L2 ball renders as a circle

const X_HALF = (W - 2 * PAD) / (2 * SCALE)
const Y_HALF = (H - 2 * PAD) / (2 * SCALE)
const X_MIN = -X_HALF
const X_MAX = X_HALF
const Y_MIN = -Y_HALF
const Y_MAX = Y_HALF

// Loss contour levels — chosen so several contours pass through the
// interesting region around w* for typical drag positions.
const CONTOUR_LEVELS = [0.05, 0.12, 0.22, 0.36, 0.55, 0.8, 1.1, 1.5]

export default function ConstraintGeometry() {
  const [kind, setKind] = useState<'L1' | 'L2'>('L1')
  const [t, setT] = useState(0.8)
  const [wStar, setWStar] = useState<[number, number]>([1.5, 0.6])
  const [dragging, setDragging] = useState(false)
  const svgRef = useRef<SVGSVGElement>(null)

  const sx = (x: number) => W / 2 + x * SCALE
  const sy = (y: number) => H / 2 - y * SCALE
  const invX = (px: number) => (px - W / 2) / SCALE
  const invY = (py: number) => -(py - H / 2) / SCALE

  const wT = useMemo(
    () => (kind === 'L1' ? constrainedOptL1(wStar, t) : constrainedOptL2(wStar, t)),
    [kind, t, wStar],
  )

  const contourPaths = useMemo(() => {
    const c = Math.cos(LOSS_THETA)
    const s = Math.sin(LOSS_THETA)
    return CONTOUR_LEVELS.map((L) => {
      const a = Math.sqrt((2 * L) / LOSS_EIGS[0])
      const b = Math.sqrt((2 * L) / LOSS_EIGS[1])
      const N = 80
      const parts: string[] = []
      for (let k = 0; k <= N; k++) {
        const theta = (k / N) * 2 * Math.PI
        const u1 = a * Math.cos(theta)
        const u2 = b * Math.sin(theta)
        // w − w* = R^T u, where R rotates by LOSS_THETA counter-clockwise
        const wx = wStar[0] + c * u1 + s * u2
        const wy = wStar[1] + -s * u1 + c * u2
        parts.push(`${k === 0 ? 'M' : 'L'} ${sx(wx).toFixed(2)} ${sy(wy).toFixed(2)}`)
      }
      parts.push('Z')
      return parts.join(' ')
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wStar])

  const constraintPath = useMemo(() => {
    if (kind === 'L2') {
      const N = 96
      const parts: string[] = []
      for (let k = 0; k <= N; k++) {
        const theta = (k / N) * 2 * Math.PI
        const wx = t * Math.cos(theta)
        const wy = t * Math.sin(theta)
        parts.push(`${k === 0 ? 'M' : 'L'} ${sx(wx).toFixed(2)} ${sy(wy).toFixed(2)}`)
      }
      parts.push('Z')
      return parts.join(' ')
    }
    return `M ${sx(t)} ${sy(0)} L ${sx(0)} ${sy(t)} L ${sx(-t)} ${sy(0)} L ${sx(0)} ${sy(-t)} Z`
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind, t])

  function onPointerMove(e: React.PointerEvent<SVGSVGElement>) {
    if (!dragging) return
    const svg = svgRef.current
    if (!svg) return
    const pt = svg.createSVGPoint()
    pt.x = e.clientX
    pt.y = e.clientY
    const { x, y } = pt.matrixTransform(svg.getScreenCTM()!.inverse())
    const wx = Math.max(X_MIN, Math.min(X_MAX, invX(x)))
    const wy = Math.max(Y_MIN, Math.min(Y_MAX, invY(y)))
    setWStar([wx, wy])
  }

  const epsZero = 0.04
  const zeros =
    (Math.abs(wT[0]) < epsZero ? 1 : 0) +
    (Math.abs(wT[1]) < epsZero ? 1 : 0)
  const wTLoss = quadLoss(wT, wStar)
  const binding =
    kind === 'L1'
      ? Math.abs(wStar[0]) + Math.abs(wStar[1]) > t
      : Math.hypot(wStar[0], wStar[1]) > t

  return (
    <figure className="my-10">
      <div className="flex flex-wrap gap-x-6 gap-y-3 items-center mb-4 font-sans text-sm">
        <div className="flex items-center gap-2">
          <span className="text-ink-muted">Penalty</span>
          <div className="inline-flex rounded border border-rule overflow-hidden">
            <button
              onClick={() => setKind('L1')}
              className="px-3 py-1 transition-colors"
              style={{
                background: kind === 'L1' ? 'var(--accent)' : 'transparent',
                color: kind === 'L1' ? 'var(--paper)' : 'var(--ink)',
              }}
            >
              L1
            </button>
            <button
              onClick={() => setKind('L2')}
              className="px-3 py-1 transition-colors border-l border-rule"
              style={{
                background: kind === 'L2' ? 'var(--accent)' : 'transparent',
                color: kind === 'L2' ? 'var(--paper)' : 'var(--ink)',
              }}
            >
              L2
            </button>
          </div>
        </div>
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Budget t = <span className="font-mono">{t.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.05}
            max={2.2}
            step={0.02}
            value={t}
            onChange={(e) => setT(parseFloat(e.target.value))}
            className="w-36"
          />
        </label>
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="w-full h-auto block select-none"
          onPointerMove={onPointerMove}
          onPointerUp={(e) => {
            setDragging(false)
            e.currentTarget.releasePointerCapture(e.pointerId)
          }}
        >
          {/* axes */}
          <line
            x1={sx(X_MIN)}
            y1={sy(0)}
            x2={sx(X_MAX)}
            y2={sy(0)}
            stroke="var(--rule)"
            strokeWidth={1}
          />
          <line
            x1={sx(0)}
            y1={sy(Y_MIN)}
            x2={sx(0)}
            y2={sy(Y_MAX)}
            stroke="var(--rule)"
            strokeWidth={1}
          />
          <text
            x={sx(X_MAX) - 4}
            y={sy(0) - 6}
            fontSize={11}
            fill="var(--ink-muted)"
            textAnchor="end"
            fontStyle="italic"
          >
            w₁
          </text>
          <text
            x={sx(0) + 8}
            y={sy(Y_MAX) + 12}
            fontSize={11}
            fill="var(--ink-muted)"
            fontStyle="italic"
          >
            w₂
          </text>

          {/* constraint region (filled, behind contours) */}
          <path
            d={constraintPath}
            fill="var(--accent)"
            opacity={0.12}
            stroke="var(--accent)"
            strokeWidth={1.5}
          />

          {/* loss contours */}
          {contourPaths.map((d, i) => (
            <path
              key={i}
              d={d}
              fill="none"
              stroke="var(--ink-muted)"
              strokeWidth={0.9}
              opacity={0.55}
            />
          ))}

          {/* dashed segment from w_t to w* */}
          <line
            x1={sx(wT[0])}
            y1={sy(wT[1])}
            x2={sx(wStar[0])}
            y2={sy(wStar[1])}
            stroke="var(--ink-muted)"
            strokeWidth={1}
            strokeDasharray="3,3"
            opacity={0.5}
          />

          {/* w* — unconstrained optimum (draggable) */}
          <circle
            cx={sx(wStar[0])}
            cy={sy(wStar[1])}
            r={14}
            fill="var(--ink)"
            opacity={0.12}
          />
          <circle
            cx={sx(wStar[0])}
            cy={sy(wStar[1])}
            r={9}
            fill="var(--paper)"
            stroke="var(--ink)"
            strokeWidth={2.5}
            className="cursor-grab"
            onPointerDown={(e) => {
              setDragging(true)
              e.currentTarget.setPointerCapture(e.pointerId)
              e.stopPropagation()
            }}
          />
          <text
            x={sx(wStar[0])}
            y={sy(wStar[1]) - 14}
            fontSize={11}
            fill="var(--ink)"
            textAnchor="middle"
            fontFamily="var(--font-sans, sans-serif)"
            pointerEvents="none"
          >
            w*
          </text>

          {/* w_t — constrained optimum */}
          <circle
            cx={sx(wT[0])}
            cy={sy(wT[1])}
            r={6}
            fill="var(--accent)"
            stroke="var(--paper)"
            strokeWidth={1.5}
          />
          <text
            x={sx(wT[0]) + 10}
            y={sy(wT[1]) + 4}
            fontSize={11}
            fill="var(--accent)"
            fontFamily="var(--font-sans, sans-serif)"
            pointerEvents="none"
          >
            wₜ
          </text>
        </svg>
      </div>

      <div className="grid grid-cols-4 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>
          w* = ({wStar[0].toFixed(2)}, {wStar[1].toFixed(2)})
        </div>
        <div>
          wₜ = ({wT[0].toFixed(3)}, {wT[1].toFixed(3)})
        </div>
        <div>L(wₜ) = {wTLoss.toFixed(3)}</div>
        <div>
          {binding
            ? `${zeros} / 2 coefficients = 0`
            : 'constraint not binding'}
        </div>
      </div>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 9.2 — Weight space in two dimensions. The dark circle is w*,
        the unconstrained optimum of a quadratic loss (drag it). The teal
        contour is the constraint region — an L1 diamond or an L2 ball of
        budget <span className="font-mono">t</span>. The teal dot is the
        constrained optimum, where the loss contour first touches the
        region. For L1, the touch is often at a corner of the diamond,
        sending a coefficient exactly to zero. For L2, it almost never is.
      </figcaption>
    </figure>
  )
}
