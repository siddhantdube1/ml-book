'use client'

import { useMemo, useState } from 'react'

// Illustrative scaling law shaped like the empirical ones (Kaplan et al.,
// Hoffmann et al. "Chinchilla"): loss falls as a power of compute toward an
// irreducible floor, and the compute-optimal split grows N and D together.
const E = 1.5 // irreducible loss
const A = 8.0
const ALPHA = 0.048

function loss(log10C: number): number {
  return E + A * Math.pow(10, -ALPHA * log10C)
}

// Chinchilla-ish compute-optimal split, C ≈ 6·N·D with ~20 tokens/param.
function optimal(log10C: number): { N: number; D: number } {
  const C = Math.pow(10, log10C)
  const N = Math.sqrt(C / 120)
  return { N, D: 20 * N }
}

function big(n: number): string {
  if (n >= 1e12) return (n / 1e12).toFixed(1) + 'T'
  if (n >= 1e9) return (n / 1e9).toFixed(0) + 'B'
  if (n >= 1e6) return (n / 1e6).toFixed(0) + 'M'
  return n.toExponential(1)
}

const W = 600
const H = 300
const PAD = 48
const XMIN = 18
const XMAX = 26

type Props = { caption?: string }

export default function ScalingLaws({ caption }: Props) {
  const [logC, setLogC] = useState(22)

  const yMin = loss(XMAX) - 0.1
  const yMax = loss(XMIN) + 0.1
  const sx = (x: number) => PAD + ((x - XMIN) / (XMAX - XMIN)) * (W - 2 * PAD)
  const sy = (y: number) => H - PAD - ((y - yMin) / (yMax - yMin)) * (H - 2 * PAD)

  const path = useMemo(() => {
    const pts: string[] = []
    for (let x = XMIN; x <= XMAX; x += 0.1) {
      pts.push(`${pts.length === 0 ? 'M' : 'L'} ${sx(x).toFixed(1)} ${sy(loss(x)).toFixed(1)}`)
    }
    return pts.join(' ')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const L = loss(logC)
  const { N, D } = optimal(logC)

  return (
    <figure className="my-10 not-prose">
      <div className="border border-rule rounded-lg bg-paper-soft p-4">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          {/* axes */}
          <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="var(--rule)" />
          <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="var(--rule)" />
          {[18, 20, 22, 24, 26].map((x) => (
            <text key={x} x={sx(x)} y={H - PAD + 16} textAnchor="middle" fontSize={10} className="font-mono" fill="var(--ink-faint)">
              10^{x}
            </text>
          ))}
          <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={11} className="font-sans" fill="var(--ink-muted)">
            training compute (FLOPs)
          </text>
          <text x={14} y={H / 2} textAnchor="middle" fontSize={11} className="font-sans" fill="var(--ink-muted)" transform={`rotate(-90 14 ${H / 2})`}>
            test loss
          </text>

          {/* irreducible floor */}
          <line x1={PAD} y1={sy(E)} x2={W - PAD} y2={sy(E)} stroke="var(--ink-faint)" strokeDasharray="3,3" opacity={0.6} />
          <text x={W - PAD} y={sy(E) - 4} textAnchor="end" fontSize={9} className="font-sans" fill="var(--ink-faint)">
            irreducible loss
          </text>

          {/* curve */}
          <path d={path} fill="none" stroke="var(--accent)" strokeWidth={2} />

          {/* marker */}
          <line x1={sx(logC)} y1={PAD} x2={sx(logC)} y2={H - PAD} stroke="var(--accent)" strokeWidth={1} opacity={0.3} />
          <circle cx={sx(logC)} cy={sy(L)} r={5} fill="var(--accent)" stroke="var(--paper)" strokeWidth={1.5} />
        </svg>
      </div>

      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 mt-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Compute <span className="font-mono">10^{logC.toFixed(1)}</span> FLOPs
          </span>
          <input
            type="range"
            min={XMIN}
            max={XMAX}
            step={0.1}
            value={logC}
            onChange={(e) => setLogC(parseFloat(e.target.value))}
            className="w-48"
          />
        </label>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-3 font-mono text-xs text-ink-muted">
        <div>
          <span className="text-ink-faint">predicted loss</span>
          <div className="text-ink text-sm">{L.toFixed(3)}</div>
        </div>
        <div>
          <span className="text-ink-faint">optimal params</span>
          <div className="text-ink text-sm">{big(N)}</div>
        </div>
        <div>
          <span className="text-ink-faint">optimal tokens</span>
          <div className="text-ink text-sm">{big(D)}</div>
        </div>
      </div>

      {caption && (
        <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
          {caption}
        </figcaption>
      )}
    </figure>
  )
}
