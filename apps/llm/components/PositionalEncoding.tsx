'use client'

import { useMemo, useState } from 'react'

// Sinusoidal positional encoding (Vaswani et al.), computed client-side.
// PE[pos][2i]   = sin(pos / 10000^(2i/d))
// PE[pos][2i+1] = cos(pos / 10000^(2i/d))
function sinusoidal(nPos: number, d: number): number[][] {
  const pe: number[][] = []
  for (let pos = 0; pos < nPos; pos++) {
    const row: number[] = []
    for (let i = 0; i < d; i++) {
      const k = Math.floor(i / 2)
      const denom = Math.pow(10000, (2 * k) / d)
      row.push(i % 2 === 0 ? Math.sin(pos / denom) : Math.cos(pos / denom))
    }
    pe.push(row)
  }
  return pe
}

function cellColour(v: number): string {
  // diverging: positive -> accent, negative -> terracotta
  const a = Math.abs(v)
  const c = v >= 0 ? 'var(--accent)' : '#c7522a'
  return `color-mix(in srgb, ${c} ${Math.round(a * 100)}%, transparent)`
}

const D = 32

type Props = { caption?: string }

export default function PositionalEncoding({ caption }: Props) {
  const [nPos, setNPos] = useState(24)
  const pe = useMemo(() => sinusoidal(nPos, D), [nPos])

  return (
    <figure className="my-10 not-prose">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Positions <span className="font-mono">{nPos}</span>
          </span>
          <input
            type="range"
            min={8}
            max={48}
            value={nPos}
            onChange={(e) => setNPos(parseInt(e.target.value))}
            className="w-40"
          />
        </label>
        <span className="text-ink-faint text-xs">{D} dimensions</span>
      </div>

      <div className="border border-rule rounded-lg bg-paper-soft p-4 overflow-x-auto">
        <div className="font-sans text-xs text-ink-faint mb-2">
          Each row is one position&rsquo;s encoding vector; each column a
          dimension. Low dimensions oscillate fast, high dimensions slow — so
          every position gets a unique fingerprint.
        </div>
        <svg
          viewBox={`0 0 ${D * 14 + 40} ${nPos * 11 + 20}`}
          className="h-auto block"
          style={{ width: '100%', minWidth: 360 }}
        >
          {pe.map((row, p) =>
            row.map((v, i) => (
              <rect
                key={`${p}-${i}`}
                x={40 + i * 14}
                y={6 + p * 11}
                width={13}
                height={10}
                fill={cellColour(v)}
              />
            )),
          )}
          {pe.map((_, p) =>
            p % 4 === 0 ? (
              <text
                key={`l${p}`}
                x={34}
                y={6 + p * 11 + 9}
                textAnchor="end"
                className="font-mono"
                fontSize={9}
                fill="var(--ink-faint)"
              >
                {p}
              </text>
            ) : null,
          )}
        </svg>
        <div className="mt-2 font-sans text-xs text-ink-faint">
          ← dimension &nbsp;·&nbsp; rows are positions, top = 0
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
