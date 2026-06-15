'use client'

import { useState } from 'react'
import { CITY_RATES } from '@/lib/features'

type Scheme = 'onehot' | 'ordinal' | 'target'

const W = 540
const ROW_H = 34
const TOP = 54
const NAME_X = 16
const MAT_X = 150
const H = TOP + CITY_RATES.length * ROW_H + 56

const NOTE: Record<Scheme, string> = {
  onehot: 'One column per category — no false ordering, but the matrix balloons with cardinality.',
  ordinal: 'One compact column — but it invents an order (is Quito really > Lagos?) the model will believe.',
  target: 'One column = the category’s mean label — compact and predictive, but it peeks at y: a leakage trap.',
}

export default function CategoricalEncoding() {
  const [scheme, setScheme] = useState<Scheme>('onehot')
  const k = CITY_RATES.length
  const cellW = (W - MAT_X - 20) / k

  return (
    <figure className="my-10">
      <div className="flex gap-2 mb-4 font-sans text-sm">
        {(['onehot', 'ordinal', 'target'] as const).map((s) => (
          <button key={s} onClick={() => setScheme(s)} className="px-3 py-1 border rounded transition-colors" style={{ borderColor: scheme === s ? 'var(--accent)' : 'var(--rule)', background: scheme === s ? 'var(--accent)' : 'transparent', color: scheme === s ? 'var(--paper)' : 'var(--ink-muted)' }}>
            {s === 'onehot' ? 'one-hot' : s === 'ordinal' ? 'ordinal' : 'target'}
          </button>
        ))}
      </div>

      <div className="border border-rule rounded-lg overflow-hidden bg-paper-soft">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block">
          <text x={NAME_X} y={28} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">category</text>
          {scheme === 'onehot' && CITY_RATES.map((c, j) => (
            <text key={`h${j}`} x={MAT_X + j * cellW + cellW / 2} y={28} fontSize={8.5} fill="var(--ink-faint)" textAnchor="middle" fontFamily="var(--font-mono, monospace)" transform={`rotate(-32 ${MAT_X + j * cellW + cellW / 2} 28)`}>is_{c.name}</text>
          ))}
          {scheme !== 'onehot' && (
            <text x={MAT_X} y={28} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">{scheme === 'ordinal' ? 'city_id' : 'city_rate'}</text>
          )}

          {CITY_RATES.map((city, i) => {
            const y = TOP + i * ROW_H
            return (
              <g key={city.name}>
                <text x={NAME_X} y={y + ROW_H / 2 + 4} fontSize={12} fill="var(--ink)" fontFamily="var(--font-mono, monospace)">{city.name}</text>

                {scheme === 'onehot' && CITY_RATES.map((_, j) => (
                  <g key={j}>
                    <rect x={MAT_X + j * cellW + 3} y={y + 4} width={cellW - 6} height={ROW_H - 8} rx={2} fill={i === j ? 'var(--accent)' : 'transparent'} stroke="var(--rule)" strokeWidth={0.75} opacity={i === j ? 0.85 : 1} />
                    <text x={MAT_X + j * cellW + cellW / 2} y={y + ROW_H / 2 + 4} fontSize={11} fill={i === j ? 'var(--paper)' : 'var(--ink-faint)'} textAnchor="middle" fontFamily="var(--font-mono, monospace)">{i === j ? 1 : 0}</text>
                  </g>
                ))}

                {scheme === 'ordinal' && (
                  <text x={MAT_X} y={y + ROW_H / 2 + 4} fontSize={13} fill="var(--ink)" fontFamily="var(--font-mono, monospace)">{i}</text>
                )}

                {scheme === 'target' && (
                  <>
                    <rect x={MAT_X} y={y + 7} width={city.rate * (W - MAT_X - 70)} height={ROW_H - 14} rx={2} fill="var(--accent)" opacity={0.55} />
                    <text x={W - 56} y={y + ROW_H / 2 + 4} fontSize={12} fill="var(--ink)" fontFamily="var(--font-mono, monospace)">{city.rate.toFixed(2)}</text>
                  </>
                )}
              </g>
            )
          })}

          <text x={NAME_X} y={H - 18} fontSize={11} fill="var(--ink-muted)" fontFamily="var(--font-sans, sans-serif)">
            {scheme === 'onehot' ? `${k} columns` : '1 column'}
          </text>
        </svg>
      </div>

      <p className="font-sans text-sm text-ink-muted mt-3 text-center max-w-prose mx-auto">{NOTE[scheme]}</p>

      <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
        Figure 17.3 — Three ways to turn a category into numbers. The same
        six-city column becomes six binary columns (one-hot), a single integer
        that smuggles in a fake ordering (ordinal), or a single column holding
        each city&rsquo;s average label (target encoding). Each trades
        column-count against information and risk — and target encoding, by
        looking at the label, opens the door to the leakage that §11 and the
        last problem are about.
      </figcaption>
    </figure>
  )
}
