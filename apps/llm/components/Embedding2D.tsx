'use client'

import { useEffect, useMemo, useState } from 'react'
import type { EmbeddingFixture } from '@trilogy/book-kit/model'

const GROUP_COLOR: Record<string, string> = {
  royalty: '#6b4a8a',
  people: '#a83263',
  country: '#3c5a8c',
  capital: '#1d6d5e',
  animal: '#c7522a',
  colour: '#5d8a3a',
  number: '#a06614',
  verb: '#5b5b62',
}

const W = 680
const H = 440
const PAD = 44

type Props = {
  /** Fixture under /public/fixtures/embeddings (default "words"). */
  fixtureId?: string
  caption?: string
}

export default function Embedding2D({ fixtureId = 'words', caption }: Props) {
  const [fix, setFix] = useState<EmbeddingFixture | null>(null)
  const [sel, setSel] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    fetch(`/fixtures/embeddings/${fixtureId}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d: EmbeddingFixture) => alive && setFix(d))
      .catch(() => {})
    return () => {
      alive = false
    }
  }, [fixtureId])

  const byText = useMemo(() => {
    const m = new Map<string, EmbeddingFixture['words'][number]>()
    fix?.words.forEach((w) => m.set(w.text, w))
    return m
  }, [fix])

  if (!fix) {
    return (
      <figure className="my-10 not-prose">
        <div className="border border-rule rounded-lg bg-paper-soft p-6 text-sm text-ink-faint font-sans">
          Loading embeddings…
        </div>
      </figure>
    )
  }

  const sx = (x: number) => PAD + x * (W - 2 * PAD)
  const sy = (y: number) => H - PAD - y * (H - 2 * PAD)

  const selWord = sel ? byText.get(sel) : null
  const neighbourSet = new Set(selWord?.neighbours ?? [])
  const groups = Array.from(new Set(fix.words.map((w) => w.group)))

  return (
    <figure className="my-10 not-prose">
      {/* Legend */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 mb-3 font-sans text-xs">
        {groups.map((g) => (
          <span key={g} className="inline-flex items-center gap-1.5 text-ink-muted">
            <span
              className="inline-block w-2.5 h-2.5 rounded-full"
              style={{ background: GROUP_COLOR[g] ?? 'var(--ink-muted)' }}
            />
            {g}
          </span>
        ))}
      </div>

      <div className="border border-rule rounded-lg overflow-x-auto bg-paper-soft">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="h-auto block"
          style={{ width: '100%', minWidth: 480 }}
          onMouseLeave={() => setSel(null)}
        >
          {/* Lines from selected word to its neighbours */}
          {selWord &&
            selWord.neighbours.map((nb) => {
              const n = byText.get(nb)
              if (!n) return null
              return (
                <line
                  key={nb}
                  x1={sx(selWord.x)}
                  y1={sy(selWord.y)}
                  x2={sx(n.x)}
                  y2={sy(n.y)}
                  stroke="var(--accent)"
                  strokeWidth={1}
                  strokeOpacity={0.5}
                />
              )
            })}

          {fix.words.map((w) => {
            const active = !sel || w.text === sel || neighbourSet.has(w.text)
            const isSel = w.text === sel
            return (
              <g
                key={w.text}
                onMouseEnter={() => setSel(w.text)}
                style={{ cursor: 'pointer' }}
                opacity={active ? 1 : 0.22}
              >
                <circle
                  cx={sx(w.x)}
                  cy={sy(w.y)}
                  r={isSel ? 6 : 4}
                  fill={GROUP_COLOR[w.group] ?? 'var(--ink-muted)'}
                  stroke={isSel ? 'var(--ink)' : 'none'}
                  strokeWidth={1.5}
                />
                <text
                  x={sx(w.x) + 7}
                  y={sy(w.y) + 3}
                  className="font-mono"
                  fontSize={isSel ? 12 : 10.5}
                  fontWeight={isSel ? 600 : 400}
                  fill="var(--ink)"
                >
                  {w.text}
                </text>
              </g>
            )
          })}
        </svg>
      </div>

      {/* Selected word neighbours */}
      <div className="mt-3 font-sans text-sm text-ink-muted min-h-[1.5rem]">
        {selWord ? (
          <>
            Nearest to{' '}
            <span className="font-mono text-ink">{selWord.text}</span>:{' '}
            {selWord.neighbours.map((n, i) => (
              <span key={n} className="font-mono text-ink">
                {n}
                {i < selWord.neighbours.length - 1 ? ', ' : ''}
              </span>
            ))}
          </>
        ) : (
          <span className="text-ink-faint">
            Hover a word to see its nearest neighbours in the full 768-dimensional
            space.
          </span>
        )}
      </div>

      {/* Analogy arithmetic */}
      <div className="mt-4 rounded-lg border border-rule bg-paper-soft px-4 py-3">
        <div className="font-sans text-xs uppercase tracking-wider text-ink-faint mb-2">
          Directions are relationships — real GPT-2 vector arithmetic
        </div>
        <div className="space-y-1.5 font-mono text-sm">
          {fix.analogies.map((a, i) => (
            <div key={i} className="text-ink">
              {a.b} − {a.a} + {a.c} ≈{' '}
              <span style={{ color: 'var(--accent)' }}>{a.predicted[0]}</span>
              {a.hit && <span className="text-ink-faint"> ✓</span>}
            </div>
          ))}
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
