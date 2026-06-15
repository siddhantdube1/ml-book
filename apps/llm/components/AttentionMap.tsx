'use client'

import { useEffect, useMemo, useState } from 'react'
import type { AttentionFixture } from '@trilogy/book-kit/model'

// Display helper: GPT-2 tokens carry leading spaces; show a thin gap marker so
// the boundary is legible, and render empty/whitespace tokens as a dot.
function label(text: string): string {
  if (text === '') return '·'
  if (/^\s+$/.test(text)) return '␣'
  return text.replace(/^ /, '␣')
}

type Props = {
  /** Fixture id under /public/fixtures/attention. */
  fixtureId: string
  /** Layer/head to open on (e.g. a known coreference head). */
  initialLayer?: number
  initialHead?: number
  /** Trimmed token text to pre-select as the query (e.g. "it"). */
  focusToken?: string
  caption?: string
}

export default function AttentionMap({
  fixtureId,
  initialLayer = 0,
  initialHead = 0,
  focusToken,
  caption,
}: Props) {
  const [fix, setFix] = useState<AttentionFixture | null>(null)
  const [error, setError] = useState(false)
  const [layer, setLayer] = useState(initialLayer)
  const [head, setHead] = useState(initialHead)
  const [query, setQuery] = useState<number | null>(null)

  useEffect(() => {
    let alive = true
    setFix(null)
    setError(false)
    fetch(`/fixtures/attention/${fixtureId}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d: AttentionFixture) => {
        if (!alive) return
        setFix(d)
        const focusIdx = focusToken
          ? d.tokens.findIndex((t) => t.text.trim() === focusToken)
          : -1
        setQuery(focusIdx >= 0 ? focusIdx : d.tokens.length - 1)
      })
      .catch(() => alive && setError(true))
    return () => {
      alive = false
    }
  }, [fixtureId, focusToken])

  const n = fix?.tokens.length ?? 0
  // attention[query][key] for the chosen layer/head
  const grid = useMemo(() => {
    if (!fix) return null
    const safeLayer = Math.min(layer, fix.numLayers - 1)
    const safeHead = Math.min(head, fix.numHeads - 1)
    return fix.attention[safeLayer][safeHead]
  }, [fix, layer, head])

  if (error) {
    return (
      <figure className="my-10 not-prose">
        <div className="border border-rule rounded-lg bg-paper-soft p-6 text-sm text-ink-muted font-sans">
          Could not load attention fixture{' '}
          <span className="font-mono">{fixtureId}</span>.
        </div>
      </figure>
    )
  }

  if (!fix || !grid || query === null) {
    return (
      <figure className="my-10 not-prose">
        <div className="border border-rule rounded-lg bg-paper-soft p-6 text-sm text-ink-faint font-sans">
          Loading attention…
        </div>
      </figure>
    )
  }

  // --- Geometry ---
  const cell = 30
  const padL = 92 // room for query labels on the left
  const padT = 92 // room for (rotated) key labels on top
  const W = padL + n * cell + 16
  const H = padT + n * cell + 16

  const tokens = fix.tokens

  return (
    <figure className="my-10 not-prose">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 mb-4 font-sans text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Layer <span className="font-mono">{layer}</span>
            <span className="text-ink-faint"> / {fix.numLayers - 1}</span>
          </span>
          <input
            type="range"
            min={0}
            max={fix.numLayers - 1}
            value={layer}
            onChange={(e) => setLayer(parseInt(e.target.value))}
            className="w-32"
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">
            Head <span className="font-mono">{head}</span>
            <span className="text-ink-faint"> / {fix.numHeads - 1}</span>
          </span>
          <input
            type="range"
            min={0}
            max={fix.numHeads - 1}
            value={head}
            onChange={(e) => setHead(parseInt(e.target.value))}
            className="w-32"
          />
        </label>
      </div>

      {/* Heatmap */}
      <div className="border border-rule rounded-lg overflow-x-auto bg-paper-soft">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="h-auto block"
          style={{ width: '100%', minWidth: W * 0.7 }}
        >
          {/* Key (column) labels, angled */}
          {tokens.map((t, j) => (
            <text
              key={`k${j}`}
              x={padL + j * cell + cell / 2}
              y={padT - 8}
              transform={`rotate(-55 ${padL + j * cell + cell / 2} ${padT - 8})`}
              textAnchor="start"
              className="font-mono"
              fontSize={11}
              fill="var(--ink-muted)"
            >
              {label(t.text)}
            </text>
          ))}

          {/* Query (row) labels */}
          {tokens.map((t, i) => (
            <text
              key={`q${i}`}
              x={padL - 8}
              y={padT + i * cell + cell / 2}
              textAnchor="end"
              dominantBaseline="middle"
              className="font-mono"
              fontSize={11}
              fill={i === query ? 'var(--accent)' : 'var(--ink-muted)'}
              fontWeight={i === query ? 600 : 400}
            >
              {label(t.text)}
            </text>
          ))}

          {/* Cells */}
          {grid.map((row, i) =>
            row.map((w, j) => {
              if (j > i) return null // causal mask: nothing attends to the future
              return (
                <rect
                  key={`${i}-${j}`}
                  x={padL + j * cell}
                  y={padT + i * cell}
                  width={cell - 1.5}
                  height={cell - 1.5}
                  rx={2}
                  fill="var(--accent)"
                  fillOpacity={Math.sqrt(w)}
                />
              )
            }),
          )}

          {/* Row highlight for the selected query + hover targets */}
          {tokens.map((_, i) => (
            <rect
              key={`row${i}`}
              x={padL - 2}
              y={padT + i * cell - 1}
              width={(i + 1) * cell + 2}
              height={cell}
              fill="transparent"
              stroke={i === query ? 'var(--accent)' : 'transparent'}
              strokeWidth={1.5}
              rx={2}
              onMouseEnter={() => setQuery(i)}
              style={{ cursor: 'pointer' }}
            />
          ))}
        </svg>
      </div>

      {/* Attention strip for the selected query token */}
      <div className="mt-4 rounded-lg border border-rule bg-paper-soft px-4 py-3">
        <div className="font-sans text-xs uppercase tracking-wider text-ink-faint mb-2">
          When predicting after{' '}
          <span className="font-mono text-ink normal-case tracking-normal">
            “{tokens[query].text.trim() || label(tokens[query].text)}”
          </span>
          , this head looks at:
        </div>
        <div className="leading-relaxed">
          {tokens.map((t, j) => {
            const w = j <= query ? grid[query][j] : 0
            return (
              <span
                key={j}
                className="font-mono text-[0.95rem]"
                style={{
                  backgroundColor: `color-mix(in srgb, var(--accent) ${Math.round(
                    Math.sqrt(w) * 100,
                  )}%, transparent)`,
                  color:
                    w > 0.5 ? 'var(--paper)' : 'var(--ink)',
                  borderRadius: 3,
                  padding: '1px 1px',
                }}
                title={`${(w * 100).toFixed(1)}%`}
              >
                {t.text === '' ? '·' : t.text}
              </span>
            )
          })}
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
