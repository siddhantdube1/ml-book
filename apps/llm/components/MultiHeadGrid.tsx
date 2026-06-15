'use client'

import { useEffect, useMemo, useState } from 'react'
import type { AttentionFixture } from '@trilogy/book-kit/model'

function label(text: string): string {
  if (text === '') return '·'
  if (/^\s+$/.test(text)) return '␣'
  return text.replace(/^ /, '␣')
}

type Props = {
  fixtureId: string
  caption?: string
}

/**
 * Shows all heads of one layer at once, so the reader can see that different
 * heads in the same layer attend to completely different things.
 */
export default function MultiHeadGrid({ fixtureId, caption }: Props) {
  const [fix, setFix] = useState<AttentionFixture | null>(null)
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState(0)

  useEffect(() => {
    let alive = true
    fetch(`/fixtures/attention/${fixtureId}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d: AttentionFixture) => alive && setFix(d))
      .catch(() => {})
    return () => {
      alive = false
    }
  }, [fixtureId])

  const n = fix?.tokens.length ?? 0

  if (!fix) {
    return (
      <figure className="my-10 not-prose">
        <div className="border border-rule rounded-lg bg-paper-soft p-6 text-sm text-ink-faint font-sans">
          Loading attention…
        </div>
      </figure>
    )
  }

  const safeLayer = Math.min(layer, fix.numLayers - 1)

  // Mini-heatmap for one head: n x n cells.
  const thumb = (h: number, size: number) => {
    const grid = fix.attention[safeLayer][h]
    const c = size / n
    return (
      <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size} className="block">
        {grid.map((row, i) =>
          row.map((w, j) =>
            j > i ? null : (
              <rect
                key={`${i}-${j}`}
                x={j * c}
                y={i * c}
                width={c}
                height={c}
                fill="var(--accent)"
                fillOpacity={Math.sqrt(w)}
              />
            ),
          ),
        )}
      </svg>
    )
  }

  return (
    <figure className="my-10 not-prose">
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
            className="w-40"
          />
        </label>
        <span className="text-ink-faint font-mono text-xs">
          “{fix.text}”
        </span>
      </div>

      <div className="border border-rule rounded-lg bg-paper-soft p-4">
        {/* Head grid */}
        <div className="grid grid-cols-4 sm:grid-cols-6 gap-2">
          {Array.from({ length: fix.numHeads }, (_, h) => (
            <button
              key={h}
              onClick={() => setHead(h)}
              className={`flex flex-col items-center gap-1 p-1.5 rounded border transition-colors ${
                h === head
                  ? 'border-accent'
                  : 'border-transparent hover:border-rule'
              }`}
            >
              {thumb(h, 48)}
              <span className="font-mono text-[0.6rem] text-ink-faint">H{h}</span>
            </button>
          ))}
        </div>

        {/* Enlarged selected head with labels */}
        <div className="mt-4 pt-4 border-t border-rule flex items-start gap-4 flex-wrap">
          <div className="shrink-0">{thumb(head, 150)}</div>
          <div className="font-sans text-sm text-ink-muted">
            <div className="font-mono text-ink mb-1">
              Layer {layer}, Head {head}
            </div>
            Rows are query tokens, columns are keys — same as Chapter 4, one head.
            <div className="mt-2 font-mono text-xs text-ink-faint">
              {fix.tokens.map((t) => label(t.text)).join(' · ')}
            </div>
          </div>
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
