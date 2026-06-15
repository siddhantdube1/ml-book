'use client'

import { useEffect, useState } from 'react'
import type { LogitLensFixture } from '@trilogy/book-kit/model'

function lab(text: string): string {
  return text.trim() === '' ? '·' : text.trim()
}

type Props = {
  fixtureIds: string[]
  caption?: string
}

/**
 * Logit lens: the model's top prediction read off each layer's residual
 * stream. Shows meaning emerging with depth. Final layer = true prediction.
 */
export default function LogitLens({ fixtureIds, caption }: Props) {
  const [which, setWhich] = useState(0)
  const [fix, setFix] = useState<LogitLensFixture | null>(null)
  const fixtureId = fixtureIds[which]

  useEffect(() => {
    let alive = true
    setFix(null)
    fetch(`/fixtures/logitlens/${fixtureId}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d: LogitLensFixture) => alive && setFix(d))
      .catch(() => {})
    return () => {
      alive = false
    }
  }, [fixtureId])

  return (
    <figure className="my-10 not-prose">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 mb-4 font-sans text-sm">
        <span className="text-ink-muted">Context</span>
        <select
          value={which}
          onChange={(e) => setWhich(parseInt(e.target.value))}
          className="bg-paper border border-rule rounded px-2 py-1 font-mono text-xs text-ink"
        >
          {fixtureIds.map((id, i) => (
            <option key={id} value={i}>
              {id}
            </option>
          ))}
        </select>
      </div>

      <div className="border border-rule rounded-lg bg-paper-soft px-5 py-4">
        {!fix ? (
          <span className="text-ink-faint font-sans text-sm">Loading…</span>
        ) : (
          <>
            <div className="font-serif text-lg mb-3">
              {fix.context}
              <span className="text-accent font-sans"> ?</span>
            </div>
            {(() => {
              const finalTok = lab(fix.layers[fix.layers.length - 1].top[0].text)
              return (
                <div className="space-y-0.5 font-mono text-sm">
                  {[...fix.layers].reverse().map((l) => {
                    const tok = lab(l.top[0].text)
                    const isFinal = tok === finalTok
                    const isEmbed = l.layer === 0
                    return (
                      <div key={l.layer} className="flex items-center gap-3">
                        <span className="text-ink-faint w-20 shrink-0 text-right">
                          {isEmbed
                            ? 'embed'
                            : l.layer === fix.layers.length - 1
                              ? 'final'
                              : `layer ${l.layer}`}
                        </span>
                        <span
                          className="w-24 shrink-0"
                          style={{ color: isFinal ? 'var(--accent)' : 'var(--ink)' }}
                        >
                          {tok}
                        </span>
                        <span className="text-ink-faint text-xs">
                          {(l.top[0].prob * 100).toFixed(1)}%
                        </span>
                      </div>
                    )
                  })}
                </div>
              )
            })()}
          </>
        )}
      </div>

      {caption && (
        <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
          {caption}
        </figcaption>
      )}
    </figure>
  )
}
