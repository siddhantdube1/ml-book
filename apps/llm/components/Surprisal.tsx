'use client'

import { useEffect, useState } from 'react'
import type { SurprisalFixture } from '@trilogy/book-kit/model'

const CAP = 14 // bits, for colour scaling

type Props = {
  fixtureIds: string[]
  caption?: string
}

/**
 * Per-token surprisal: shade each real token by how surprised the model was to
 * see it. This is the training signal — pretraining minimises exactly this.
 */
export default function Surprisal({ fixtureIds, caption }: Props) {
  const [which, setWhich] = useState(0)
  const [fix, setFix] = useState<SurprisalFixture | null>(null)
  const fixtureId = fixtureIds[which]

  useEffect(() => {
    let alive = true
    setFix(null)
    fetch(`/fixtures/surprisal/${fixtureId}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d: SurprisalFixture) => alive && setFix(d))
      .catch(() => {})
    return () => {
      alive = false
    }
  }, [fixtureId])

  const scored = fix?.tokens.filter((t) => t.surprisal != null) ?? []
  const avg =
    scored.length > 0
      ? scored.reduce((s, t) => s + (t.surprisal ?? 0), 0) / scored.length
      : 0

  return (
    <figure className="my-10 not-prose">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 mb-4 font-sans text-sm">
        <span className="text-ink-muted">Sentence</span>
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
            <div className="font-serif text-xl leading-loose">
              {fix.tokens.map((t, i) => {
                const s = t.surprisal ?? 0
                return (
                  <span
                    key={i}
                    style={{
                      backgroundColor:
                        t.surprisal == null
                          ? 'transparent'
                          : `color-mix(in srgb, var(--accent) ${Math.round(
                              Math.min(s / CAP, 1) * 100,
                            )}%, transparent)`,
                      color: s > CAP * 0.62 ? 'var(--paper)' : 'var(--ink)',
                      borderRadius: 3,
                      padding: '1px 2px',
                    }}
                    title={
                      t.surprisal == null
                        ? 'first token — no prediction'
                        : `${t.surprisal} bits · model gave ${(
                            (t.prob ?? 0) * 100
                          ).toFixed(1)}%`
                    }
                  >
                    {t.text}
                  </span>
                )
              })}
            </div>
            <div className="mt-4 font-sans text-xs text-ink-faint">
              Brighter = more surprising (higher loss). Hover a token for its
              probability. Average surprisal:{' '}
              <span className="font-mono text-ink-muted">
                {avg.toFixed(2)} bits/token
              </span>
              .
            </div>
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
