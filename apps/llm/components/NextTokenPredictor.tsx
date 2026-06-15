'use client'

import { useEffect, useState } from 'react'
import type { NextTokenFixture } from '@trilogy/book-kit/model'

function tokenLabel(text: string): string {
  if (text === '') return '·'
  if (/^\s+$/.test(text)) return '␣'
  return text.replace(/^ /, '␣')
}

const SHOWN = 10

type Props = {
  /** Fixture ids under /public/fixtures/nexttoken; the first is the default. */
  fixtureIds: string[]
  caption?: string
}

/**
 * Read-only view of GPT-2's next-token distribution: pick a context, see what
 * the model thinks comes next. The bare objective of a language model, made
 * concrete. (Chapter 14 adds the sampling controls.)
 */
export default function NextTokenPredictor({ fixtureIds, caption }: Props) {
  const [which, setWhich] = useState(0)
  const [fix, setFix] = useState<NextTokenFixture | null>(null)
  const fixtureId = fixtureIds[which]

  useEffect(() => {
    let alive = true
    setFix(null)
    fetch(`/fixtures/nexttoken/${fixtureId}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d: NextTokenFixture) => alive && setFix(d))
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
            <div className="font-serif text-lg mb-1">
              {fix.context}
              <span className="text-accent font-sans"> ?</span>
            </div>
            <div className="font-sans text-xs text-ink-faint mb-4">
              GPT-2&rsquo;s most likely next tokens.
            </div>
            <div className="space-y-1">
              {fix.topk.slice(0, SHOWN).map((t, i) => {
                const max = fix.topk[0].prob || 1
                return (
                  <div key={i} className="flex items-center gap-2">
                    <span className="font-mono text-sm w-24 shrink-0 text-right text-ink">
                      {tokenLabel(t.text)}
                    </span>
                    <div className="flex-1 h-4">
                      <div
                        className="h-4 rounded-sm"
                        style={{
                          width: `${(t.prob / max) * 100}%`,
                          background: 'var(--accent)',
                          minWidth: 2,
                        }}
                      />
                    </div>
                    <span className="font-mono text-xs w-12 shrink-0 text-ink-muted">
                      {(t.prob * 100).toFixed(1)}
                    </span>
                  </div>
                )
              })}
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
