'use client'

import { useEffect, useMemo, useState } from 'react'

// The GPT-2 BPE merge table is large, so load it as an on-demand chunk the
// first time the widget mounts rather than bundling it into the page.
type Codec = {
  encode: (s: string) => number[]
  decode: (ids: number[]) => string
}

const PRESETS: { label: string; text: string }[] = [
  { label: 'spelling', text: 'strawberry' },
  { label: 'numbers', text: '1234567 + 89 = ?' },
  { label: 'a long word', text: 'antidisestablishmentarianism' },
  { label: 'code', text: 'for i in range(10):\n    print(i)' },
  { label: 'not English', text: 'Översättning är svårt.' },
]

// Alternating chip tints so token boundaries are visible.
const TINTS = [18, 38]

function display(text: string): string {
  // Make leading/trailing spaces and newlines visible inside a chip.
  return text.replace(/ /g, '␣').replace(/\n/g, '⏎')
}

type Props = {
  initialText?: string
  caption?: string
}

export default function LiveTokenizer({
  initialText = 'The cat sat on the mat.',
  caption,
}: Props) {
  const [text, setText] = useState(initialText)
  const [codec, setCodec] = useState<Codec | null>(null)

  useEffect(() => {
    let alive = true
    import('gpt-tokenizer/encoding/r50k_base').then((m) => {
      if (alive) setCodec({ encode: m.encode, decode: m.decode })
    })
    return () => {
      alive = false
    }
  }, [])

  const tokens = useMemo(() => {
    if (!codec) return []
    const ids = codec.encode(text)
    return ids.map((id) => ({ id, text: codec.decode([id]) }))
  }, [codec, text])

  return (
    <figure className="my-10 not-prose">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={3}
        spellCheck={false}
        className="w-full bg-paper border border-rule rounded-lg px-4 py-3 font-serif text-lg text-ink resize-y focus:outline-none focus:border-accent transition-colors"
      />

      <div className="flex flex-wrap items-center gap-2 mt-2 font-sans text-xs">
        <span className="text-ink-faint">try:</span>
        {PRESETS.map((p) => (
          <button
            key={p.label}
            onClick={() => setText(p.text)}
            className="px-2 py-0.5 border border-rule rounded-full text-ink-muted hover:border-accent hover:text-accent transition-colors"
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* Token chips */}
      <div className="mt-4 rounded-lg border border-rule bg-paper-soft px-4 py-3 leading-loose">
        {!codec ? (
          <span className="text-ink-faint font-sans text-sm">
            Loading tokenizer…
          </span>
        ) : tokens.length === 0 ? (
          <span className="text-ink-faint font-sans text-sm">
            Type something above.
          </span>
        ) : (
          tokens.map((t, i) => (
            <span
              key={i}
              className="font-mono text-[0.95rem] rounded-sm"
              style={{
                backgroundColor: `color-mix(in srgb, var(--accent) ${
                  TINTS[i % TINTS.length]
                }%, transparent)`,
                padding: '2px 1px',
                marginRight: 1,
              }}
              title={`id ${t.id}`}
            >
              {display(t.text)}
            </span>
          ))
        )}
      </div>

      {/* Counts */}
      <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 font-mono text-xs text-ink-muted">
        <span>
          <span className="text-ink-faint">characters</span> {text.length}
        </span>
        <span>
          <span className="text-ink-faint">tokens</span> {tokens.length}
        </span>
        <span>
          <span className="text-ink-faint">ratio</span>{' '}
          {tokens.length > 0
            ? (text.length / tokens.length).toFixed(2)
            : '—'}{' '}
          chars/token
        </span>
        <span>
          <span className="text-ink-faint">ids</span>{' '}
          {tokens
            .slice(0, 12)
            .map((t) => t.id)
            .join(' ')}
          {tokens.length > 12 ? ' …' : ''}
        </span>
      </div>

      {caption && (
        <figcaption className="font-sans text-sm text-ink-muted mt-4 text-center max-w-prose mx-auto">
          {caption}
        </figcaption>
      )}
    </figure>
  )
}
