import Link from 'next/link'
import { chapters, parts, BUILT } from '@/lib/chapters'

export default function Home() {
  return (
    <main className="max-w-wide mx-auto px-6 py-16 md:py-24">
      <header className="max-w-prose">
        <p className="font-sans text-xs uppercase tracking-[0.18em] text-ink-muted mb-4">
          An interactive book · Book II of the trilogy
        </p>
        <h1 className="font-serif text-5xl font-normal leading-tight tracking-tight">
          How Language Models Work
        </h1>
        <p className="mt-5 text-xl text-ink-muted leading-relaxed">
          From next-token prediction to the full transformer and the behaviour
          that falls out of it. Drag a query through an attention map, steer a
          real model&rsquo;s sampling distribution, and build the pieces
          yourself — one mechanism at a time.
        </p>
      </header>

      <div className="mt-16 space-y-14">
        {parts.map((part) => (
          <section key={part.name}>
            <div className="border-b border-rule pb-2 mb-5">
              <h2 className="font-sans text-sm font-medium uppercase tracking-[0.12em] text-ink">
                {part.name}
              </h2>
              <p className="mt-1 text-sm text-ink-muted">{part.blurb}</p>
            </div>
            <ol className="space-y-1">
              {part.chapters.map((num) => {
                const ch = chapters.find((c) => c.num === num)
                if (!ch) return null
                const ready = BUILT.has(ch.slug)
                return (
                  <li key={ch.slug}>
                    {ready ? (
                      <Link
                        href={`/chapters/${ch.slug}`}
                        className="group flex items-baseline gap-3 py-1.5"
                      >
                        <span className="font-mono text-sm text-ink-faint w-7 shrink-0">
                          {ch.num}
                        </span>
                        <span className="text-lg text-ink group-hover:text-accent transition-colors">
                          {ch.title}
                        </span>
                      </Link>
                    ) : (
                      <div className="flex items-baseline gap-3 py-1.5">
                        <span className="font-mono text-sm text-ink-faint w-7 shrink-0">
                          {ch.num}
                        </span>
                        <span className="text-lg text-ink-faint">
                          {ch.title}
                        </span>
                        <span className="font-sans text-[0.65rem] uppercase tracking-wider text-ink-faint border border-rule rounded-full px-2 py-0.5">
                          soon
                        </span>
                      </div>
                    )}
                  </li>
                )
              })}
            </ol>
          </section>
        ))}
      </div>
    </main>
  )
}
