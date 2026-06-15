'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { adjacentChapters, type ChapterMeta } from '@/lib/chapters'

function slugFromPath(pathname: string | null): string | null {
  if (!pathname) return null
  const m = /\/chapters\/([^/]+)/.exec(pathname)
  return m ? m[1] : null
}

export default function ChapterNav() {
  const pathname = usePathname()
  const slug = slugFromPath(pathname)
  if (!slug) return null

  const { prev, next } = adjacentChapters(slug)
  // Nothing to show if this isn't a known chapter.
  if (!prev && !next) return null

  return (
    <nav
      className="chapter-nav not-prose mt-20 pt-8 border-t border-rule grid grid-cols-1 sm:grid-cols-2 gap-4"
      aria-label="Chapter navigation"
    >
      {prev ? (
        <NavCard chapter={prev} direction="prev" />
      ) : (
        <TocCard direction="prev" />
      )}
      {next ? (
        <NavCard chapter={next} direction="next" />
      ) : (
        <TocCard direction="next" />
      )}
    </nav>
  )
}

function NavCard({
  chapter,
  direction,
}: {
  chapter: ChapterMeta
  direction: 'prev' | 'next'
}) {
  const isNext = direction === 'next'
  return (
    <Link
      href={`/chapters/${chapter.slug}`}
      className={`group block rounded-lg border border-rule px-5 py-4 hover:border-accent transition-colors ${
        isNext ? 'sm:text-right' : ''
      }`}
    >
      <span className="font-sans text-xs uppercase tracking-[0.16em] text-ink-faint">
        {isNext ? 'Next ›' : '‹ Previous'}
      </span>
      <span className="mt-1 block font-sans text-xs uppercase tracking-wider text-ink-muted">
        Chapter {chapter.num}
      </span>
      <span className="mt-0.5 block text-lg leading-snug text-ink group-hover:text-accent transition-colors">
        {chapter.title}
      </span>
    </Link>
  )
}

function TocCard({ direction }: { direction: 'prev' | 'next' }) {
  const isNext = direction === 'next'
  return (
    <Link
      href="/"
      className={`group block rounded-lg border border-rule px-5 py-4 hover:border-accent transition-colors ${
        isNext ? 'sm:text-right' : ''
      }`}
    >
      <span className="font-sans text-xs uppercase tracking-[0.16em] text-ink-faint">
        {isNext ? 'Finish ›' : '‹ Back'}
      </span>
      <span className="mt-0.5 block text-lg leading-snug text-ink group-hover:text-accent transition-colors">
        Table of contents
      </span>
    </Link>
  )
}
