'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { adjacentChapters, type ChapterMeta } from './registry'

function slugFromPath(pathname: string | null): string | null {
  if (!pathname) return null
  const m = /\/chapters\/([^/]+)/.exec(pathname)
  return m ? m[1] : null
}

/**
 * Prev/next chapter cards. Book-agnostic: pass the current book's chapter
 * registry. Renders nothing outside a recognised /chapters/<slug> route.
 */
export default function ChapterNav({
  chapters,
  tocHref = '/',
}: {
  chapters: ChapterMeta[]
  tocHref?: string
}) {
  const pathname = usePathname()
  const slug = slugFromPath(pathname)
  if (!slug) return null

  const { prev, next } = adjacentChapters(chapters, slug)
  if (!prev && !next) return null

  return (
    <nav
      className="chapter-nav not-prose mt-20 pt-8 border-t border-rule grid grid-cols-1 sm:grid-cols-2 gap-4"
      aria-label="Chapter navigation"
    >
      {prev ? (
        <NavCard chapter={prev} direction="prev" />
      ) : (
        <TocCard direction="prev" tocHref={tocHref} />
      )}
      {next ? (
        <NavCard chapter={next} direction="next" />
      ) : (
        <TocCard direction="next" tocHref={tocHref} />
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

function TocCard({
  direction,
  tocHref,
}: {
  direction: 'prev' | 'next'
  tocHref: string
}) {
  const isNext = direction === 'next'
  return (
    <Link
      href={tocHref}
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
