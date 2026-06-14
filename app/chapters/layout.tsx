import Link from 'next/link'
import ChapterNav from '@/components/ChapterNav'

export default function ChaptersLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen">
      <nav className="max-w-wide mx-auto px-6 pt-8">
        <Link
          href="/"
          className="font-sans text-sm text-ink-muted hover:text-ink transition-colors inline-flex items-center gap-1.5"
        >
          <span>‹</span>
          <span>Table of contents</span>
        </Link>
      </nav>
      <article className="max-w-wide mx-auto px-6 py-12 md:py-16 prose-chapter">
        {children}
        <ChapterNav />
      </article>
    </div>
  )
}
