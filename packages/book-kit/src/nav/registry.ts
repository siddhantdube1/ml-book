// Chapter registry types + helpers, shared across every book.
// Each book supplies its own `ChapterMeta[]`; the platform stays book-agnostic.

export type ChapterMeta = {
  num: number
  slug: string
  title: string
}

export function chapterHref(slug: string): string {
  return `/chapters/${slug}`
}

/** Find a chapter by its folder slug within a given book's registry. */
export function chapterBySlug(
  chapters: ChapterMeta[],
  slug: string,
): ChapterMeta | undefined {
  return chapters.find((c) => c.slug === slug)
}

/** The chapters immediately before and after the given slug, if any. */
export function adjacentChapters(
  chapters: ChapterMeta[],
  slug: string,
): { prev?: ChapterMeta; next?: ChapterMeta } {
  const i = chapters.findIndex((c) => c.slug === slug)
  if (i === -1) return {}
  return {
    prev: i > 0 ? chapters[i - 1] : undefined,
    next: i < chapters.length - 1 ? chapters[i + 1] : undefined,
  }
}
