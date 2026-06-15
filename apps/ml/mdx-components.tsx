import type { MDXComponents } from 'mdx/types'
import Link from 'next/link'
import type { AnchorHTMLAttributes } from 'react'

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    ...components,
    // Route internal links (the auto-generated "Chapter N" references and the
    // "Next: Chapter N" footer) through Next.js client-side navigation. A full
    // page load would briefly reset the in-memory theme on <html>; client nav
    // preserves it, matching the prev/next cards.
    a: ({ href, ...props }: AnchorHTMLAttributes<HTMLAnchorElement>) => {
      const target = href ?? ''
      if (target.startsWith('/')) {
        return <Link href={target} {...props} />
      }
      // Hash anchors and external URLs keep the plain element.
      return <a href={target} {...props} />
    },
  }
}
