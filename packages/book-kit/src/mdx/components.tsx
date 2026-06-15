import type { MDXComponents } from 'mdx/types'
import Link from 'next/link'
import type { AnchorHTMLAttributes } from 'react'

/**
 * Shared MDX component overrides. Routes internal links (auto-generated
 * "Chapter N" references, "Next: Chapter N" footers, cross-book links)
 * through Next.js client-side navigation so the in-memory theme on <html>
 * survives the click. External and hash links keep the plain element.
 *
 * Each book re-exports this from its root `mdx-components.tsx`.
 */
export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    ...components,
    a: ({ href, ...props }: AnchorHTMLAttributes<HTMLAnchorElement>) => {
      const target = href ?? ''
      if (target.startsWith('/')) {
        return <Link href={target} {...props} />
      }
      return <a href={target} {...props} />
    },
  }
}
