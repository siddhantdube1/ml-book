import katex from 'katex'

/**
 * Renders a LaTeX expression using KaTeX. Use this for math inside React
 * components — figure captions, axis labels, in-widget annotations.
 *
 * Named `Tex` rather than `Math` to avoid shadowing JavaScript's global
 * `Math` object inside components that use `Math.sqrt`, `Math.cos`, etc.
 *
 * For math inside MDX prose, use the standard `$...$` and `$$...$$`
 * syntax; the MDX pipeline handles that automatically.
 *
 * Examples:
 *   <Tex>{String.raw`\sqrt{2}`}</Tex>
 *   <Tex>{'x^2 + y^2 = r^2'}</Tex>
 *   <Tex display>{String.raw`\sum_{i=1}^n x_i`}</Tex>
 *
 * Tip: prefer `String.raw` for any expression with backslashes — it lets
 * you write `\sqrt{2}` instead of `\\sqrt{2}`.
 */
export function Tex({
  children,
  display = false,
}: {
  children: string
  display?: boolean
}) {
  const html = katex.renderToString(children, {
    displayMode: display,
    throwOnError: false,
    output: 'html',
  })
  const Tag = display ? 'div' : 'span'
  return (
    <Tag
      aria-label={children}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}
