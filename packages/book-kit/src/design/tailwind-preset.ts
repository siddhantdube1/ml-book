import type { Config } from 'tailwindcss'

// Shared Tailwind preset: the book's typefaces, colour tokens (mapped to the
// CSS custom properties defined in theme/globals.css), and layout widths.
// Each book's tailwind.config does `presets: [bookKitPreset]` and points its
// `content` globs at both its own source and packages/book-kit.
const bookKitPreset: Partial<Config> = {
  theme: {
    extend: {
      fontFamily: {
        serif: ['var(--font-serif)', 'Charter', 'Georgia', 'serif'],
        sans: ['var(--font-sans)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-mono)', 'ui-monospace', 'monospace'],
      },
      colors: {
        paper: 'var(--paper)',
        'paper-soft': 'var(--paper-soft)',
        ink: 'var(--ink)',
        'ink-muted': 'var(--ink-muted)',
        'ink-faint': 'var(--ink-faint)',
        rule: 'var(--rule)',
        accent: 'var(--accent)',
      },
      maxWidth: {
        prose: '68ch',
        wide: '88ch',
      },
    },
  },
}

export default bookKitPreset
