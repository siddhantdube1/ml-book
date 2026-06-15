import type { Config } from 'tailwindcss'
import bookKitPreset from '@trilogy/book-kit/tailwind-preset'

const config: Config = {
  presets: [bookKitPreset as Config],
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    // Generate utilities for classes used inside shared book-kit components.
    '../../packages/book-kit/src/**/*.{js,ts,jsx,tsx}',
  ],
}

export default config
