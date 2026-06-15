import type { Metadata } from 'next'
import '@trilogy/book-kit/theme/globals.css'
import {
  ThemeToggle,
  themeInitScript,
  FONT_STYLESHEET_HREF,
} from '@trilogy/book-kit/theme'

const SITE_NAME = 'How Language Models Work'
const SITE_DESCRIPTION =
  'An interactive book on large language models — from next-token prediction to the full transformer and its behaviour, with live attention maps, sampling you can steer, and runnable code.'

function siteUrl(): string {
  if (process.env.NEXT_PUBLIC_SITE_URL) return process.env.NEXT_PUBLIC_SITE_URL
  if (process.env.VERCEL_URL) return `https://${process.env.VERCEL_URL}`
  return 'http://localhost:3000'
}

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl()),
  title: {
    default: SITE_NAME,
    template: `%s · ${SITE_NAME}`,
  },
  description: SITE_DESCRIPTION,
  applicationName: SITE_NAME,
  authors: [{ name: 'Siddhant' }],
  keywords: [
    'large language models',
    'LLM',
    'transformer',
    'attention',
    'interactive textbook',
    'tokenization',
    'sampling',
  ],
  openGraph: {
    type: 'website',
    url: '/',
    siteName: SITE_NAME,
    title: SITE_NAME,
    description: SITE_DESCRIPTION,
    locale: 'en_GB',
  },
  twitter: {
    card: 'summary_large_image',
    title: SITE_NAME,
    description: SITE_DESCRIPTION,
  },
  robots: { index: true, follow: true },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link href={FONT_STYLESHEET_HREF} rel="stylesheet" />
      </head>
      <body>
        <ThemeToggle />
        {children}
      </body>
    </html>
  )
}
