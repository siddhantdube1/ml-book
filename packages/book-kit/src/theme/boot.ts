// Runs before first paint to set the theme, avoiding a flash of the wrong
// mode. Reads a saved choice, otherwise falls back to the OS preference.
// Inject verbatim via <script dangerouslySetInnerHTML> in each book's <head>.
export const themeInitScript = `(function(){try{var t=localStorage.getItem('theme');if(t!=='light'&&t!=='dark'){t=window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';}document.documentElement.setAttribute('data-theme',t);}catch(e){}})();`

// The book's three typefaces: Source Serif 4 (body), Geist (UI), JetBrains
// Mono (code). Shared so every book renders identically.
export const FONT_STYLESHEET_HREF =
  'https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&family=Geist:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap'
