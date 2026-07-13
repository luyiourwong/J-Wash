// Token display: leading/trailing spaces made visible with ˽ (like the original
// JLens) — ' Paris' → '˽Paris', 'foo ' → 'foo˽'. Without this it's impossible to
// tell ' Euro' from 'Euro' in the UI.
export function fmtTok(s) {
  if (s == null) return ''
  return String(s).replace(/^ +| +$/g, (m) => '˽'.repeat(m.length))
}
