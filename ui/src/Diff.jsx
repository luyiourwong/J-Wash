import { useMemo } from 'react'
import { fmtTok } from './tok'

const MAX_COLS = 200

function top1(frame, layer) {
  const d = frame.layers[layer]
  if (!d) return null
  return { str: fmtTok(d.m_strs[0]), rank: d.m_rank[0], alt: d.m_strs.slice(0, 3).map(fmtTok).join(', ') }
}

export default function LensDiff({ framesA, framesB, labelA, labelB, onClose }) {
  const layers = useMemo(() => {
    if (!framesA.length || !framesB.length) return []
    const a = new Set(Object.keys(framesA[0].layers))
    return Object.keys(framesB[0].layers).filter((l) => a.has(l)).map(Number).sort((x, y) => x - y)
  }, [framesA, framesB])

  const cols = Math.min(framesA.length, framesB.length, MAX_COLS)
  if (!cols || !layers.length) {
    return (
      <div className="lensview">
        <div className="lv-controls">
          diff not possible: incompatible layers or frames
          <button style={{ marginLeft: 'auto' }} onClick={onClose}>close</button>
        </div>
      </div>
    )
  }

  let diffCount = 0
  const rows = layers.map((layer) => {
    const cells = []
    for (let i = 0; i < cols; i++) {
      const a = top1(framesA[i], String(layer))
      const b = top1(framesB[i], String(layer))
      const same = a && b && a.str === b.str
      if (!same) diffCount++
      cells.push({ a, b, same, tokA: framesA[i].tok, tokB: framesB[i].tok })
    }
    return { layer, cells }
  })

  return (
    <div className="lensview">
      <div className="lv-controls">
        <span>diff: <b>A</b> = {labelA} · <b>B</b> = {labelB}</span>
        <span className="lv-sep" />
        <span>{diffCount} divergent cells / {cols * layers.length}</span>
        <button style={{ marginLeft: 'auto' }} onClick={onClose}>close</button>
      </div>
      <div className="lv-scroll">
        <table className="diff-table">
          <thead>
            <tr>
              <th></th>
              {Array.from({ length: cols }, (_, i) => (
                <th key={i} title={`A: ${framesA[i].tok} · B: ${framesB[i].tok}`}>
                  {fmtTok(framesA[i].tok).slice(0, 6) || '·'}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ layer, cells }) => (
              <tr key={layer}>
                <td className="diff-lay">L{layer}</td>
                {cells.map((c, i) => (
                  <td key={i} className={c.same ? 'diff-same' : 'diff-diff'}
                    title={`A(${c.tokA}): ${c.a?.alt || '—'}\nB(${c.tokB}): ${c.b?.alt || '—'}`}>
                    <div className="diff-a">{c.a?.str.slice(0, 7) || '—'}</div>
                    <div className="diff-b">{c.b?.str.slice(0, 7) || '—'}</div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
