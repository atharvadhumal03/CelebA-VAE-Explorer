import { useState, useEffect } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'

const ATTR_NAMES = [
  '5_o_Clock_Shadow','Arched_Eyebrows','Attractive','Bags_Under_Eyes','Bald',
  'Bangs','Big_Lips','Big_Nose','Black_Hair','Blond_Hair','Blurry','Brown_Hair',
  'Bushy_Eyebrows','Chubby','Double_Chin','Eyeglasses','Goatee','Gray_Hair',
  'Heavy_Makeup','High_Cheekbones','Male','Mouth_Slightly_Open','Mustache',
  'Narrow_Eyes','No_Beard','Oval_Face','Pale_Skin','Pointy_Nose',
  'Receding_Hairline','Rosy_Cheeks','Sideburns','Smiling','Straight_Hair',
  'Wavy_Hair','Wearing_Earrings','Wearing_Hat','Wearing_Lipstick',
  'Wearing_Necklace','Wearing_Necktie','Young',
]

export default function Explore() {
  const [tsneData, setTsneData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [attrIdx, setAttrIdx] = useState(20) // Male by default

  useEffect(() => {
    axios.get('/tsne')
      .then(({ data }) => setTsneData(data))
      .catch(e => setError(e.response?.data?.detail ?? 'Failed to load t-SNE data'))
      .finally(() => setLoading(false))
  }, [])

  const plotData = tsneData
    ? buildPlotData(tsneData, attrIdx)
    : []

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <SectionLabel>Latent Space · t-SNE</SectionLabel>
        <select
          value={attrIdx}
          onChange={e => setAttrIdx(Number(e.target.value))}
          className="ml-auto panel mono text-xs text-[var(--body)] bg-[var(--panel)] px-3 py-1.5
            border border-[var(--rim)] rounded outline-none focus:border-[var(--accent)] cursor-pointer"
        >
          {ATTR_NAMES.map((a, i) => (
            <option key={i} value={i}>{a.replace(/_/g, ' ')}</option>
          ))}
        </select>
      </div>

      {error && (
        <p className="mono text-xs text-[var(--warn)] border border-[var(--warn)]/30 rounded px-3 py-2">
          {error}
        </p>
      )}

      {loading && (
        <div className="panel flex items-center justify-center h-[520px]">
          <div className="flex flex-col items-center gap-3">
            <Spinner />
            <span className="mono text-xs text-[var(--muted)]">loading t-SNE projection…</span>
          </div>
        </div>
      )}

      {!loading && tsneData && (
        <div className="panel overflow-hidden">
          <Plot
            data={plotData}
            layout={{
              paper_bgcolor: '#0d1117',
              plot_bgcolor: '#080c10',
              margin: { t: 8, b: 40, l: 40, r: 8 },
              xaxis: {
                color: '#3a4d60',
                gridcolor: '#1e2d3d',
                zerolinecolor: '#1e2d3d',
                tickfont: { family: 'IBM Plex Mono', size: 9 },
              },
              yaxis: {
                color: '#3a4d60',
                gridcolor: '#1e2d3d',
                zerolinecolor: '#1e2d3d',
                tickfont: { family: 'IBM Plex Mono', size: 9 },
              },
              legend: {
                font: { family: 'IBM Plex Mono', size: 10, color: '#8ba0b4' },
                bgcolor: '#0d1117',
                bordercolor: '#1e2d3d',
                borderwidth: 1,
              },
              hovermode: 'closest',
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%', height: '520px' }}
            useResizeHandler
          />
        </div>
      )}

      <p className="mono text-[10px] text-[var(--muted)]">
        {tsneData?.points?.length?.toLocaleString() ?? '—'} points · coloured by{' '}
        <span className="text-[var(--accent)]">{ATTR_NAMES[attrIdx].replace(/_/g, ' ')}</span>
      </p>
    </div>
  )
}

function buildPlotData(tsneData, attrIdx) {
  const xs0 = [], ys0 = [], xs1 = [], ys1 = []

  for (const pt of tsneData.points) {
    const val = pt.attributes?.[attrIdx] ?? 0
    if (val === 1) {
      xs1.push(pt.x)
      ys1.push(pt.y)
    } else {
      xs0.push(pt.x)
      ys0.push(pt.y)
    }
  }

  const attrLabel = ATTR_NAMES[attrIdx].replace(/_/g, ' ')

  return [
    {
      x: xs0, y: ys0,
      type: 'scatter', mode: 'markers',
      name: `No ${attrLabel}`,
      marker: { color: '#1d4ed8', size: 3, opacity: 0.6 },
    },
    {
      x: xs1, y: ys1,
      type: 'scatter', mode: 'markers',
      name: attrLabel,
      marker: { color: '#3b82f6', size: 4, opacity: 0.85 },
    },
  ]
}

function SectionLabel({ children }) {
  return (
    <div className="flex items-center gap-3">
      <span className="mono text-[10px] text-[var(--accent)] tracking-widest uppercase">{children}</span>
      <div className="h-px bg-[var(--rim)] w-12" />
    </div>
  )
}

function Spinner() {
  return (
    <svg className="animate-spin h-5 w-5 text-[var(--accent)]" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
    </svg>
  )
}
