import { useState, useRef, useCallback } from 'react'
import axios from 'axios'

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

export default function Search() {
  const [query, setQuery] = useState(null)
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const fileRef = useRef()

  const handleFile = useCallback(async (file) => {
    if (!file || !file.type.startsWith('image/')) return
    setError(null)
    setResults([])
    setQuery(URL.createObjectURL(file))

    const form = new FormData()
    form.append('file', file)
    setLoading(true)
    try {
      const { data } = await axios.post('/search', form)
      setResults(data.results ?? [])
    } catch (e) {
      setError(e.response?.data?.detail ?? 'Request failed')
    } finally {
      setLoading(false)
    }
  }, [])

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  return (
    <div className="space-y-5 max-w-4xl">
      <SectionLabel>Similarity Search</SectionLabel>

      <div className="grid grid-cols-[240px_1fr] gap-5">
        {/* Left: upload */}
        <div className="space-y-3">
          <div
            className={`dropzone flex flex-col items-center justify-center gap-2 h-52 transition-all
              ${dragging ? 'active' : ''}`}
            onClick={() => fileRef.current.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            {query
              ? <img src={query} alt="query" className="max-h-full max-w-full object-contain rounded" />
              : <>
                  <UploadIcon />
                  <span className="text-xs text-[var(--subtle)] text-center px-4">
                    Upload a face to find its nearest neighbours
                  </span>
                </>
            }
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={e => handleFile(e.target.files[0])}
            />
          </div>
          <div className="panel px-3 py-2 flex items-center justify-between">
            <span className="mono text-[10px] text-[var(--muted)] tracking-widest">QUERY</span>
            {loading && <Spinner />}
          </div>
        </div>

        {/* Right: results */}
        <div className="space-y-2">
          {error && (
            <p className="mono text-xs text-[var(--warn)] border border-[var(--warn)]/30 rounded px-3 py-2">
              {error}
            </p>
          )}

          {results.length === 0 && !loading && (
            <div className="panel flex items-center justify-center h-52 text-[var(--muted)]">
              <span className="mono text-xs">results will appear here</span>
            </div>
          )}

          {loading && results.length === 0 && (
            <div className="panel flex items-center justify-center h-52">
              <Spinner large />
            </div>
          )}

          {results.length > 0 && (
            <div className="grid grid-cols-5 gap-2">
              {results.map((r, i) => (
                <ResultCard key={i} rank={i + 1} result={r} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function ResultCard({ rank, result }) {
  const [showAttrs, setShowAttrs] = useState(false)
  const activeAttrs = result.attributes
    ? result.attributes.map((v, i) => v === 1 ? ATTR_NAMES[i] : null).filter(Boolean)
    : []

  return (
    <div className="panel overflow-hidden group cursor-pointer" onClick={() => setShowAttrs(s => !s)}>
      <div className="relative">
        <img
          src={`data:image/jpeg;base64,${result.image_b64}`}
          alt={`result ${rank}`}
          className="w-full aspect-square object-cover"
        />
        <div className="absolute top-1 left-1 bg-[var(--ink)]/80 rounded px-1.5 py-0.5">
          <span className="mono text-[9px] text-[var(--accent)]">#{rank}</span>
        </div>
        <div className="absolute bottom-1 right-1 bg-[var(--ink)]/80 rounded px-1.5 py-0.5">
          <span className="mono text-[9px] text-[var(--muted)]">{result.distance?.toFixed(3)}</span>
        </div>
      </div>

      {showAttrs && activeAttrs.length > 0 && (
        <div className="p-1.5 flex flex-wrap gap-1 border-t border-[var(--rim)]">
          {activeAttrs.slice(0, 6).map(a => (
            <span key={a} className="mono text-[8px] text-[var(--accent)] bg-[var(--accent)]/10 px-1 rounded">
              {a.replace(/_/g, ' ')}
            </span>
          ))}
          {activeAttrs.length > 6 && (
            <span className="mono text-[8px] text-[var(--muted)]">+{activeAttrs.length - 6}</span>
          )}
        </div>
      )}
    </div>
  )
}

function SectionLabel({ children }) {
  return (
    <div className="flex items-center gap-3">
      <span className="mono text-[10px] text-[var(--accent)] tracking-widest uppercase">{children}</span>
      <div className="flex-1 h-px bg-[var(--rim)]" />
    </div>
  )
}

function Spinner({ large }) {
  const sz = large ? 'h-6 w-6' : 'h-3 w-3'
  return (
    <svg className={`animate-spin ${sz} text-[var(--accent)]`} viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
    </svg>
  )
}

function UploadIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
      className="text-[var(--muted)]">
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  )
}
