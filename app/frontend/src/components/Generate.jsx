import { useState, useRef, useCallback } from 'react'
import axios from 'axios'

export default function Generate() {
  return (
    <div className="space-y-8 max-w-4xl">
      <InterpolateSection />
      <GenerateSection />
    </div>
  )
}

/* ── Interpolation ─────────────────────────────────────────────────────────── */

function InterpolateSection() {
  const [imgA, setImgA] = useState(null)
  const [imgB, setImgB] = useState(null)
  const [strip, setStrip] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [steps, setSteps] = useState(8)
  const fileARef = useRef()
  const fileBRef = useRef()
  const fileDataA = useRef(null)
  const fileDataB = useRef(null)

  const handleFile = (which, file) => {
    if (!file || !file.type.startsWith('image/')) return
    const url = URL.createObjectURL(file)
    if (which === 'A') { setImgA(url); fileDataA.current = file }
    else               { setImgB(url); fileDataB.current = file }
    setStrip([])
    setError(null)
  }

  const run = useCallback(async () => {
    if (!fileDataA.current || !fileDataB.current) return
    setError(null)
    setStrip([])
    setLoading(true)
    const form = new FormData()
    form.append('file_a', fileDataA.current)
    form.append('file_b', fileDataB.current)
    form.append('steps', steps)
    try {
      const { data } = await axios.post('/interpolate', form)
      setStrip(data.frames ?? [])
    } catch (e) {
      setError(e.response?.data?.detail ?? 'Request failed')
    } finally {
      setLoading(false)
    }
  }, [steps])

  return (
    <div className="space-y-4">
      <SectionLabel>Interpolation</SectionLabel>

      <div className="grid grid-cols-[120px_1fr_120px] gap-4 items-start">
        <FaceSlot label="FACE A" src={imgA} fileRef={fileARef} onChange={f => handleFile('A', f)} />

        <div className="space-y-3">
          {/* steps control */}
          <div className="panel flex items-center gap-3 px-4 py-2">
            <span className="mono text-[10px] text-[var(--muted)] tracking-widest">STEPS</span>
            <input
              type="range" min={4} max={16} step={1} value={steps}
              onChange={e => setSteps(Number(e.target.value))}
              className="flex-1 accent-[var(--accent)] h-1 cursor-pointer"
            />
            <span className="mono text-xs text-[var(--accent)] w-4 text-right">{steps}</span>
          </div>

          <button
            onClick={run}
            disabled={!imgA || !imgB || loading}
            className="w-full panel py-2 mono text-xs text-[var(--accent)] tracking-widest
              hover:bg-[var(--surface)] disabled:opacity-30 disabled:cursor-not-allowed
              transition-colors flex items-center justify-center gap-2"
          >
            {loading ? <><Spinner /> RUNNING…</> : 'INTERPOLATE →'}
          </button>

          {error && (
            <p className="mono text-xs text-[var(--warn)] border border-[var(--warn)]/30 rounded px-3 py-2">
              {error}
            </p>
          )}

          {/* strip */}
          {strip.length > 0 && (
            <div className="panel overflow-hidden">
              <div className="flex items-center gap-2 px-3 py-1.5 border-b border-[var(--rim)]">
                <span className="mono text-[9px] text-[var(--muted)] tracking-widest">
                  INTERPOLATION STRIP · {strip.length} FRAMES
                </span>
              </div>
              <div className="flex overflow-x-auto">
                {strip.map((b64, i) => (
                  <img
                    key={i}
                    src={`data:image/jpeg;base64,${b64}`}
                    alt={`frame ${i}`}
                    className="h-28 object-cover flex-shrink-0"
                    style={{ width: `${100 / strip.length}%`, minWidth: 60 }}
                  />
                ))}
              </div>
            </div>
          )}

          {!strip.length && !loading && (
            <div className="panel flex items-center justify-center h-28 text-[var(--muted)]">
              <span className="mono text-xs">upload both faces then click interpolate</span>
            </div>
          )}
        </div>

        <FaceSlot label="FACE B" src={imgB} fileRef={fileBRef} onChange={f => handleFile('B', f)} />
      </div>
    </div>
  )
}

function FaceSlot({ label, src, fileRef, onChange }) {
  const [dragging, setDragging] = useState(false)

  return (
    <div className="space-y-1">
      <div
        className={`dropzone flex flex-col items-center justify-center h-28 transition-all
          ${dragging ? 'active' : ''}`}
        onClick={() => fileRef.current.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => { e.preventDefault(); setDragging(false); onChange(e.dataTransfer.files[0]) }}
      >
        {src
          ? <img src={src} alt={label} className="max-h-full max-w-full object-contain rounded" />
          : <span className="mono text-[9px] text-[var(--muted)]">click / drop</span>
        }
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={e => onChange(e.target.files[0])}
        />
      </div>
      <p className="mono text-[9px] text-[var(--muted)] text-center tracking-widest">{label}</p>
    </div>
  )
}

/* ── Generation ────────────────────────────────────────────────────────────── */

function GenerateSection() {
  const [faces, setFaces] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [count, setCount] = useState(6)

  const run = useCallback(async () => {
    setError(null)
    setFaces([])
    setLoading(true)
    try {
      const requests = Array.from({ length: count }, () => axios.get('/generate'))
      const responses = await Promise.all(requests)
      setFaces(responses.map(r => r.data.image_b64))
    } catch (e) {
      setError(e.response?.data?.detail ?? 'Request failed')
    } finally {
      setLoading(false)
    }
  }, [count])

  return (
    <div className="space-y-4">
      <SectionLabel>Generation · Prior Sampling</SectionLabel>

      <div className="flex items-center gap-4">
        <div className="panel flex items-center gap-3 px-4 py-2">
          <span className="mono text-[10px] text-[var(--muted)] tracking-widest">COUNT</span>
          <input
            type="range" min={1} max={12} step={1} value={count}
            onChange={e => setCount(Number(e.target.value))}
            className="w-32 accent-[var(--accent)] h-1 cursor-pointer"
          />
          <span className="mono text-xs text-[var(--accent)] w-4 text-right">{count}</span>
        </div>

        <button
          onClick={run}
          disabled={loading}
          className="panel px-5 py-2 mono text-xs text-[var(--accent)] tracking-widest
            hover:bg-[var(--surface)] disabled:opacity-30 disabled:cursor-not-allowed
            transition-colors flex items-center gap-2"
        >
          {loading ? <><Spinner /> SAMPLING…</> : 'SAMPLE FROM PRIOR →'}
        </button>
      </div>

      {error && (
        <p className="mono text-xs text-[var(--warn)] border border-[var(--warn)]/30 rounded px-3 py-2">
          {error}
        </p>
      )}

      {loading && faces.length === 0 && (
        <div className="panel flex items-center justify-center h-40">
          <Spinner large />
        </div>
      )}

      {faces.length > 0 && (
        <div
          className="grid gap-3"
          style={{ gridTemplateColumns: `repeat(${Math.min(faces.length, 6)}, minmax(0, 1fr))` }}
        >
          {faces.map((b64, i) => (
            <div key={i} className="panel overflow-hidden group">
              <img
                src={`data:image/jpeg;base64,${b64}`}
                alt={`generated ${i}`}
                className="w-full aspect-square object-cover group-hover:scale-105 transition-transform duration-300"
              />
            </div>
          ))}
        </div>
      )}

      {!loading && faces.length === 0 && (
        <div className="panel flex items-center justify-center h-40">
          <span className="mono text-xs text-[var(--muted)]">
            samples from z ~ N(0, I) will appear here
          </span>
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
