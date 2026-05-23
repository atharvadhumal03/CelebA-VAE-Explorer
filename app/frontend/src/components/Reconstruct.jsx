import { useState, useRef, useCallback } from 'react'
import axios from 'axios'

export default function Reconstruct() {
  const [original, setOriginal] = useState(null)
  const [reconstructed, setReconstructed] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const fileRef = useRef()

  const handleFile = useCallback(async (file) => {
    if (!file || !file.type.startsWith('image/')) return
    setError(null)
    setReconstructed(null)
    setMetrics(null)
    setOriginal(URL.createObjectURL(file))

    const form = new FormData()
    form.append('file', file)
    setLoading(true)
    try {
      const { data } = await axios.post('/reconstruct', form)
      setReconstructed(`data:image/jpeg;base64,${data.image_b64}`)
      setMetrics({ mse: data.mse, psnr: data.psnr })
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
    <div className="space-y-5 max-w-3xl">
      <SectionLabel>Reconstruction</SectionLabel>

      {/* Drop zone */}
      <div
        className={`dropzone flex flex-col items-center justify-center gap-2 h-36 transition-all
          ${dragging ? 'active' : ''}`}
        onClick={() => fileRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <UploadIcon />
        <span className="text-sm text-[var(--subtle)]">
          Drop a face image or <span className="text-[var(--accent)]">click to browse</span>
        </span>
        <span className="mono text-[10px] text-[var(--muted)]">JPG · PNG · WEBP</span>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={e => handleFile(e.target.files[0])}
        />
      </div>

      {error && (
        <p className="mono text-xs text-[var(--warn)] border border-[var(--warn)]/30 rounded px-3 py-2">
          {error}
        </p>
      )}

      {/* Result panels */}
      {(original || loading) && (
        <div className="grid grid-cols-2 gap-4">
          <ImagePanel label="ORIGINAL" src={original} />
          <ImagePanel label="RECONSTRUCTED" src={reconstructed} loading={loading} />
        </div>
      )}

      {/* Metrics */}
      {metrics && (
        <div className="panel flex gap-6 px-5 py-3">
          <Metric label="MSE" value={metrics.mse?.toFixed(4)} />
          <div className="w-px bg-[var(--rim)]" />
          <Metric label="PSNR" value={`${metrics.psnr?.toFixed(2)} dB`} />
        </div>
      )}
    </div>
  )
}

function ImagePanel({ label, src, loading }) {
  return (
    <div className="panel overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-[var(--rim)]">
        <span className="mono text-[10px] text-[var(--muted)] tracking-widest">{label}</span>
        {loading && <Spinner />}
      </div>
      <div className="flex items-center justify-center bg-[var(--ink)] h-52">
        {src
          ? <img src={src} alt={label} className="max-h-full max-w-full object-contain" />
          : <span className="mono text-xs text-[var(--muted)]">—</span>
        }
      </div>
    </div>
  )
}

function Metric({ label, value }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="mono text-[10px] text-[var(--muted)] tracking-widest">{label}</span>
      <span className="mono text-sm text-[var(--bright)]">{value ?? '—'}</span>
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

function Spinner() {
  return (
    <svg className="animate-spin h-3 w-3 text-[var(--accent)]" viewBox="0 0 24 24" fill="none">
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
