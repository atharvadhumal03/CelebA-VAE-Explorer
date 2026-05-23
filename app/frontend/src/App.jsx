import { useState } from 'react'
import Reconstruct from './components/Reconstruct'
import Search from './components/Search'
import Explore from './components/Explore'
import Generate from './components/Generate'

const TABS = ['Reconstruct', 'Search', 'Explore', 'Generate']

export default function App() {
  const [active, setActive] = useState('Reconstruct')

  return (
    <div className="min-h-screen" style={{ background: 'var(--ink)', color: 'var(--body)' }}>
      {/* Header */}
      <header style={{ borderBottom: '1px solid var(--rim)' }} className="px-8 py-4 flex items-center gap-4">
        <h1 className="text-base font-semibold tracking-tight" style={{ color: 'var(--bright)', fontFamily: 'IBM Plex Mono, monospace' }}>
          LatentLens
        </h1>
        <span className="text-xs" style={{ color: 'var(--muted)', fontFamily: 'IBM Plex Mono, monospace' }}>
          VAE · CelebA · 128-dim
        </span>
      </header>

      {/* Tab bar */}
      <nav className="flex gap-1 px-8 pt-4">
        {TABS.map(tab => (
          <button
            key={tab}
            onClick={() => setActive(tab)}
            style={{
              fontFamily: 'IBM Plex Mono, monospace',
              fontSize: '11px',
              letterSpacing: '0.08em',
              background: active === tab ? 'var(--surface)' : 'transparent',
              color: active === tab ? 'var(--bright)' : 'var(--subtle)',
              border: '1px solid',
              borderColor: active === tab ? 'var(--rim)' : 'transparent',
              borderBottom: active === tab ? `1px solid var(--surface)` : '1px solid transparent',
            }}
            className="px-4 py-2 rounded-t transition-colors hover:text-[var(--bright)]"
          >
            {tab.toUpperCase()}
          </button>
        ))}
      </nav>

      {/* Tab content */}
      <main className="px-8 py-6">
        {active === 'Reconstruct' && <Reconstruct />}
        {active === 'Search'      && <Search />}
        {active === 'Explore'     && <Explore />}
        {active === 'Generate'    && <Generate />}
      </main>
    </div>
  )
}
