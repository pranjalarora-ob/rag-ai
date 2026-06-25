import { useState, useRef, useEffect, useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, CartesianGrid,
} from 'recharts'
import './App.css'

// ── Types ─────────────────────────────────────────────────────────────────────

interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  loading?: boolean
}

interface ChartPoint {
  name: string
  value: number
  label: string
}

// ── Constants ─────────────────────────────────────────────────────────────────

const CUSTOMER_ID = '7768d491-9dd4-4ac0-b85f-3ff68fc7d87e'
const API_URL = '/rag/openai/chat'
const SUGGESTIONS = [
  'List projects in Gurugram with estimated value > 2 cr',
  'How many projects are in Design-Sales stage?',
  'Show top 5 projects by estimated value',
  'Projects where area is more than 5000 sqft',
]

const CHART_COLORS = [
  '#6ea8fe', '#7ee0c0', '#f9a8d4', '#fcd34d',
  '#a78bfa', '#fb923c', '#34d399', '#60a5fa',
  '#f87171', '#e879f9',
]

let msgId = 0

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtNum(n: number, isArea: boolean): string {
  if (isArea) return `${n.toLocaleString()} sqft`
  if (n >= 1e7) return `₹${(n / 1e7).toFixed(2)} Cr`
  if (n >= 1e5) return `₹${(n / 1e5).toFixed(2)} L`
  return `₹${n.toLocaleString()}`
}

function parseTableForChart(content: string): { data: ChartPoint[], yLabel: string } | null {
  const tableLines = content.split('\n').filter(l => l.trim().startsWith('|'))
  if (tableLines.length < 3) return null

  const headers = tableLines[0].split('|').map(h => h.trim()).filter(Boolean)
  const dataRows = tableLines.slice(2).map(line =>
    line.split('|').map(c => c.trim()).filter(Boolean)
  )
  if (headers.length < 2 || dataRows.length < 2) return null

  const nameIdx = headers.findIndex(h => /name/i.test(h))
  if (nameIdx === -1) return null

  const valueIdx = headers.findIndex(h => /value/i.test(h))
  const areaIdx  = headers.findIndex(h => /area/i.test(h))
  const colIdx   = valueIdx !== -1 ? valueIdx : areaIdx
  if (colIdx === -1) return null

  const isArea = colIdx === areaIdx && valueIdx === -1
  const yLabel = isArea ? 'Area (sqft)' : 'Est. Value (₹ Cr)'

  const raw_data: ChartPoint[] = dataRows
    .map(cells => {
      const nameRaw = (cells[nameIdx] || '').replace(/^\d{10}-/, '').slice(0, 24)
      const raw = parseFloat((cells[colIdx] || '0').replace(/[^\d.e+\-]/g, ''))
      if (!Number.isFinite(raw) || raw <= 0) return null
      const value = isArea ? Math.round(raw) : Math.round(raw / 1e7 * 100) / 100
      return { name: nameRaw, value, label: fmtNum(raw, isArea) }
    })
    .filter((d): d is ChartPoint => d !== null && d.value > 0)

  // Remove statistical outliers using IQR so garbage DB values don't dominate
  const sorted = [...raw_data].sort((a, b) => a.value - b.value)
  const q1 = sorted[Math.floor(sorted.length * 0.25)].value
  const q3 = sorted[Math.floor(sorted.length * 0.75)].value
  const iqr = q3 - q1
  const fence = q3 + 3 * iqr  // 3× IQR to allow legitimate high-value outliers
  const data = raw_data.filter(d => d.value <= fence).slice(0, 20)

  return data.length >= 2 ? { data, yLabel } : null
}

// ── ChartView ─────────────────────────────────────────────────────────────────

function ChartView({ data, yLabel }: { data: ChartPoint[], yLabel: string }) {
  const [type, setType] = useState<'bar' | 'pie'>('bar')

  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload?.length) return null
    return (
      <div className="chart-tooltip">
        <p className="ct-name">{payload[0].payload.name}</p>
        <p className="ct-val">{payload[0].payload.label}</p>
      </div>
    )
  }

  return (
    <div className="chart-wrapper">
      <div className="chart-controls">
        <span className="chart-ylabel">{yLabel}</span>
        <div className="chart-type-btns">
          <button className={`ctype-btn ${type === 'bar' ? 'on' : ''}`} onClick={() => setType('bar')}>Bar</button>
          <button className={`ctype-btn ${type === 'pie' ? 'on' : ''}`} onClick={() => setType('pie')}>Pie</button>
        </div>
      </div>

      {type === 'bar' ? (
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data} margin={{ top: 4, right: 12, left: 8, bottom: 56 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#242b38" />
            <XAxis dataKey="name" tick={{ fill: '#8892a4', fontSize: 11 }} angle={-32} textAnchor="end" interval={0} />
            <YAxis
              tick={{ fill: '#8892a4', fontSize: 11 }}
              width={56}
              tickFormatter={(v: number) => {
                if (v === 0) return '0'
                const abs = Math.abs(v)
                if (abs >= 1e12) return `${(v / 1e12).toFixed(1)}T`
                if (abs >= 1e9)  return `${(v / 1e9).toFixed(1)}B`
                if (abs >= 1e6)  return `${(v / 1e6).toFixed(1)}M`
                if (abs >= 1e3)  return `${(v / 1e3).toFixed(1)}K`
                return String(v)
              }}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(110,168,254,0.08)' }} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {data.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              outerRadius={100}
              dataKey="value"
              labelLine={false}
              label={({ name, percent, cx: pcx, cy: pcy, midAngle, outerRadius: or }: any) => {
                if ((percent ?? 0) < 0.04) return null
                const rad = (or ?? 100) + 18
                const angle = (midAngle ?? 0) * Math.PI / 180
                const x = (pcx ?? 0) + rad * Math.cos(-angle)
                const y = (pcy ?? 0) + rad * Math.sin(-angle)
                return (
                  <text x={x} y={y} fill="#8892a4" fontSize={11} textAnchor={x > (pcx ?? 0) ? 'start' : 'end'} dominantBaseline="central">
                    {`${(name ?? '').slice(0, 10)} ${((percent ?? 0) * 100).toFixed(0)}%`}
                  </text>
                )
              }}
            >
              {data.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
          </PieChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

// ── MessageBubble ─────────────────────────────────────────────────────────────

function MessageBubble({ msg }: { msg: Message }) {
  const [showChart, setShowChart] = useState(false)
  const chartData = useMemo(() => {
    if (msg.role !== 'assistant' || msg.loading) return null
    return parseTableForChart(msg.content)
  }, [msg.content, msg.role, msg.loading])

  return (
    <div className={`message message--${msg.role}`}>
      <div className="avatar">{msg.role === 'user' ? 'U' : '◈'}</div>
      <div className="bubble">
        {msg.loading ? (
          <span className="typing"><span /><span /><span /></span>
        ) : (
          <>
            {chartData && (
              <button
                className={`chart-toggle-btn ${showChart ? 'on' : ''}`}
                onClick={() => setShowChart(v => !v)}
              >
                <svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor">
                  <rect x="1" y="8" width="3" height="7" rx="1" />
                  <rect x="6" y="4" width="3" height="11" rx="1" />
                  <rect x="11" y="1" width="3" height="14" rx="1" />
                </svg>
                {showChart ? 'View Table' : 'View Chart'}
              </button>
            )}
            {showChart && chartData ? (
              <ChartView data={chartData.data} yLabel={chartData.yLabel} />
            ) : (
              <div className="md">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])
  useEffect(() => {
    const el = inputRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }, [input])

  const send = async (question?: string) => {
    const q = (question ?? input).trim()
    if (!q || loading) return
    setInput('')
    setLoading(true)

    const userMsg: Message = { id: ++msgId, role: 'user', content: q }
    const botId = ++msgId
    const botMsg: Message = { id: botId, role: 'assistant', content: '', loading: true }

    setMessages(prev => [...prev, userMsg, botMsg])

    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, customerId: CUSTOMER_ID }),
      })
      if (!res.ok) throw new Error(`Server error ${res.status}`)

      if (res.body) {
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let acc = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          acc += decoder.decode(value, { stream: true })
          setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: acc, loading: false } : m))
        }
        if (!acc) throw new Error('Empty response')
      } else {
        const text = await res.text()
        setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: text, loading: false } : m))
      }
    } catch (err: any) {
      setMessages(prev => prev.map(m =>
        m.id === botId ? { ...m, content: `⚠️ ${err.message}`, loading: false } : m
      ))
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-logo">
          <span className="header-icon">◈</span>
          Project Intelligence
        </div>
      </header>

      <main className="chat-area">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">◈</div>
            <h2>What would you like to know?</h2>
            <p>Ask anything about your projects — filter by city, cost, area, team, stage, and more.</p>
            <div className="suggestions">
              {SUGGESTIONS.map((s, i) => (
                <button key={i} className="suggestion" onClick={() => send(s)}>{s}</button>
              ))}
            </div>
          </div>
        ) : (
          <div className="messages">
            {messages.map(msg => <MessageBubble key={msg.id} msg={msg} />)}
            <div ref={bottomRef} />
          </div>
        )}
      </main>

      <footer className="input-bar">
        <div className="input-wrap">
          <textarea
            ref={inputRef}
            className="input"
            placeholder="Ask about your projects…"
            value={input}
            rows={1}
            onChange={e => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={loading}
          />
          <button className="send-btn" onClick={() => send()} disabled={loading || !input.trim()} aria-label="Send">
            {loading ? <span className="send-spinner" /> : (
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M8 13V3M3 8l5-5 5 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            )}
          </button>
        </div>
        <p className="hint">Enter to send · Shift+Enter for new line</p>
      </footer>
    </div>
  )
}
