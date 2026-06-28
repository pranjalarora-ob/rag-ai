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

interface Chat {
  id: string
  title: string
  messages: Message[]
  updatedAt: number
}

interface ChartPoint {
  name: string
  value: number
  label: string
}

// ── Constants ─────────────────────────────────────────────────────────────────

const CUSTOMER_ID = '7768d491-9dd4-4ac0-b85f-3ff68fc7d87e'
const API_URL = '/rag/planner/stream'
const STORAGE_KEY = 'pi_chats_v1'
const MAX_SAVED_CHATS = 30
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

// Format a rupee amount compactly; falls back to the raw string for un-parseable cells.
function fmtCurrency(raw: string): string {
  const n = parseFloat((raw || '').replace(/[^\d.eE+\-]/g, ''))
  if (!Number.isFinite(n) || n <= 0) return raw
  if (n >= 1e7) return `₹${(n / 1e7).toFixed(2)} Cr`
  if (n >= 1e5) return `₹${(n / 1e5).toFixed(2)} L`
  return `₹${n.toLocaleString('en-IN')}`
}

// ── Chat persistence (localStorage; swap for an API later) ──────────────────────

function loadChats(): Chat[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const chats = JSON.parse(raw) as Chat[]
    // Don't persist transient loading flags between sessions.
    return chats.map(c => ({ ...c, messages: c.messages.map(m => ({ ...m, loading: false })) }))
  } catch { return [] }
}

function saveChats(chats: Chat[]) {
  try {
    const trimmed = chats.slice(0, MAX_SAVED_CHATS)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed))
  } catch { /* quota / disabled storage — ignore */ }
}

function chatTitle(text: string): string {
  const t = text.trim().replace(/\s+/g, ' ')
  return t.length > 42 ? t.slice(0, 42) + '…' : t
}

// A markdown table extracted from an assistant message, with any prose around it.
interface ParsedTable {
  before: string
  after: string
  headers: string[]
  rows: string[][]
}

function parseTableBlock(content: string): ParsedTable | null {
  const lines = content.split('\n')
  const tableIdx = lines.findIndex(l => l.trim().startsWith('|'))
  if (tableIdx === -1) return null

  let end = tableIdx
  while (end < lines.length && lines[end].trim().startsWith('|')) end++

  const tableLines = lines.slice(tableIdx, end)
  if (tableLines.length < 3) return null

  const split = (l: string) => l.split('|').map(c => c.trim()).filter((_, i, a) => i > 0 && i < a.length - 1)
  const headers = split(tableLines[0])
  const rows = tableLines.slice(2).map(split).filter(r => r.length === headers.length)
  if (headers.length < 2 || rows.length < 1) return null

  return {
    before: lines.slice(0, tableIdx).join('\n').trim(),
    after: lines.slice(end).join('\n').trim(),
    headers,
    rows,
  }
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

  const data = removeOutliers(raw_data).slice(0, 20)

  return data.length >= 2 ? { data, yLabel } : null
}

// Drop garbage DB values that span many orders of magnitude (e.g. 6e58 alongside
// 9e9). Detection runs in LOG space using the Median Absolute Deviation, which —
// unlike IQR on raw values — stays robust even when several junk values exist at
// different magnitudes. A point is kept if its log10 is within `threshold` MADs
// of the median log10.
function removeOutliers(points: ChartPoint[], threshold = 3): ChartPoint[] {
  if (points.length < 3) return points

  const median = (arr: number[]) => {
    const s = [...arr].sort((a, b) => a - b)
    const mid = Math.floor(s.length / 2)
    return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2
  }

  const logs = points.map(p => Math.log10(p.value))
  const medLog = median(logs)
  const mad = median(logs.map(l => Math.abs(l - medLog)))
  // No spread (all similar magnitude) → nothing to trim.
  if (mad === 0) return points

  const kept = points.filter((_, i) => Math.abs(logs[i] - medLog) <= threshold * mad)
  // Never return fewer than 2 points if we started with more — fall back to the
  // tightest cluster around the median rather than an empty chart.
  return kept.length >= 2 ? kept : points
}

// ── ChartView ─────────────────────────────────────────────────────────────────

function ChartView({ data, yLabel }: { data: ChartPoint[], yLabel: string }) {
  const [type, setType] = useState<'bar' | 'hbar' | 'pie'>('hbar')

  const compact = (v: number) => {
    if (v === 0) return '0'
    const abs = Math.abs(v)
    if (abs >= 1e12) return `${(v / 1e12).toFixed(1)}T`
    if (abs >= 1e9)  return `${(v / 1e9).toFixed(1)}B`
    if (abs >= 1e6)  return `${(v / 1e6).toFixed(1)}M`
    if (abs >= 1e3)  return `${(v / 1e3).toFixed(1)}K`
    return String(v)
  }

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
          <button className={`ctype-btn ${type === 'hbar' ? 'on' : ''}`} onClick={() => setType('hbar')}>H-Bar</button>
          <button className={`ctype-btn ${type === 'bar' ? 'on' : ''}`} onClick={() => setType('bar')}>Bar</button>
          <button className={`ctype-btn ${type === 'pie' ? 'on' : ''}`} onClick={() => setType('pie')}>Pie</button>
        </div>
      </div>

      {type === 'hbar' ? (
        <ResponsiveContainer width="100%" height={Math.max(180, data.length * 34 + 40)}>
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#242b38" horizontal={false} />
            <XAxis type="number" tick={{ fill: '#8892a4', fontSize: 11 }} tickFormatter={compact} />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fill: '#8892a4', fontSize: 11 }}
              width={120}
              interval={0}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(110,168,254,0.08)' }} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {data.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      ) : type === 'bar' ? (
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data} margin={{ top: 4, right: 12, left: 8, bottom: 56 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#242b38" />
            <XAxis dataKey="name" tick={{ fill: '#8892a4', fontSize: 11 }} angle={-32} textAnchor="end" interval={0} />
            <YAxis tick={{ fill: '#8892a4', fontSize: 11 }} width={56} tickFormatter={compact} />
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

// ── DataTable ─────────────────────────────────────────────────────────────────

type ColKind = 'currency' | 'area' | 'index' | 'text'

function colKind(header: string): ColKind {
  if (/^#$/.test(header.trim())) return 'index'
  if (/value|cost|budget|worth|price|amount/i.test(header)) return 'currency'
  if (/area|sqft|sq\.?\s?ft/i.test(header)) return 'area'
  return 'text'
}

function cellNumber(raw: string): number {
  const n = parseFloat((raw || '').replace(/[^\d.eE+\-]/g, ''))
  return Number.isFinite(n) ? n : NaN
}

function DataTable({ headers, rows }: { headers: string[]; rows: string[][] }) {
  const [view, setView] = useState<'table' | 'cards'>('table')
  const [sortCol, setSortCol] = useState<number | null>(null)
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  const kinds = useMemo(() => headers.map(colKind), [headers])
  const valueIdx = kinds.indexOf('currency')

  // KPI cards computed from the currency column (if any)
  const kpi = useMemo(() => {
    if (valueIdx === -1) return null
    const nums = rows.map(r => cellNumber(r[valueIdx])).filter(n => Number.isFinite(n) && n > 0)
    if (!nums.length) return null
    const total = nums.reduce((a, b) => a + b, 0)
    return { count: rows.length, total, avg: total / nums.length }
  }, [rows, valueIdx])

  const sortedRows = useMemo(() => {
    if (sortCol === null) return rows
    const kind = kinds[sortCol]
    const copy = [...rows]
    copy.sort((a, b) => {
      const av = a[sortCol] ?? '', bv = b[sortCol] ?? ''
      if (kind === 'currency' || kind === 'area' || kind === 'index') {
        const an = cellNumber(av), bn = cellNumber(bv)
        return (Number.isFinite(an) ? an : -Infinity) - (Number.isFinite(bn) ? bn : -Infinity)
      }
      return av.localeCompare(bv)
    })
    if (sortDir === 'desc') copy.reverse()
    return copy
  }, [rows, sortCol, sortDir, kinds])

  const onSort = (i: number) => {
    if (sortCol === i) setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortCol(i); setSortDir('desc') }
  }

  const fmtCell = (raw: string, kind: ColKind) =>
    kind === 'currency' ? fmtCurrency(raw)
      : kind === 'area' && cellNumber(raw) ? `${cellNumber(raw).toLocaleString('en-IN')} sqft`
      : raw

  const exportCSV = () => {
    const esc = (s: string) => `"${(s || '').replace(/"/g, '""')}"`
    const csv = [headers.map(esc).join(','), ...sortedRows.map(r => r.map(esc).join(','))].join('\n')
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
    const a = document.createElement('a')
    a.href = url; a.download = 'projects.csv'; a.click()
    URL.revokeObjectURL(url)
  }

  const exportPDF = () => {
    const w = window.open('', '_blank')
    if (!w) return
    const th = headers.map(h => `<th>${h}</th>`).join('')
    const tr = sortedRows.map(r => `<tr>${r.map((c, i) => `<td>${fmtCell(c, kinds[i])}</td>`).join('')}</tr>`).join('')
    w.document.write(`<!doctype html><html><head><title>Projects</title>
      <style>body{font-family:system-ui,sans-serif;padding:28px;color:#111}
      h1{font-size:18px;margin:0 0 4px}p{color:#666;font-size:12px;margin:0 0 18px}
      table{border-collapse:collapse;width:100%;font-size:12px}
      th,td{border:1px solid #ddd;padding:8px 10px;text-align:left}
      th{background:#f4f5f7;text-transform:uppercase;font-size:10px;letter-spacing:.04em}</style>
      </head><body><h1>Project Intelligence — Export</h1><p>${sortedRows.length} record(s)</p>
      <table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table>
      <script>window.onload=()=>{window.print()}</script></body></html>`)
    w.document.close()
  }

  return (
    <div className="dt">
      {kpi && (
        <div className="kpi-row">
          <div className="kpi"><span className="kpi-label">Results</span><span className="kpi-value">{kpi.count}</span></div>
          <div className="kpi"><span className="kpi-label">Total {headers[valueIdx]}</span><span className="kpi-value">{fmtNum(kpi.total, false)}</span></div>
          <div className="kpi"><span className="kpi-label">Average</span><span className="kpi-value">{fmtNum(kpi.avg, false)}</span></div>
        </div>
      )}

      <div className="dt-toolbar">
        <div className="dt-views">
          <button className={`dt-vbtn ${view === 'table' ? 'on' : ''}`} onClick={() => setView('table')}>Table</button>
          <button className={`dt-vbtn ${view === 'cards' ? 'on' : ''}`} onClick={() => setView('cards')}>Cards</button>
        </div>
        <div className="dt-actions">
          <button className="dt-act" onClick={exportCSV}>↓ CSV</button>
          <button className="dt-act" onClick={exportPDF}>↓ PDF</button>
        </div>
      </div>

      {view === 'table' ? (
        <div className="dt-scroll">
          <table className="dt-table">
            <thead>
              <tr>
                {headers.map((h, i) => (
                  <th key={i} onClick={() => onSort(i)} className={sortCol === i ? 'sorted' : ''}>
                    {h}
                    <span className="dt-arrow">{sortCol === i ? (sortDir === 'asc' ? '↑' : '↓') : '↕'}</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedRows.map((r, ri) => (
                <tr key={ri}>
                  {r.map((c, ci) => (
                    <td key={ci} className={kinds[ci] === 'currency' || kinds[ci] === 'area' ? 'num' : ''}>
                      {fmtCell(c, kinds[ci])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="dt-cards">
          {sortedRows.map((r, ri) => (
            <div className="dt-card" key={ri}>
              {r.map((c, ci) => kinds[ci] === 'index' ? null : (
                <div className="dt-card-row" key={ci}>
                  <span className="dt-card-key">{headers[ci]}</span>
                  <span className={`dt-card-val ${kinds[ci] === 'currency' ? 'accent' : ''}`}>{fmtCell(c, kinds[ci])}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Skeleton ──────────────────────────────────────────────────────────────────

function ResultSkeleton() {
  return (
    <div className="skel">
      <div className="skel-kpis">
        <div className="skel-box sk-shimmer" />
        <div className="skel-box sk-shimmer" />
        <div className="skel-box sk-shimmer" />
      </div>
      <div className="skel-table">
        <div className="skel-row sk-head sk-shimmer" />
        {Array.from({ length: 4 }).map((_, i) => (
          <div className="skel-row sk-shimmer" key={i} />
        ))}
      </div>
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

  const table = useMemo(() => {
    if (msg.role !== 'assistant' || msg.loading) return null
    return parseTableBlock(msg.content)
  }, [msg.content, msg.role, msg.loading])

  return (
    <div className={`message message--${msg.role}`}>
      <div className="avatar">{msg.role === 'user' ? 'U' : '◈'}</div>
      <div className="bubble">
        {msg.loading && !msg.content ? (
          <ResultSkeleton />
        ) : table ? (
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
            {table.before && (
              <div className="md"><ReactMarkdown remarkPlugins={[remarkGfm]}>{table.before}</ReactMarkdown></div>
            )}
            {showChart && chartData
              ? <ChartView data={chartData.data} yLabel={chartData.yLabel} />
              : <DataTable headers={table.headers} rows={table.rows} />}
            {table.after && (
              <div className="md" style={{ marginTop: 10 }}><ReactMarkdown remarkPlugins={[remarkGfm]}>{table.after}</ReactMarkdown></div>
            )}
          </>
        ) : (
          <div className="md">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
            {msg.loading && <span className="stream-cursor" />}
          </div>
        )}
      </div>
    </div>
  )
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [chats, setChats] = useState<Chat[]>(loadChats)
  const [activeId, setActiveId] = useState<string | null>(() => loadChats()[0]?.id ?? null)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const activeChat = chats.find(c => c.id === activeId) ?? null
  const messages = activeChat?.messages ?? []

  useEffect(() => { saveChats(chats) }, [chats])
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])
  useEffect(() => {
    const el = inputRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }, [input])

  // Patch the messages of a specific chat and bump it to the top of the list.
  const patchChat = (chatId: string, updater: (msgs: Message[]) => Message[]) => {
    setChats(prev => {
      const next = prev.map(c =>
        c.id === chatId ? { ...c, messages: updater(c.messages), updatedAt: Date.now() } : c
      )
      next.sort((a, b) => b.updatedAt - a.updatedAt)
      return next
    })
  }

  const newChat = () => {
    setActiveId(null)
    setInput('')
    inputRef.current?.focus()
  }

  const deleteChat = (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setChats(prev => prev.filter(c => c.id !== id))
    if (activeId === id) setActiveId(null)
  }

  const send = async (question?: string) => {
    const q = (question ?? input).trim()
    if (!q || loading) return
    setInput('')
    setLoading(true)

    // Resolve (or create) the chat this message belongs to.
    let chatId: string
    if (!activeId || !chats.some(c => c.id === activeId)) {
      chatId = (crypto as any).randomUUID?.() ?? String(Date.now())
      const fresh: Chat = { id: chatId, title: chatTitle(q), messages: [], updatedAt: Date.now() }
      setChats(prev => [fresh, ...prev])
      setActiveId(chatId)
    } else {
      chatId = activeId
    }

    // Prior turns become context for the planner (skip empty/loading bubbles).
    const history = messages
      .filter(m => m.content && !m.loading)
      .map(m => ({ role: m.role, content: m.content }))

    const userMsg: Message = { id: ++msgId, role: 'user', content: q }
    const botId = ++msgId
    const botMsg: Message = { id: botId, role: 'assistant', content: '', loading: true }

    patchChat(chatId, msgs => [...msgs, userMsg, botMsg])

    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, customerId: CUSTOMER_ID, history }),
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
          patchChat(chatId, msgs => msgs.map(m => m.id === botId ? { ...m, content: acc, loading: false } : m))
        }
        if (!acc) throw new Error('Empty response')
      } else {
        const text = await res.text()
        patchChat(chatId, msgs => msgs.map(m => m.id === botId ? { ...m, content: text, loading: false } : m))
      }
    } catch (err: any) {
      patchChat(chatId, msgs => msgs.map(m =>
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
    <div className={`shell ${sidebarOpen ? '' : 'shell--collapsed'}`}>
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <button className="new-chat-btn" onClick={newChat}>
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
            <path d="M8 3.5v9M3.5 8h9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
          New chat
        </button>

        <div className="chat-list">
          {chats.length === 0 ? (
            <p className="chat-list-empty">No conversations yet</p>
          ) : chats.map(c => (
            <div
              key={c.id}
              className={`chat-item ${c.id === activeId ? 'on' : ''}`}
              onClick={() => setActiveId(c.id)}
            >
              <svg className="chat-item-icon" width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
                <path d="M2 3a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1H6l-3 3v-3H3a1 1 0 0 1-1-1V3Z" />
              </svg>
              <span className="chat-item-title">{c.title}</span>
              <button className="chat-item-del" onClick={(e) => deleteChat(c.id, e)} aria-label="Delete chat">×</button>
            </div>
          ))}
        </div>

        <div className="sidebar-foot">Stored locally on this device</div>
      </aside>

      {/* ── Main ── */}
      <div className="app">
        <header className="header">
          <div className="header-logo">
            <button className="sidebar-toggle" onClick={() => setSidebarOpen(v => !v)} aria-label="Toggle sidebar">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
              </svg>
            </button>
            <span className="header-icon">◈</span>
            Project Intelligence
          </div>
          <span className="header-badge">Online</span>
        </header>

        <main className="chat-area">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">◈</div>
              <h2>What would you like to know?</h2>
              <p>Ask anything about your projects — filter by city, cost, area, team, stage, and more.</p>
              <div className="suggestions">
                {SUGGESTIONS.map((s, i) => (
                  <button key={i} className="suggestion" onClick={() => send(s)}><span>{s}</span></button>
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
          <p className="hint"><kbd>Enter</kbd> to send · <kbd>Shift</kbd> + <kbd>Enter</kbd> for new line</p>
        </footer>
      </div>
    </div>
  )
}
