import { useState, useRef, useEffect } from 'react'
import { useMutation, gql } from '@apollo/client'
import { GraphiQL } from 'graphiql'
import { createGraphiQLFetcher } from '@graphiql/toolkit'
import 'graphiql/graphiql.css'

const INFER = gql`
  mutation Infer($prompt: String!, $webSearch: Boolean) {
    infer(prompt: $prompt, webSearch: $webSearch) {
      text
    }
  }
`

type Tab = 'chat' | 'explorer'
type Message = { role: 'user' | 'assistant'; text: string; error?: boolean; searched?: boolean }

const GQL_URL = import.meta.env.VITE_GRAPHQL_URL || 'http://localhost:8080/graphql'
const fetcher = createGraphiQLFetcher({ url: GQL_URL })

// ── Chat tab ──────────────────────────────────────────────────────────────────

function ChatTab() {
  const [prompt, setPrompt] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [webSearch, setWebSearch] = useState(false)
  const [infer, { loading }] = useMutation(INFER)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const send = async () => {
    if (!prompt.trim() || loading) return
    const userText = prompt
    const didSearch = webSearch
    setMessages(m => [...m, { role: 'user', text: userText, searched: didSearch }])
    setPrompt('')
    try {
      const { data } = await infer({ variables: { prompt: userText, webSearch: didSearch } })
      setMessages(m => [...m, { role: 'assistant', text: data.infer.text }])
    } catch (err) {
      setMessages(m => [...m, { role: 'assistant', text: `Error: ${err}`, error: true }])
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 49px)' }}>

      {/* Message list */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '1.25rem 1rem', display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
        {messages.length === 0 && (
          <p style={{ color: '#9ca3af', textAlign: 'center', marginTop: '4rem', fontSize: 14 }}>
            Send a message to start inferencing.
          </p>
        )}

        {messages.map((m, i) => (
          <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: m.role === 'user' ? 'flex-end' : 'flex-start' }}>
            {m.searched && (
              <span style={{ fontSize: 11, color: '#6b7280', marginBottom: 3, display: 'flex', alignItems: 'center', gap: 3 }}>
                🌐 web search
              </span>
            )}
            <div style={{
              maxWidth: '70%',
              padding: '0.55rem 0.9rem',
              borderRadius: m.role === 'user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
              background: m.role === 'user' ? '#0070f3' : m.error ? '#fef2f2' : '#f3f4f6',
              color: m.role === 'user' ? '#fff' : m.error ? '#b91c1c' : '#111827',
              fontSize: 14,
              lineHeight: 1.55,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}>
              {m.text}
            </div>
          </div>
        ))}

        {loading && (
          <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
            <div style={{ padding: '0.55rem 0.9rem', borderRadius: '16px 16px 16px 4px', background: '#f3f4f6', color: '#6b7280', fontSize: 14 }}>
              {webSearch ? '🌐 searching…' : <span style={{ letterSpacing: 2 }}>•••</span>}
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div style={{ padding: '0.75rem 1rem', borderTop: '1px solid #e5e7eb', display: 'flex', gap: '0.5rem', alignItems: 'center', background: '#fff' }}>

        {/* Web search toggle */}
        <button
          onClick={() => setWebSearch(s => !s)}
          title={webSearch ? 'Web search on' : 'Web search off'}
          style={{
            padding: '0.45rem 0.75rem',
            borderRadius: 20,
            border: `1.5px solid ${webSearch ? '#0070f3' : '#d1d5db'}`,
            background: webSearch ? '#eff6ff' : '#fff',
            color: webSearch ? '#0070f3' : '#6b7280',
            cursor: 'pointer',
            fontSize: 13,
            fontWeight: 500,
            whiteSpace: 'nowrap',
            flexShrink: 0,
          }}
        >
          🌐 {webSearch ? 'On' : 'Off'}
        </button>

        <input
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
          placeholder={webSearch ? 'Ask anything — I\'ll search the web…' : 'Type a message…'}
          style={{ flex: 1, padding: '0.6rem 1rem', borderRadius: 20, border: '1px solid #d1d5db', fontSize: 14, outline: 'none' }}
        />

        <button
          onClick={send}
          disabled={loading}
          style={{
            padding: '0.6rem 1.25rem',
            borderRadius: 20,
            border: 'none',
            background: loading ? '#93c5fd' : '#0070f3',
            color: '#fff',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: 14,
            fontWeight: 500,
            flexShrink: 0,
          }}
        >
          Send
        </button>
      </div>
    </div>
  )
}

// ── Root ──────────────────────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab] = useState<Tab>('chat')

  const tabBtn = (t: Tab, label: string) => (
    <button
      onClick={() => setTab(t)}
      style={{
        padding: '0.6rem 1.1rem',
        border: 'none',
        borderBottom: tab === t ? '2px solid #0070f3' : '2px solid transparent',
        background: 'none',
        cursor: 'pointer',
        fontSize: 13,
        fontWeight: tab === t ? 600 : 400,
        color: tab === t ? '#0070f3' : '#6b7280',
        whiteSpace: 'nowrap',
      }}
    >
      {label}
    </button>
  )

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', height: '100vh', display: 'flex', flexDirection: 'column', background: '#fff' }}>

      {/* Header / tab bar */}
      <div style={{ borderBottom: '1px solid #e5e7eb', display: 'flex', alignItems: 'center', padding: '0 0.5rem', flexShrink: 0 }}>
        <span style={{ fontWeight: 700, fontSize: 14, padding: '0 0.75rem', color: '#111827' }}>
          Inference Server
        </span>
        <div style={{ width: 1, height: 20, background: '#e5e7eb', margin: '0 0.25rem' }} />
        {tabBtn('chat', 'Chat')}
        {tabBtn('explorer', 'GraphQL Explorer')}
      </div>

      {/* Tab content */}
      {tab === 'chat' && <ChatTab />}
      {tab === 'explorer' && (
        <div style={{ flex: 1, overflow: 'hidden' }}>
          <GraphiQL fetcher={fetcher} />
        </div>
      )}
    </div>
  )
}
