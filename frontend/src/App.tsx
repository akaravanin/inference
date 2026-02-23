import { useState, useRef, useEffect } from 'react'
import { useApolloClient, gql } from '@apollo/client'
import { GraphiQL } from 'graphiql'
import { createGraphiQLFetcher } from '@graphiql/toolkit'
import 'graphiql/graphiql.css'

const INFER_STREAM = gql`
  subscription InferStream($prompt: String!, $webSearch: Boolean) {
    inferStream(prompt: $prompt, webSearch: $webSearch) {
      token
      done
    }
  }
`

type Tab = 'chat' | 'explorer'
type Message = { role: 'user' | 'assistant'; text: string; error?: boolean; searched?: boolean }

const GQL_HTTP = import.meta.env.VITE_GRAPHQL_URL    || 'http://localhost:8080/graphql'
const GQL_WS   = import.meta.env.VITE_GRAPHQL_WS_URL || 'ws://localhost:8080/graphql/ws'
const fetcher  = createGraphiQLFetcher({ url: GQL_HTTP, subscriptionUrl: GQL_WS })

// ── Chat tab ──────────────────────────────────────────────────────────────────

function ChatTab() {
  const [prompt, setPrompt]   = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [webSearch, setWebSearch] = useState(false)
  const [loading, setLoading]   = useState(false)
  const client    = useApolloClient()
  const bottomRef = useRef<HTMLDivElement>(null)
  const subRef    = useRef<{ unsubscribe(): void } | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Clean up any active subscription on unmount
  useEffect(() => () => { subRef.current?.unsubscribe() }, [])

  const send = () => {
    if (!prompt.trim() || loading) return
    const userText  = prompt
    const didSearch = webSearch
    setMessages(m => [
      ...m,
      { role: 'user',      text: userText, searched: didSearch },
      { role: 'assistant', text: '' },
    ])
    setPrompt('')
    setLoading(true)

    const observable = client.subscribe({
      query: INFER_STREAM,
      variables: { prompt: userText, webSearch: didSearch },
    })

    subRef.current = observable.subscribe({
      next({ data }) {
        const { token, done } = data.inferStream
        if (done) {
          setLoading(false)
          subRef.current?.unsubscribe()
        } else {
          setMessages(m => {
            const copy = [...m]
            const last = copy[copy.length - 1]
            copy[copy.length - 1] = { ...last, text: last.text + token }
            return copy
          })
        }
      },
      error(err) {
        setMessages(m => {
          const copy = [...m]
          copy[copy.length - 1] = { role: 'assistant', text: `Error: ${err}`, error: true }
          return copy
        })
        setLoading(false)
      },
    })
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
              <span style={{ fontSize: 11, color: '#6b7280', marginBottom: 3 }}>🌐 web search</span>
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
              // Blinking cursor while the last assistant message is still streaming
              borderRight: (!m.error && m.role === 'assistant' && i === messages.length - 1 && loading)
                ? '2px solid #6b7280' : 'none',
            }}>
              {m.text || (m.role === 'assistant' && loading && i === messages.length - 1 ? '\u00A0' : '')}
            </div>
          </div>
        ))}

        {/* "Searching…" indicator before first token arrives */}
        {loading && messages[messages.length - 1]?.text === '' && (
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
            flexShrink: 0,
          }}
        >
          🌐 {webSearch ? 'On' : 'Off'}
        </button>

        <input
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
          placeholder={webSearch ? "Ask anything — I'll search the web…" : 'Type a message…'}
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
      <div style={{ borderBottom: '1px solid #e5e7eb', display: 'flex', alignItems: 'center', padding: '0 0.5rem', flexShrink: 0 }}>
        <span style={{ fontWeight: 700, fontSize: 14, padding: '0 0.75rem', color: '#111827' }}>
          Inference Server
        </span>
        <div style={{ width: 1, height: 20, background: '#e5e7eb', margin: '0 0.25rem' }} />
        {tabBtn('chat', 'Chat')}
        {tabBtn('explorer', 'GraphQL Explorer')}
      </div>

      {tab === 'chat' && <ChatTab />}
      {tab === 'explorer' && (
        <div style={{ flex: 1, overflow: 'hidden' }}>
          <GraphiQL fetcher={fetcher} />
        </div>
      )}
    </div>
  )
}
