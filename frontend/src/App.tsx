import { useState, useRef, useEffect } from 'react'
import { useApolloClient, gql, useQuery, useMutation } from '@apollo/client'
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

// Single source of truth for adapter state: list + active
const ADAPTERS_QUERY = gql`
  query Adapters {
    adapters {
      available
      active
    }
  }
`

const FINE_TUNE_STATUS = gql`
  query FineTuneStatus {
    fineTuneStatus {
      running
      step
      totalSteps
      loss
      error
      completedAdapter
    }
  }
`

const START_FINE_TUNE = gql`
  mutation StartFineTune(
    $adapterName: String
    $datasetName: String
    $numSamples: Int
    $numEpochs: Int
    $learningRate: Float
    $loraR: Int
    $loraAlpha: Int
  ) {
    startFineTune(
      adapterName: $adapterName
      datasetName: $datasetName
      numSamples: $numSamples
      numEpochs: $numEpochs
      learningRate: $learningRate
      loraR: $loraR
      loraAlpha: $loraAlpha
    ) {
      ok
      error
    }
  }
`

const LOAD_ADAPTER = gql`
  mutation LoadAdapter($adapterName: String!) {
    loadAdapter(adapterName: $adapterName) {
      ok
      error
      activeAdapter
    }
  }
`

const USE_BASE_MODEL = gql`
  mutation UseBaseModel {
    useBaseModel {
      ok
      activeAdapter
    }
  }
`

type Tab = 'chat' | 'finetune' | 'explorer'
type Message = { role: 'user' | 'assistant'; text: string; error?: boolean; searched?: boolean }

const GQL_HTTP = import.meta.env.VITE_GRAPHQL_URL    || 'http://localhost:8080/graphql'
const GQL_WS   = import.meta.env.VITE_GRAPHQL_WS_URL || 'ws://localhost:8080/graphql/ws'
const fetcher  = createGraphiQLFetcher({ url: GQL_HTTP, subscriptionUrl: GQL_WS })

// ── Chat tab ──────────────────────────────────────────────────────────────────

function ChatTab() {
  const [prompt, setPrompt]     = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [webSearch, setWebSearch] = useState(false)
  const [loading, setLoading]   = useState(false)
  const client    = useApolloClient()
  const bottomRef = useRef<HTMLDivElement>(null)
  const subRef    = useRef<{ unsubscribe(): void } | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

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
              borderRight: (!m.error && m.role === 'assistant' && i === messages.length - 1 && loading)
                ? '2px solid #6b7280' : 'none',
            }}>
              {m.text || (m.role === 'assistant' && loading && i === messages.length - 1 ? '\u00A0' : '')}
            </div>
          </div>
        ))}

        {loading && messages[messages.length - 1]?.text === '' && (
          <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
            <div style={{ padding: '0.55rem 0.9rem', borderRadius: '16px 16px 16px 4px', background: '#f3f4f6', color: '#6b7280', fontSize: 14 }}>
              {webSearch ? '🌐 searching…' : <span style={{ letterSpacing: 2 }}>•••</span>}
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

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

// ── Fine-tune tab ─────────────────────────────────────────────────────────────

function FineTuneTab() {
  const [adapterName, setAdapterName]     = useState('my-lora')
  const [dataset, setDataset]             = useState('tatsu-lab/alpaca')
  const [numSamples, setNumSamples]       = useState(500)
  const [numEpochs, setNumEpochs]         = useState(1)
  const [learningRate, setLearningRate]   = useState(0.0002)
  const [loraR, setLoraR]                 = useState(8)
  const [loraAlpha, setLoraAlpha]         = useState(16)
  const [msg, setMsg]                     = useState('')

  // Fetch adapter list fresh on every tab mount; refetch after mutations
  const { data: adaptersData, refetch: refetchAdapters } = useQuery(ADAPTERS_QUERY, {
    fetchPolicy: 'network-only',
  })
  const available: string[] = adaptersData?.adapters?.available ?? []
  const activeAdapter: string | null = adaptersData?.adapters?.active ?? null

  const { data: statusData, startPolling, stopPolling } = useQuery(FINE_TUNE_STATUS, {
    fetchPolicy: 'network-only',
  })
  const [startFineTune] = useMutation(START_FINE_TUNE)
  const [loadAdapter]   = useMutation(LOAD_ADAPTER)
  const [useBaseModel]  = useMutation(USE_BASE_MODEL)

  const status = statusData?.fineTuneStatus

  useEffect(() => {
    if (status?.running) {
      startPolling(2000)
    } else {
      stopPolling()
    }
  }, [status?.running, startPolling, stopPolling])

  // Refresh adapter list when training finishes (new adapter appeared on disk)
  useEffect(() => {
    if (status?.completedAdapter) {
      refetchAdapters()
    }
  }, [status?.completedAdapter, refetchAdapters])

  const handleStart = async () => {
    setMsg('')
    const res = await startFineTune({
      variables: { adapterName, datasetName: dataset, numSamples, numEpochs, learningRate, loraR, loraAlpha },
    })
    const result = res.data?.startFineTune
    if (!result?.ok) {
      setMsg(`Error: ${result?.error ?? 'unknown'}`)
    } else {
      setMsg('Fine-tuning started!')
      startPolling(2000)
    }
  }

  const handleLoad = async (name: string) => {
    setMsg('')
    const res = await loadAdapter({ variables: { adapterName: name } })
    const result = res.data?.loadAdapter
    if (result?.ok) {
      await refetchAdapters()   // go straight to server, no cache games
      setMsg('')
    } else {
      setMsg(`Error: ${result?.error ?? 'unknown'}`)
    }
  }

  const handleUnload = async () => {
    setMsg('')
    await useBaseModel()
    await refetchAdapters()
  }

  const pct = status?.totalSteps
    ? Math.round((status.step / status.totalSteps) * 100)
    : 0

  const inputStyle: React.CSSProperties = {
    padding: '0.45rem 0.7rem',
    borderRadius: 8,
    border: '1px solid #d1d5db',
    fontSize: 13,
    width: '100%',
    boxSizing: 'border-box',
  }
  const labelStyle: React.CSSProperties = { fontSize: 12, color: '#6b7280', marginBottom: 3, display: 'block' }

  return (
    <div style={{ padding: '1.5rem', maxWidth: 680, margin: '0 auto', overflowY: 'auto', height: 'calc(100vh - 49px)', boxSizing: 'border-box' }}>

      {/* ── Adapter list ── */}
      <div style={{ marginBottom: 28 }}>
        <h3 style={{ fontSize: 14, fontWeight: 600, margin: '0 0 10px' }}>Models</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>

          {/* Base model card */}
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '0.6rem 0.9rem', borderRadius: 10,
            border: `1.5px solid ${activeAdapter === null ? '#0070f3' : '#e5e7eb'}`,
            background: activeAdapter === null ? '#eff6ff' : '#fafafa',
          }}>
            <div>
              <span style={{ fontSize: 13, fontWeight: 500 }}>Base model</span>
              {activeAdapter === null && (
                <span style={{ marginLeft: 8, fontSize: 11, color: '#0070f3', fontWeight: 600 }}>ACTIVE</span>
              )}
            </div>
            {activeAdapter !== null && (
              <button onClick={handleUnload} style={{ padding: '0.3rem 0.75rem', borderRadius: 20, border: '1px solid #d1d5db', background: '#fff', fontSize: 12, cursor: 'pointer' }}>
                Switch to base
              </button>
            )}
          </div>

          {/* Saved adapter cards */}
          {available.map(name => (
            <div key={name} style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '0.6rem 0.9rem', borderRadius: 10,
              border: `1.5px solid ${activeAdapter === name ? '#0070f3' : '#e5e7eb'}`,
              background: activeAdapter === name ? '#eff6ff' : '#fafafa',
            }}>
              <div>
                <span style={{ fontSize: 13, fontWeight: 500 }}>LoRA: {name}</span>
                {activeAdapter === name && (
                  <span style={{ marginLeft: 8, fontSize: 11, color: '#0070f3', fontWeight: 600 }}>ACTIVE</span>
                )}
              </div>
              {activeAdapter !== name ? (
                <button
                  onClick={() => handleLoad(name)}
                  style={{ padding: '0.3rem 0.75rem', borderRadius: 20, border: 'none', background: '#0070f3', color: '#fff', fontSize: 12, cursor: 'pointer', fontWeight: 500 }}
                >
                  Load
                </button>
              ) : (
                <button onClick={handleUnload} style={{ padding: '0.3rem 0.75rem', borderRadius: 20, border: '1px solid #d1d5db', background: '#fff', fontSize: 12, cursor: 'pointer' }}>
                  Unload
                </button>
              )}
            </div>
          ))}

          {available.length === 0 && (
            <p style={{ fontSize: 13, color: '#9ca3af', margin: 0 }}>No adapters trained yet.</p>
          )}
        </div>
      </div>

      {/* ── Training form ── */}
      <div style={{ marginBottom: 24 }}>
        <h3 style={{ fontSize: 14, fontWeight: 600, margin: '0 0 12px' }}>Train new LoRA adapter</h3>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
          <div>
            <label style={labelStyle}>Adapter name</label>
            <input style={inputStyle} value={adapterName} onChange={e => setAdapterName(e.target.value)} />
          </div>
          <div>
            <label style={labelStyle}>Dataset (HuggingFace ID)</label>
            <input style={inputStyle} value={dataset} onChange={e => setDataset(e.target.value)} />
          </div>
          <div>
            <label style={labelStyle}>Samples</label>
            <input style={inputStyle} type="number" min={50} max={50000} value={numSamples} onChange={e => setNumSamples(+e.target.value)} />
          </div>
          <div>
            <label style={labelStyle}>Epochs</label>
            <input style={inputStyle} type="number" min={1} max={10} value={numEpochs} onChange={e => setNumEpochs(+e.target.value)} />
          </div>
          <div>
            <label style={labelStyle}>Learning rate</label>
            <input style={inputStyle} type="number" step={0.00001} value={learningRate} onChange={e => setLearningRate(+e.target.value)} />
          </div>
          <div>
            <label style={labelStyle}>LoRA r / alpha</label>
            <div style={{ display: 'flex', gap: 6 }}>
              <input style={{ ...inputStyle, width: '50%' }} type="number" min={1} max={64} value={loraR} onChange={e => setLoraR(+e.target.value)} placeholder="r" />
              <input style={{ ...inputStyle, width: '50%' }} type="number" min={1} max={128} value={loraAlpha} onChange={e => setLoraAlpha(+e.target.value)} placeholder="alpha" />
            </div>
          </div>
        </div>

        <button
          onClick={handleStart}
          disabled={status?.running}
          style={{
            padding: '0.55rem 1.25rem', borderRadius: 20, border: 'none',
            background: status?.running ? '#93c5fd' : '#0070f3',
            color: '#fff', cursor: status?.running ? 'not-allowed' : 'pointer',
            fontSize: 14, fontWeight: 500,
          }}
        >
          {status?.running ? 'Training…' : 'Start fine-tuning'}
        </button>
      </div>

      {/* ── Training progress ── */}
      {(status?.running || status?.error) && (
        <div style={{ marginBottom: 24, padding: '1rem', background: '#f9fafb', borderRadius: 10, border: '1px solid #e5e7eb' }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, margin: '0 0 10px' }}>Training progress</h3>

          {status?.running && (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#6b7280', marginBottom: 4 }}>
                <span>Step {status.step} / {status.totalSteps}</span>
                <span>{pct}%{status.loss != null ? ` — loss: ${status.loss}` : ''}</span>
              </div>
              <div style={{ background: '#e5e7eb', borderRadius: 4, height: 8 }}>
                <div style={{ background: '#0070f3', borderRadius: 4, height: 8, width: `${pct}%`, transition: 'width 0.4s' }} />
              </div>
            </>
          )}

          {status?.error && (
            <p style={{ color: '#b91c1c', fontSize: 13, margin: 0 }}>Error: {status.error}</p>
          )}
        </div>
      )}

      {msg && (
        <p style={{ fontSize: 13, color: '#b91c1c', margin: 0 }}>{msg}</p>
      )}
    </div>
  )
}

// ── Root ──────────────────────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab] = useState<Tab>('chat')

  // Poll adapter state for the header pill (independent of FineTuneTab)
  const { data: adaptersData } = useQuery(ADAPTERS_QUERY, {
    pollInterval: 5000,
    fetchPolicy: 'network-only',
  })
  const activeAdapter: string | null = adaptersData?.adapters?.active ?? null

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
      <div style={{ borderBottom: '1px solid #e5e7eb', display: 'flex', alignItems: 'center', padding: '0 0.5rem', flexShrink: 0, gap: '0.25rem' }}>
        <span style={{ fontWeight: 700, fontSize: 14, padding: '0 0.75rem', color: '#111827', flexShrink: 0 }}>
          Inference Server
        </span>
        <div style={{ width: 1, height: 20, background: '#e5e7eb', margin: '0 0.25rem', flexShrink: 0 }} />
        {tabBtn('chat', 'Chat')}
        {tabBtn('finetune', 'Fine-tune')}
        {tabBtn('explorer', 'GraphQL Explorer')}

        <div style={{ flex: 1 }} />
        <span style={{
          padding: '0.25rem 0.65rem',
          borderRadius: 20,
          background: activeAdapter ? '#ecfdf5' : '#f3f4f6',
          color: activeAdapter ? '#065f46' : '#6b7280',
          fontSize: 12,
          fontWeight: 500,
          border: `1px solid ${activeAdapter ? '#a7f3d0' : '#e5e7eb'}`,
          marginRight: '0.5rem',
          flexShrink: 0,
        }}>
          {activeAdapter ? `LoRA: ${activeAdapter}` : 'base model'}
        </span>
      </div>

      {tab === 'chat'     && <ChatTab />}
      {tab === 'finetune' && <FineTuneTab />}
      {tab === 'explorer' && (
        <div style={{ flex: 1, overflow: 'hidden' }}>
          <GraphiQL fetcher={fetcher} />
        </div>
      )}
    </div>
  )
}
