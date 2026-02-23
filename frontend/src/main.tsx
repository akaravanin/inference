import React from 'react'
import ReactDOM from 'react-dom/client'
import { ApolloClient, InMemoryCache, ApolloProvider, HttpLink, split } from '@apollo/client'
import { GraphQLWsLink } from '@apollo/client/link/subscriptions'
import { getMainDefinition } from '@apollo/client/utilities'
import { createClient } from 'graphql-ws'
import App from './App'

const GQL_HTTP = import.meta.env.VITE_GRAPHQL_URL    || 'http://localhost:8080/graphql'
const GQL_WS   = import.meta.env.VITE_GRAPHQL_WS_URL || 'ws://localhost:8080/graphql/ws'

const httpLink = new HttpLink({ uri: GQL_HTTP })

const wsLink = new GraphQLWsLink(createClient({ url: GQL_WS }))

// Subscriptions → WebSocket, everything else → HTTP
const link = split(
  ({ query }) => {
    const def = getMainDefinition(query)
    return def.kind === 'OperationDefinition' && def.operation === 'subscription'
  },
  wsLink,
  httpLink,
)

const client = new ApolloClient({ link, cache: new InMemoryCache() })

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ApolloProvider client={client}>
      <App />
    </ApolloProvider>
  </React.StrictMode>,
)
