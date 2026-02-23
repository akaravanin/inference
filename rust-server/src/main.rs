use async_graphql::{Context, Object, Schema, SimpleObject, Subscription};
use async_graphql_axum::{GraphQL, GraphQLSubscription};
use axum::Router;
use futures_util::{Stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;

// ── Types ─────────────────────────────────────────────────────────────────────

#[derive(SimpleObject)]
struct InferenceResult {
    text: String,
}

#[derive(SimpleObject, Clone)]
struct TokenChunk {
    token: String,
    done: bool,
}

#[derive(Serialize)]
struct WorkerRequest {
    prompt: String,
    web_search: bool,
}

#[derive(Deserialize)]
struct WorkerResponse {
    text: String,
}

fn worker_url() -> String {
    std::env::var("WORKER_URL").unwrap_or_else(|_| "http://python-worker:8001".to_string())
}

// ── Query ─────────────────────────────────────────────────────────────────────

struct QueryRoot;

#[Object]
impl QueryRoot {
    async fn health(&self) -> &str {
        "ok"
    }
}

// ── Mutation (non-streaming, kept for GraphQL Explorer) ───────────────────────

struct MutationRoot;

#[Object]
impl MutationRoot {
    async fn infer(
        &self,
        ctx: &Context<'_>,
        prompt: String,
        web_search: Option<bool>,
    ) -> async_graphql::Result<InferenceResult> {
        let client = ctx.data::<Client>()?;
        let resp: WorkerResponse = client
            .post(format!("{}/infer", worker_url()))
            .json(&WorkerRequest { prompt, web_search: web_search.unwrap_or(false) })
            .send()
            .await?
            .json()
            .await?;
        Ok(InferenceResult { text: resp.text })
    }
}

// ── Subscription (streaming) ──────────────────────────────────────────────────

struct SubscriptionRoot;

#[Subscription]
impl SubscriptionRoot {
    async fn infer_stream(
        &self,
        ctx: &Context<'_>,
        prompt: String,
        web_search: Option<bool>,
    ) -> impl Stream<Item = async_graphql::Result<TokenChunk>> {
        let client = ctx.data::<Client>().unwrap().clone();

        async_stream::stream! {
            let response = match client
                .post(format!("{}/stream", worker_url()))
                .json(&WorkerRequest { prompt, web_search: web_search.unwrap_or(false) })
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    yield Err(async_graphql::Error::new(e.to_string()));
                    return;
                }
            };

            let mut byte_stream = response.bytes_stream();
            let mut buf = String::new();

            while let Some(chunk) = byte_stream.next().await {
                match chunk {
                    Err(e) => {
                        yield Err(async_graphql::Error::new(e.to_string()));
                        return;
                    }
                    Ok(bytes) => {
                        buf.push_str(&String::from_utf8_lossy(&bytes));

                        // SSE frames are separated by double newlines
                        while let Some(pos) = buf.find("\n\n") {
                            let frame = buf[..pos].to_string();
                            buf = buf[pos + 2..].to_string();

                            if let Some(data) = frame.strip_prefix("data: ") {
                                if data.trim() == "[DONE]" {
                                    yield Ok(TokenChunk { token: String::new(), done: true });
                                    return;
                                }
                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                                    if let Some(token) = v["token"].as_str() {
                                        yield Ok(TokenChunk { token: token.to_string(), done: false });
                                    }
                                }
                            }
                        }
                    }
                }
            }

            yield Ok(TokenChunk { token: String::new(), done: true });
        }
    }
}

// ── App ───────────────────────────────────────────────────────────────────────

type AppSchema = Schema<QueryRoot, MutationRoot, SubscriptionRoot>;

#[tokio::main]
async fn main() {
    let client = Client::new();

    let schema = Schema::build(QueryRoot, MutationRoot, SubscriptionRoot)
        .data(client)
        .finish();

    let app = Router::new()
        .route_service("/graphql", GraphQL::new(schema.clone()))
        .route_service("/graphql/ws", GraphQLSubscription::new(schema))
        .layer(CorsLayer::permissive());

    println!("GraphQL server listening on :8080  (WS: /graphql/ws)");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
