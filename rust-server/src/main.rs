use async_graphql::{Context, Object, Schema, SimpleObject, Subscription};
use async_graphql_axum::{GraphQL, GraphQLSubscription};
use axum::Router;
use futures_util::{Stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
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

#[derive(SimpleObject, Deserialize)]
struct ModelInfo {
    model_id: String,
    device: String,
    vram_used: String,
    active_adapter: Option<String>,
    load_in_4bit: bool,
}

#[derive(SimpleObject, Deserialize)]
struct AdaptersInfo {
    available: Vec<String>,
    active: Option<String>,
}

#[derive(SimpleObject, Deserialize)]
struct FineTuneStatus {
    running: bool,
    step: i32,
    total_steps: i32,
    loss: Option<f64>,
    error: Option<String>,
    completed_adapter: Option<String>,
}

#[derive(SimpleObject, Deserialize)]
struct AdapterResult {
    ok: bool,
    error: Option<String>,
    active_adapter: Option<String>,
}

#[derive(SimpleObject, Deserialize)]
struct StartFineTuneResult {
    ok: bool,
    error: Option<String>,
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

#[derive(Serialize)]
struct FineTuneRequest {
    adapter_name: String,
    dataset_name: String,
    num_samples: i32,
    num_epochs: i32,
    learning_rate: f64,
    lora_r: i32,
    lora_alpha: i32,
}

#[derive(Serialize)]
struct LoadAdapterRequest {
    adapter_name: String,
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

    async fn adapters(&self, ctx: &Context<'_>) -> async_graphql::Result<AdaptersInfo> {
        let client = ctx.data::<Client>()?;
        let info = client
            .get(format!("{}/adapters", worker_url()))
            .send()
            .await?
            .json::<AdaptersInfo>()
            .await?;
        Ok(info)
    }

    async fn model_info(&self, ctx: &Context<'_>) -> async_graphql::Result<ModelInfo> {
        let client = ctx.data::<Client>()?;
        let info = client
            .get(format!("{}/model-info", worker_url()))
            .send()
            .await?
            .json::<ModelInfo>()
            .await?;
        Ok(info)
    }

    async fn fine_tune_status(&self, ctx: &Context<'_>) -> async_graphql::Result<FineTuneStatus> {
        let client = ctx.data::<Client>()?;
        let status = client
            .get(format!("{}/fine-tune/status", worker_url()))
            .send()
            .await?
            .json::<FineTuneStatus>()
            .await?;
        Ok(status)
    }
}

// ── Mutation ──────────────────────────────────────────────────────────────────

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

    async fn start_fine_tune(
        &self,
        ctx: &Context<'_>,
        adapter_name: Option<String>,
        dataset_name: Option<String>,
        num_samples: Option<i32>,
        num_epochs: Option<i32>,
        learning_rate: Option<f64>,
        lora_r: Option<i32>,
        lora_alpha: Option<i32>,
    ) -> async_graphql::Result<StartFineTuneResult> {
        let client = ctx.data::<Client>()?;
        let body = FineTuneRequest {
            adapter_name: adapter_name.unwrap_or_else(|| "my-lora".to_string()),
            dataset_name: dataset_name.unwrap_or_else(|| "tatsu-lab/alpaca".to_string()),
            num_samples: num_samples.unwrap_or(500),
            num_epochs: num_epochs.unwrap_or(1),
            learning_rate: learning_rate.unwrap_or(2e-4),
            lora_r: lora_r.unwrap_or(8),
            lora_alpha: lora_alpha.unwrap_or(16),
        };
        let result = client
            .post(format!("{}/fine-tune", worker_url()))
            .json(&body)
            .send()
            .await?
            .json::<StartFineTuneResult>()
            .await?;
        Ok(result)
    }

    async fn load_adapter(
        &self,
        ctx: &Context<'_>,
        adapter_name: String,
    ) -> async_graphql::Result<AdapterResult> {
        let client = ctx.data::<Client>()?;
        let result = client
            .post(format!("{}/fine-tune/load", worker_url()))
            .json(&LoadAdapterRequest { adapter_name })
            .send()
            .await?
            .json::<AdapterResult>()
            .await?;
        Ok(result)
    }

    async fn use_base_model(
        &self,
        ctx: &Context<'_>,
    ) -> async_graphql::Result<AdapterResult> {
        let client = ctx.data::<Client>()?;
        let result = client
            .post(format!("{}/fine-tune/unload", worker_url()))
            .send()
            .await?
            .json::<Value>()
            .await?;
        Ok(AdapterResult {
            ok: result["ok"].as_bool().unwrap_or(false),
            error: None,
            active_adapter: None,
        })
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
