use async_graphql::{Context, EmptySubscription, Object, Schema, SimpleObject};
use async_graphql_axum::GraphQL;
use axum::Router;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;

// ── GraphQL types ─────────────────────────────────────────────────────────────

#[derive(SimpleObject)]
struct InferenceResult {
    text: String,
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

// ── Schema ────────────────────────────────────────────────────────────────────

struct QueryRoot;

#[Object]
impl QueryRoot {
    async fn health(&self) -> &str {
        "ok"
    }
}

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
        let worker_url = std::env::var("WORKER_URL")
            .unwrap_or_else(|_| "http://python-worker:8001".to_string());

        let resp: WorkerResponse = client
            .post(format!("{}/infer", worker_url))
            .json(&WorkerRequest {
                prompt,
                web_search: web_search.unwrap_or(false),
            })
            .send()
            .await?
            .json()
            .await?;

        Ok(InferenceResult { text: resp.text })
    }
}

type AppSchema = Schema<QueryRoot, MutationRoot, EmptySubscription>;

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let client = Client::new();

    let schema = Schema::build(QueryRoot, MutationRoot, EmptySubscription)
        .data(client)
        .finish();

    let app = Router::new()
        .route_service("/graphql", GraphQL::new(schema))
        .layer(CorsLayer::permissive());

    println!("GraphQL server listening on :8080");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
