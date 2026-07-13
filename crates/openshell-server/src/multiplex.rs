// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol multiplexing for gRPC and HTTP on the same port.
//!
//! This module implements connection-level multiplexing that routes requests
//! to either the gRPC service or HTTP endpoints based on the request headers.

use bytes::Bytes;
use http::{HeaderValue, Request, Response};
use http_body::Body;
use http_body_util::BodyExt;
use hyper::body::Incoming;
use hyper_util::{
    rt::{TokioExecutor, TokioIo, TokioTimer},
    server::conn::auto::Builder,
    service::TowerToHyperService,
};
use metrics::{counter, histogram};
use openshell_core::Config;
use openshell_core::proto::{
    inference_server::InferenceServer, open_shell_server::OpenShellServer,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncWrite};
use tower::ServiceExt;
use tower_http::request_id::{MakeRequestId, RequestId};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::{
    OpenShellService, ServerState,
    auth::authenticator::AuthenticatorChain,
    auth::authz::AuthzPolicy,
    auth::identity::Identity,
    auth::oidc::{self, OidcAuthenticator},
    auth::principal::{Principal, UserPrincipal},
    http_router,
    inference::InferenceService,
    service_http_router,
};

/// Adapter that exposes HTTP headers to OpenTelemetry's `Extractor` trait
/// for W3C trace context propagation. gRPC metadata maps directly to HTTP/2
/// headers, so `traceparent` arrives as a regular header.
struct HeaderExtractor<'a>(&'a http::HeaderMap);

impl opentelemetry::propagation::Extractor for HeaderExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(http::HeaderName::as_str).collect()
    }
}

/// Request-ID generator that produces a UUID v4 for each inbound request.
#[derive(Clone)]
struct UuidRequestId;

impl MakeRequestId for UuidRequestId {
    fn make_request_id<B>(&mut self, _req: &Request<B>) -> Option<RequestId> {
        let id = uuid::Uuid::new_v4().to_string();
        Some(RequestId::new(HeaderValue::from_str(&id).unwrap()))
    }
}

/// gRPC methods that never get a span.
///
/// `Health` is a liveness poll: every client and every kubelet probe calls it
/// on a timer, so tracing it turns the backend into a wall of empty 0ms root
/// traces that bury real work. Add other high-frequency pollers here as they
/// appear.
const UNTRACED_GRPC_METHODS: &[&str] = &["Health"];

/// Whether the request is gRPC, by content-type. The multiplexer routes on the
/// same header, so this agrees with the branch the request actually takes.
fn is_grpc_request<B>(req: &Request<B>) -> bool {
    req.headers()
        .get(http::header::CONTENT_TYPE)
        .is_some_and(|v| v.as_bytes().starts_with(b"application/grpc"))
}

/// Split a gRPC path (`/openshell.v1.OpenShell/GetSandbox`) into its fully
/// qualified service and its method. `None` when the path has no such shape.
fn grpc_service_and_method(path: &str) -> Option<(&str, &str)> {
    let (service, method) = path.trim_start_matches('/').rsplit_once('/')?;
    (!service.is_empty() && !method.is_empty()).then_some((service, method))
}

/// Build a tracing span for an inbound request, recording the `request_id`
/// header (set by [`UuidRequestId`] or supplied by the client).
///
/// gRPC requests are named per the `OTel` RPC semantic conventions:
/// `{rpc.service}/{rpc.method}` (`openshell.v1.OpenShell/GetSandbox`), carried
/// on the `otel.name` field because a tracing span name must be `'static`.
/// Methods in [`UNTRACED_GRPC_METHODS`] get no span at all.
///
/// When the request carries a W3C `traceparent` header, the span is parented
/// to the upstream trace context so distributed traces connect across the
/// client-gateway boundary. Invalid or missing `traceparent` values are
/// silently ignored (the span starts a new trace root).
///
/// Long-lived streaming RPCs (`ConnectSupervisor`, `RelayStream`) produce spans
/// that live for the duration of the connection, so those spans are not
/// exported until the stream closes.
fn make_request_span<B>(req: &Request<B>) -> Span {
    let path = req.uri().path();
    let request_id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-");

    let rpc = is_grpc_request(req)
        .then(|| grpc_service_and_method(path))
        .flatten();

    let span = match rpc {
        Some((_, method)) if UNTRACED_GRPC_METHODS.contains(&method) => return Span::none(),
        Some((service, method)) => {
            let otel_name = format!("{service}/{method}");
            tracing::info_span!(
                "request",
                method = %req.method(),
                path,
                request_id,
                otel.name = otel_name.as_str(),
                otel.kind = "server",
                rpc.system = "grpc",
                rpc.service = service,
                rpc.method = method,
            )
        }
        None if matches!(path, "/health" | "/healthz" | "/readyz") => tracing::debug_span!(
            "request",
            method = %req.method(),
            path,
            request_id,
            otel.kind = "server",
        ),
        None => tracing::info_span!(
            "request",
            method = %req.method(),
            path,
            request_id,
            otel.kind = "server",
        ),
    };

    // Parent the span to any inbound W3C trace context. No-op when the header
    // is absent or malformed (the span starts a new trace root).
    let parent_cx = opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.extract(&HeaderExtractor(req.headers()))
    });
    // Returns Err when no OTel layer is installed; harmless.
    let _ = span.set_parent(parent_cx);

    span
}

/// Log response status and latency within the request span.
fn log_response<B>(res: &Response<B>, latency: Duration, _span: &Span) {
    tracing::info!(
        status = res.status().as_u16(),
        latency_ms = latency.as_millis(),
        "response"
    );
}

/// Wrap a service with the standard request-ID middleware stack.
///
/// Layer order: `SetRequestId` → `TraceLayer` → `PropagateRequestId`.
macro_rules! request_id_middleware {
    ($service:expr) => {{
        let x_request_id = ::http::HeaderName::from_static("x-request-id");
        ::tower::ServiceBuilder::new()
            .layer(::tower_http::request_id::SetRequestIdLayer::new(
                x_request_id.clone(),
                UuidRequestId,
            ))
            .layer(
                ::tower_http::trace::TraceLayer::new_for_http()
                    .make_span_with(make_request_span)
                    .on_request(())
                    .on_response(log_response),
            )
            .layer(::tower_http::request_id::PropagateRequestIdLayer::new(
                x_request_id,
            ))
            .service($service)
    }};
}

/// Maximum inbound gRPC message size (1 MB).
///
/// Replaces tonic's implicit 4 MB default with a conservative limit to
/// bound memory allocation from a single request. Sandbox creation is
/// the largest payload and well within this cap under normal use.
const MAX_GRPC_DECODE_SIZE: usize = 1_048_576;

/// Multiplexed gRPC/HTTP service.
#[derive(Clone)]
pub struct MultiplexService {
    state: Arc<ServerState>,
}

impl MultiplexService {
    /// Create a new multiplex service.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn new(state: Arc<ServerState>) -> Self {
        Self { state }
    }

    /// Serve a connection, routing to gRPC or HTTP based on content-type.
    pub async fn serve<S>(&self, stream: S) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        self.serve_with_peer_identity(stream, None).await
    }

    /// Serve a TLS connection with an optional mTLS peer identity.
    pub async fn serve_with_peer_identity<S>(
        &self,
        stream: S,
        peer_identity: Option<Identity>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        let openshell = OpenShellServer::new(OpenShellService::new(self.state.clone()))
            .max_decoding_message_size(MAX_GRPC_DECODE_SIZE);
        let inference = InferenceServer::new(InferenceService::new(self.state.clone()))
            .max_decoding_message_size(MAX_GRPC_DECODE_SIZE);
        let authz_policy = self.state.config.oidc.as_ref().map(|oidc| AuthzPolicy {
            admin_role: oidc.admin_role.clone(),
            user_role: oidc.user_role.clone(),
            scopes_enabled: !oidc.scopes_claim.is_empty(),
        });
        let authenticator_chain = build_authenticator_chain(&self.state);
        let grpc_service = AuthGrpcRouter::with_peer_identity(
            GrpcRouter::new(openshell, inference),
            authenticator_chain,
            authz_policy,
            self.state
                .config
                .mtls_auth
                .enabled
                .then_some(peer_identity)
                .flatten(),
            self.state.config.mtls_auth.enabled,
            self.state.config.auth.allow_unauthenticated_users,
        );
        let grpc_service =
            GrpcRateLimitService::new(grpc_service, self.state.grpc_rate_limiter.clone());
        let http_service = http_router(self.state.clone());

        let grpc_service = request_id_middleware!(grpc_service);
        let http_service = request_id_middleware!(http_service);

        let service = MultiplexedService::new(grpc_service, http_service);

        let mut builder = Builder::new(TokioExecutor::new());
        // Server-side HTTP/2 keepalive: supervisors hold long-lived sessions, and without
        // it the gateway never PINGs them, so idle/half-dead connections linger and orphan
        // in-flight relay execs. The timer is required — hyper panics on the keepalive
        // interval without one.
        builder
            .http2()
            .timer(TokioTimer::new())
            .adaptive_window(true)
            .keep_alive_interval(Some(Duration::from_secs(20)))
            .keep_alive_timeout(Duration::from_secs(10));

        builder
            .serve_connection_with_upgrades(TokioIo::new(stream), service)
            .await?;

        Ok(())
    }

    /// Serve a plaintext HTTP connection for sandbox service endpoints only.
    pub async fn serve_service_http<S>(
        &self,
        stream: S,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        let http_service = TowerToHyperService::new(request_id_middleware!(service_http_router(
            self.state.clone()
        )));

        Builder::new(TokioExecutor::new())
            .serve_connection_with_upgrades(TokioIo::new(stream), http_service)
            .await?;

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct GrpcRateLimiter {
    requests: u64,
    window: Duration,
    state: Arc<Mutex<GrpcRateLimitState>>,
}

#[derive(Debug)]
struct GrpcRateLimitState {
    window_started: Instant,
    remaining: u64,
}

impl GrpcRateLimiter {
    pub fn from_config(config: &Config) -> Option<Self> {
        let (requests, window) = config.grpc_rate_limit()?;
        Some(Self {
            requests,
            window,
            state: Arc::new(Mutex::new(GrpcRateLimitState {
                window_started: Instant::now(),
                remaining: requests,
            })),
        })
    }

    fn allow(&self) -> bool {
        let now = Instant::now();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if now.duration_since(state.window_started) >= self.window {
            state.window_started = now;
            state.remaining = self.requests;
        }
        if state.remaining == 0 {
            false
        } else {
            state.remaining -= 1;
            true
        }
    }

    /// Report whether the limiter currently has capacity without consuming a
    /// token, rolling the window over first so an elapsed window reports
    /// capacity again.
    ///
    /// Used by `poll_ready` so an exhausted limiter reports readiness instead
    /// of blocking on inner-service backpressure: `call` can then return
    /// `RESOURCE_EXHAUSTED` immediately rather than waiting for the inner gRPC
    /// service to become ready.
    fn has_capacity(&self) -> bool {
        let now = Instant::now();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if now.duration_since(state.window_started) >= self.window {
            state.window_started = now;
            state.remaining = self.requests;
        }
        state.remaining > 0
    }
}

#[derive(Clone)]
struct GrpcRateLimitService<S> {
    inner: S,
    limiter: Option<GrpcRateLimiter>,
    /// Set by `poll_ready` when it reports synthetic readiness for an
    /// exhausted limiter without polling the inner service. The paired `call`
    /// must then reject with `RESOURCE_EXHAUSTED` instead of forwarding to an
    /// inner service that never reported readiness — even if the rate-limit
    /// window rolls over in between. Reset whenever `poll_ready` defers to the
    /// inner service.
    rate_limited: bool,
}

impl<S> GrpcRateLimitService<S> {
    fn new(inner: S, limiter: Option<GrpcRateLimiter>) -> Self {
        Self {
            inner,
            limiter,
            rate_limited: false,
        }
    }
}

impl<S, B> tower::Service<Request<B>> for GrpcRateLimitService<S>
where
    S: tower::Service<Request<B>, Response = Response<tonic::body::Body>>,
    S::Future: Send + 'static,
    B: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        // When the limiter is exhausted, report ready so `call` can return
        // RESOURCE_EXHAUSTED immediately. Delegating to the inner service here
        // would make rate-limited requests wait on inner backpressure (a
        // pending inner `poll_ready`) before they are rejected. The check is
        // non-consuming: the token is only consumed in `call` via `allow`.
        //
        // Crucially, this path does NOT poll the inner service, so the inner
        // service has not reported readiness. Record that decision so the
        // paired `call` rejects rather than forwarding to a service that never
        // became ready — even if the rate-limit window rolls over in between.
        if self
            .limiter
            .as_ref()
            .is_some_and(|limiter| !limiter.has_capacity())
        {
            self.rate_limited = true;
            return Poll::Ready(Ok(()));
        }
        self.rate_limited = false;
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<B>) -> Self::Future {
        // If `poll_ready` short-circuited an exhausted limiter, it never polled
        // the inner service to readiness. Honor that decision regardless of the
        // limiter's current state (the window may have rolled over since): the
        // Tower contract forbids forwarding to an inner service that did not
        // report readiness.
        if std::mem::take(&mut self.rate_limited) {
            let response =
                tonic::Status::resource_exhausted("gRPC rate limit exceeded").into_http();
            return Box::pin(async move { Ok(response) });
        }
        if self
            .limiter
            .as_ref()
            .is_some_and(|limiter| !limiter.allow())
        {
            let response =
                tonic::Status::resource_exhausted("gRPC rate limit exceeded").into_http();
            return Box::pin(async move { Ok(response) });
        }
        let future = self.inner.call(req);
        Box::pin(future)
    }
}

/// Combined gRPC service that routes between `OpenShell` and Inference services
/// based on the request path prefix.
#[derive(Clone)]
pub struct GrpcRouter<N, I> {
    openshell: N,
    inference: I,
}

impl<N, I> GrpcRouter<N, I> {
    fn new(openshell: N, inference: I) -> Self {
        Self {
            openshell,
            inference,
        }
    }
}

const INFERENCE_PATH_PREFIX: &str = "/openshell.inference.v1.Inference/";

impl<N, I, B> tower::Service<Request<B>> for GrpcRouter<N, I>
where
    N: tower::Service<Request<B>> + Clone + Send + 'static,
    N::Response: Send,
    N::Future: Send,
    N::Error: Send,
    I: tower::Service<Request<B>, Response = N::Response, Error = N::Error>
        + Clone
        + Send
        + 'static,
    I::Future: Send,
    B: Send + 'static,
{
    type Response = N::Response;
    type Error = N::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: Request<B>) -> Self::Future {
        let is_inference = req.uri().path().starts_with(INFERENCE_PATH_PREFIX);

        if is_inference {
            let mut svc = self.inference.clone();
            Box::pin(async move { svc.ready().await?.call(req).await })
        } else {
            let mut svc = self.openshell.clone();
            Box::pin(async move { svc.ready().await?.call(req).await })
        }
    }
}

/// Assemble the authenticator chain for the gateway.
///
/// Chain order (first-match-wins):
/// 1. `K8sServiceAccountAuthenticator` (path-scoped to `IssueSandboxToken`)
///    — exchanges a projected SA token for a `Principal::Sandbox` so the
///    `IssueSandboxToken` handler can mint a gateway JWT. No-op on every
///    other path; only present when the gateway runs in-cluster.
/// 2. `SandboxJwtAuthenticator` — validates gateway-minted JWTs. Recognized
///    via a distinctive `kid` so non-matching Bearer tokens fall through.
/// 3. `OidcAuthenticator` — validates user Bearer tokens against the
///    configured OIDC issuer. Returns `Unauthenticated` for missing
///    Bearer headers so non-OIDC clients can't sneak through.
///
/// Once sandbox authentication is configured, callers must present an
/// explicit credential for authenticated gRPC methods. Missing bearer auth
/// is promoted to an mTLS user only when `mtls_auth.enabled` is configured
/// for local single-user gateways, or to an unsafe local developer user when
/// `auth.allow_unauthenticated_users` is explicitly enabled.
///
/// When neither OIDC nor sandbox credentials are configured (a barebones
/// dev gateway), the chain is left as `None` so the router short-circuits
/// to pass-through unless mTLS or local unauthenticated users are enabled.
fn build_authenticator_chain(state: &ServerState) -> Option<AuthenticatorChain> {
    let mut authenticators: Vec<Arc<dyn crate::auth::authenticator::Authenticator>> = Vec::new();
    if let Some(k8s) = state.k8s_sa_authenticator.clone() {
        authenticators.push(k8s);
    }
    if let Some(jwt) = state.sandbox_jwt_authenticator.clone() {
        authenticators.push(jwt);
    }
    if let Some(cache) = state.oidc_cache.clone() {
        authenticators.push(Arc::new(OidcAuthenticator::new(cache)));
    }
    if authenticators.is_empty() {
        return None;
    }
    Some(AuthenticatorChain::new(authenticators))
}

/// gRPC router wrapper that runs the [`AuthenticatorChain`] and inserts the
/// resulting [`Principal`] into the request's extensions.
///
/// Behavior:
/// - Strip any external `x-openshell-auth-source` marker first (so callers
///   cannot spoof a sandbox identity).
/// - Health probes / reflection bypass the chain entirely.
/// - When no chain is configured (OIDC not configured), forward without
///   authentication — preserves today's pass-through behavior.
/// - Otherwise, run the chain. The first match produces a `Principal`.
///   `Principal::User` is gated by the RBAC `AuthzPolicy`.
///   `Principal::Sandbox` is gated by a supervisor-method allowlist, then
///   handlers enforce same-sandbox scope on request bodies.
#[derive(Clone)]
pub struct AuthGrpcRouter<S> {
    inner: S,
    authenticator_chain: Option<AuthenticatorChain>,
    authz_policy: Option<AuthzPolicy>,
    /// mTLS peer identity extracted from the TLS handshake.
    peer_identity: Option<Identity>,
    mtls_auth_enabled: bool,
    allow_unauthenticated_users: bool,
}

impl<S> AuthGrpcRouter<S> {
    #[cfg(test)]
    fn new(
        inner: S,
        authenticator_chain: Option<AuthenticatorChain>,
        authz_policy: Option<AuthzPolicy>,
    ) -> Self {
        Self::with_peer_identity(inner, authenticator_chain, authz_policy, None, false, false)
    }

    fn with_peer_identity(
        inner: S,
        authenticator_chain: Option<AuthenticatorChain>,
        authz_policy: Option<AuthzPolicy>,
        peer_identity: Option<Identity>,
        mtls_auth_enabled: bool,
        allow_unauthenticated_users: bool,
    ) -> Self {
        Self {
            inner,
            authenticator_chain,
            authz_policy,
            peer_identity,
            mtls_auth_enabled,
            allow_unauthenticated_users,
        }
    }
}

fn unauthenticated_dev_user_principal() -> Principal {
    Principal::User(UserPrincipal {
        identity: Identity {
            subject: "unauthenticated-local-dev".to_string(),
            display_name: Some("Unauthenticated Local Dev".to_string()),
            roles: vec!["openshell-user".to_string(), "openshell-admin".to_string()],
            scopes: vec!["openshell:all".to_string()],
            provider: crate::auth::identity::IdentityProvider::LocalDev,
        },
    })
}

fn status_response(status: tonic::Status) -> Response<tonic::body::Body> {
    status.into_http()
}

impl<S, B> tower::Service<Request<B>> for AuthGrpcRouter<S>
where
    S: tower::Service<Request<B>, Response = Response<tonic::body::Body>> + Clone + Send + 'static,
    S::Future: Send,
    S::Error: Send + Into<Box<dyn std::error::Error + Send + Sync>>,
    B: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: Request<B>) -> Self::Future {
        let chain = self.authenticator_chain.clone();
        let authz_policy = self.authz_policy.clone();
        let peer_identity = self.peer_identity.clone();
        let mtls_auth_enabled = self.mtls_auth_enabled;
        let allow_unauthenticated_users = self.allow_unauthenticated_users;
        let mut inner = self.inner.clone();

        Box::pin(async move {
            let mut req = req;

            let path = req.uri().path().to_string();

            // Health probes and reflection — truly unauthenticated.
            if oidc::is_unauthenticated_method(&path) {
                return inner.ready().await?.call(req).await;
            }

            let principal = if let Some(chain) = chain {
                match chain.authenticate(req.headers(), &path).await {
                    Ok(Some(p)) => p,
                    Ok(None) => match (mtls_auth_enabled, peer_identity) {
                        (true, Some(identity)) => Principal::User(UserPrincipal { identity }),
                        _ if allow_unauthenticated_users => unauthenticated_dev_user_principal(),
                        _ => {
                            return Ok(status_response(tonic::Status::unauthenticated(
                                "missing authorization header",
                            )));
                        }
                    },
                    Err(status) => return Ok(status_response(status)),
                }
            } else if mtls_auth_enabled {
                let Some(identity) = peer_identity else {
                    return Ok(status_response(tonic::Status::unauthenticated(
                        "missing client certificate",
                    )));
                };
                Principal::User(UserPrincipal { identity })
            } else if allow_unauthenticated_users {
                unauthenticated_dev_user_principal()
            } else {
                // No auth configured — pass through for dev /
                // fronting-proxy deployments.
                return inner.ready().await?.call(req).await;
            };

            match principal {
                Principal::User(ref user) => {
                    if !crate::auth::method_authz::is_user_callable(&path) {
                        return Ok(status_response(tonic::Status::permission_denied(
                            "this method requires a sandbox principal",
                        )));
                    }
                    if let Some(ref policy) = authz_policy
                        && let Err(status) = policy.check(&user.identity, &path)
                    {
                        return Ok(status_response(status));
                    }
                }
                Principal::Sandbox(_) => {
                    if !crate::auth::sandbox_methods::is_sandbox_callable(&path) {
                        return Ok(status_response(tonic::Status::permission_denied(
                            "sandbox principals may not call this method",
                        )));
                    }
                }
                Principal::Anonymous => {
                    return Ok(status_response(tonic::Status::unauthenticated(
                        "anonymous callers may not call authenticated methods",
                    )));
                }
            }

            req.extensions_mut().insert(principal);
            inner.ready().await?.call(req).await
        })
    }
}

/// Service that multiplexes between gRPC and HTTP.
#[derive(Clone)]
pub struct MultiplexedService<G, H> {
    grpc: G,
    http: H,
}

impl<G, H> MultiplexedService<G, H> {
    /// Create a new multiplexed service from gRPC and HTTP services.
    #[must_use]
    pub fn new(grpc: G, http: H) -> Self {
        Self { grpc, http }
    }
}

impl<G, H, GBody, HBody> hyper::service::Service<Request<Incoming>> for MultiplexedService<G, H>
where
    G: tower::Service<Request<BoxBody>, Response = Response<GBody>> + Clone + Send + 'static,
    G::Future: Send,
    G::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
    GBody: Body<Data = Bytes> + Send + 'static,
    GBody::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
    H: tower::Service<Request<BoxBody>, Response = Response<HBody>> + Clone + Send + 'static,
    H::Future: Send,
    H::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
    HBody: Body<Data = Bytes> + Send + 'static,
    HBody::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    type Response = Response<BoxBody>;
    type Error = Box<dyn std::error::Error + Send + Sync>;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&self, req: Request<Incoming>) -> Self::Future {
        let is_grpc = req
            .headers()
            .get("content-type")
            .is_some_and(|v| v.as_bytes().starts_with(b"application/grpc"));

        if is_grpc {
            let method = grpc_method_from_path(req.uri().path());
            let start = Instant::now();
            let mut grpc = self.grpc.clone();
            Box::pin(async move {
                let (parts, body) = req.into_parts();
                let body = body.map_err(Into::into).boxed_unsync();
                let req = Request::from_parts(parts, BoxBody(body));

                let res = grpc
                    .ready()
                    .await
                    .map_err(Into::into)?
                    .call(req)
                    .await
                    .map_err(Into::into)?;

                let code = grpc_status_from_response(&res);
                let elapsed = start.elapsed().as_secs_f64();
                counter!("openshell_server_grpc_requests_total", "method" => method.clone(), "code" => code.clone()).increment(1);
                histogram!("openshell_server_grpc_request_duration_seconds", "method" => method, "code" => code).record(elapsed);

                let (parts, body) = res.into_parts();
                let body = body.map_err(Into::into).boxed_unsync();
                Ok(Response::from_parts(parts, BoxBody(body)))
            })
        } else {
            let path = normalize_http_path(req.uri().path());
            let start = Instant::now();
            let mut http = self.http.clone();
            Box::pin(async move {
                let (parts, body) = req.into_parts();
                let body = body.map_err(Into::into).boxed_unsync();
                let req = Request::from_parts(parts, BoxBody(body));

                let res = http
                    .ready()
                    .await
                    .map_err(Into::into)?
                    .call(req)
                    .await
                    .map_err(Into::into)?;

                let status = res.status().as_u16().to_string();
                let elapsed = start.elapsed().as_secs_f64();
                counter!("openshell_server_http_requests_total", "path" => path, "status" => status.clone()).increment(1);
                histogram!("openshell_server_http_request_duration_seconds", "path" => path, "status" => status).record(elapsed);

                let (parts, body) = res.into_parts();
                let body = body.map_err(Into::into).boxed_unsync();
                Ok(Response::from_parts(parts, BoxBody(body)))
            })
        }
    }
}

fn grpc_method_from_path(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_string()
}

fn grpc_status_from_response<B>(res: &Response<B>) -> String {
    res.headers()
        .get("grpc-status")
        .and_then(|v| v.to_str().ok())
        .map_or_else(|| "0".to_string(), ToString::to_string)
}

fn normalize_http_path(path: &str) -> &'static str {
    match path {
        p if p.starts_with("/_ws_tunnel") => "/_ws_tunnel",
        p if p.starts_with("/auth/") => "/auth",
        _ => "unknown",
    }
}

/// Extract an [`Identity`] from the peer certificates presented during a TLS
/// handshake. Returns `None` if no client certificate was presented.
pub fn extract_peer_identity<S>(tls_stream: &tokio_rustls::server::TlsStream<S>) -> Option<Identity>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    use crate::auth::identity::IdentityProvider;
    use x509_parser::prelude::*;

    let (_, server_conn) = tls_stream.get_ref();
    let certs = server_conn.peer_certificates()?;
    let first = certs.first()?;

    let (_, cert) = X509Certificate::from_der(first.as_ref()).ok()?;
    let subject = cert.subject();

    let cn = subject
        .iter_common_name()
        .next()
        .and_then(|attr| attr.as_str().ok())
        .unwrap_or("unknown")
        .to_string();

    let roles: Vec<String> = subject
        .iter_organizational_unit()
        .filter_map(|attr| attr.as_str().ok().map(String::from))
        .collect();

    Some(Identity {
        subject: cn.clone(),
        display_name: Some(cn),
        roles,
        scopes: Vec::new(),
        provider: IdentityProvider::Mtls,
    })
}

/// Boxed body type for uniform handling.
pub struct BoxBody(
    http_body_util::combinators::UnsyncBoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>,
);

impl Body for BoxBody {
    type Data = Bytes;
    type Error = Box<dyn std::error::Error + Send + Sync>;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        Pin::new(&mut self.0).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.0.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.0.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use http_body_util::Empty;
    use opentelemetry::trace::TraceContextExt;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::Service;

    #[test]
    fn uuid_request_id_generates_valid_uuid() {
        let mut maker = UuidRequestId;
        let req = Request::builder().body(()).unwrap();
        let id = maker.make_request_id(&req).expect("should produce an ID");
        let value = id.header_value().to_str().unwrap();
        uuid::Uuid::parse_str(value).expect("should be a valid UUID");
    }

    #[test]
    fn uuid_request_id_generates_unique_ids() {
        let mut maker = UuidRequestId;
        let req = Request::builder().body(()).unwrap();
        let id1 = maker.make_request_id(&req).unwrap();
        let id2 = maker.make_request_id(&req).unwrap();
        assert_ne!(id1.header_value(), id2.header_value());
    }

    async fn test_health_store() -> Arc<crate::Store> {
        Arc::new(
            crate::Store::connect("sqlite::memory:")
                .await
                .expect("connect in-memory sqlite store for tests"),
        )
    }

    async fn start_http_server_with_middleware() -> std::net::SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let http_service = crate::http::health_router(test_health_store().await);
        let http_service = request_id_middleware!(http_service);

        let service = MultiplexedService::new(http_service.clone(), http_service);

        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    continue;
                };
                let svc = service.clone();
                tokio::spawn(async move {
                    let _ = Builder::new(TokioExecutor::new())
                        .serve_connection(TokioIo::new(stream), svc)
                        .await;
                });
            }
        });

        addr
    }

    async fn http1_get(
        addr: std::net::SocketAddr,
        path: &str,
        headers: &[(&str, &str)],
    ) -> Response<Incoming> {
        let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let (mut sender, conn) = hyper::client::conn::http1::Builder::new()
            .handshake(TokioIo::new(stream))
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = conn.await;
        });

        let mut builder = Request::builder()
            .method("GET")
            .uri(format!("http://{addr}{path}"));
        for (k, v) in headers {
            builder = builder.header(*k, *v);
        }
        let req = builder.body(Empty::<Bytes>::new()).unwrap();
        sender.send_request(req).await.unwrap()
    }

    #[tokio::test]
    async fn http_response_includes_request_id() {
        let addr = start_http_server_with_middleware().await;
        let resp = http1_get(addr, "/healthz", &[]).await;
        assert_eq!(resp.status(), 200);

        let request_id = resp
            .headers()
            .get("x-request-id")
            .expect("response should include x-request-id header");
        let id_str = request_id.to_str().unwrap();
        uuid::Uuid::parse_str(id_str).expect("should be a valid UUID");
    }

    #[tokio::test]
    async fn http_preserves_client_request_id() {
        let addr = start_http_server_with_middleware().await;
        let client_id = "my-custom-correlation-id";
        let resp = http1_get(addr, "/healthz", &[("x-request-id", client_id)]).await;
        assert_eq!(resp.status(), 200);

        let request_id = resp
            .headers()
            .get("x-request-id")
            .expect("response should include x-request-id header");
        assert_eq!(request_id.to_str().unwrap(), client_id);
    }

    #[tokio::test]
    async fn each_request_gets_unique_id() {
        let addr = start_http_server_with_middleware().await;

        let mut ids = Vec::new();
        for _ in 0..3 {
            let resp = http1_get(addr, "/healthz", &[]).await;
            let id = resp
                .headers()
                .get("x-request-id")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            ids.push(id);
        }

        assert_ne!(ids[0], ids[1]);
        assert_ne!(ids[1], ids[2]);
        assert_ne!(ids[0], ids[2]);
    }

    #[tokio::test]
    async fn grpc_path_includes_request_id() {
        let addr = start_http_server_with_middleware().await;
        let resp = http1_get(
            addr,
            "/openshell.v1.OpenShell/Health",
            &[
                ("content-type", "application/grpc"),
                ("x-request-id", "grpc-corr-id"),
            ],
        )
        .await;

        let request_id = resp
            .headers()
            .get("x-request-id")
            .expect("gRPC-routed response should include x-request-id header");
        assert_eq!(request_id.to_str().unwrap(), "grpc-corr-id");
    }

    #[derive(Clone)]
    struct CountingGrpcService {
        calls: Arc<AtomicUsize>,
    }

    impl Service<Request<()>> for CountingGrpcService {
        type Response = Response<tonic::body::Body>;
        type Error = std::convert::Infallible;
        type Future = std::future::Ready<Result<Self::Response, Self::Error>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _req: Request<()>) -> Self::Future {
            self.calls.fetch_add(1, Ordering::Relaxed);
            std::future::ready(Ok(Response::new(tonic::body::Body::empty())))
        }
    }

    /// Inner service that is never ready, used to prove the rate limiter does
    /// not wait on inner-service backpressure when it is already exhausted.
    /// Counts `call` invocations so tests can assert the limiter never forwards
    /// to an inner service that did not report readiness.
    #[derive(Clone)]
    struct PendingInnerService {
        calls: Arc<AtomicUsize>,
    }

    impl PendingInnerService {
        fn new() -> Self {
            Self {
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl Service<Request<()>> for PendingInnerService {
        type Response = Response<tonic::body::Body>;
        type Error = std::convert::Infallible;
        type Future = std::future::Ready<Result<Self::Response, Self::Error>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Pending
        }

        fn call(&mut self, _req: Request<()>) -> Self::Future {
            self.calls.fetch_add(1, Ordering::Relaxed);
            std::future::ready(Ok(Response::new(tonic::body::Body::empty())))
        }
    }

    #[tokio::test]
    async fn grpc_rate_limit_poll_ready_short_circuits_exhausted_limiter() {
        // An exhausted limiter must report ready even when the inner service is
        // pending, so `call` returns RESOURCE_EXHAUSTED instead of waiting on
        // inner backpressure.
        let config = Config::new(None).with_grpc_rate_limit(Some(1), Some(60));
        let limiter = GrpcRateLimiter::from_config(&config).expect("limiter should be enabled");
        // Consume the single token so the limiter is exhausted.
        assert!(limiter.allow());

        let mut exhausted = GrpcRateLimitService::new(PendingInnerService::new(), Some(limiter));
        let mut cx = Context::from_waker(std::task::Waker::noop());
        assert!(
            matches!(exhausted.poll_ready(&mut cx), Poll::Ready(Ok(()))),
            "exhausted limiter should report ready despite a pending inner service",
        );
        let response = exhausted.call(Request::new(())).await.unwrap();
        assert_eq!(grpc_status_from_response(&response), "8");

        // A limiter with capacity must still respect inner backpressure.
        let config = Config::new(None).with_grpc_rate_limit(Some(1), Some(60));
        let limiter = GrpcRateLimiter::from_config(&config);
        let mut with_capacity = GrpcRateLimitService::new(PendingInnerService::new(), limiter);
        assert!(
            with_capacity.poll_ready(&mut cx).is_pending(),
            "limiter with capacity should defer to the pending inner service",
        );
    }

    #[tokio::test]
    async fn grpc_rate_limit_call_rejects_after_poll_ready_short_circuit_despite_window_rollover() {
        // Regression: when `poll_ready` reports synthetic readiness for an
        // exhausted limiter, it does NOT poll the inner service. If the
        // rate-limit window then rolls over before `call`, the request must
        // still be rejected rather than forwarded to an inner service that
        // never reported readiness (a Tower contract violation).
        let config = Config::new(None).with_grpc_rate_limit(Some(1), Some(60));
        let limiter = GrpcRateLimiter::from_config(&config).expect("limiter should be enabled");
        // Exhaust the single token.
        assert!(limiter.allow());

        // Pending inner service: its `poll_ready` never reports ready and its
        // `call` increments a counter. A ready result from the wrapper
        // therefore proves the limiter short-circuited rather than delegating,
        // and `calls == 0` proves the wrapper never forwarded.
        let inner = PendingInnerService::new();
        let calls = inner.calls.clone();
        let mut service = GrpcRateLimitService::new(inner, Some(limiter.clone()));

        // poll_ready short-circuits the exhausted limiter and records synthetic
        // readiness without polling the inner service.
        let mut cx = Context::from_waker(std::task::Waker::noop());
        assert!(
            matches!(service.poll_ready(&mut cx), Poll::Ready(Ok(()))),
            "exhausted limiter should report ready despite a pending inner service",
        );

        // The window rolls over between poll_ready and call: the limiter now
        // has capacity again.
        {
            let mut state = limiter
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.window_started = state
                .window_started
                .checked_sub(Duration::from_secs(61))
                .expect("test window rewind should be valid");
        }

        let response = service.call(Request::new(())).await.unwrap();
        assert_eq!(grpc_status_from_response(&response), "8");
        assert_eq!(
            calls.load(Ordering::Relaxed),
            0,
            "inner service must not be called when poll_ready short-circuited the limiter",
        );
    }

    #[tokio::test]
    async fn grpc_rate_limit_returns_resource_exhausted_after_limit() {
        let config = Config::new(None).with_grpc_rate_limit(Some(1), Some(60));
        let limiter = GrpcRateLimiter::from_config(&config);
        let calls = Arc::new(AtomicUsize::new(0));
        let mut service = GrpcRateLimitService::new(
            CountingGrpcService {
                calls: calls.clone(),
            },
            limiter,
        );

        let first = service
            .ready()
            .await
            .unwrap()
            .call(Request::new(()))
            .await
            .unwrap();
        assert_eq!(grpc_status_from_response(&first), "0");

        let second = service
            .ready()
            .await
            .unwrap()
            .call(Request::new(()))
            .await
            .unwrap();
        assert_eq!(grpc_status_from_response(&second), "8");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn grpc_rate_limit_disabled_passes_requests_through() {
        let config = Config::new(None).with_grpc_rate_limit(Some(0), Some(60));
        let limiter = GrpcRateLimiter::from_config(&config);
        let calls = Arc::new(AtomicUsize::new(0));
        let mut service = GrpcRateLimitService::new(
            CountingGrpcService {
                calls: calls.clone(),
            },
            limiter,
        );

        for _ in 0..3 {
            let response = service
                .ready()
                .await
                .unwrap()
                .call(Request::new(()))
                .await
                .unwrap();
            assert_eq!(grpc_status_from_response(&response), "0");
        }
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn grpc_rate_limit_resets_after_window() {
        let config = Config::new(None).with_grpc_rate_limit(Some(1), Some(60));
        let limiter = GrpcRateLimiter::from_config(&config).expect("limiter should be enabled");
        let calls = Arc::new(AtomicUsize::new(0));
        let mut service = GrpcRateLimitService::new(
            CountingGrpcService {
                calls: calls.clone(),
            },
            Some(limiter.clone()),
        );

        let first = service
            .ready()
            .await
            .unwrap()
            .call(Request::new(()))
            .await
            .unwrap();
        assert_eq!(grpc_status_from_response(&first), "0");

        {
            let mut state = limiter
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.window_started = state
                .window_started
                .checked_sub(Duration::from_secs(61))
                .expect("test window rewind should be valid");
        }

        let second = service
            .ready()
            .await
            .unwrap()
            .call(Request::new(()))
            .await
            .unwrap();
        assert_eq!(grpc_status_from_response(&second), "0");
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn grpc_rate_limit_state_is_shared_across_service_clones() {
        let config = Config::new(None).with_grpc_rate_limit(Some(1), Some(60));
        let limiter = GrpcRateLimiter::from_config(&config);
        let calls = Arc::new(AtomicUsize::new(0));
        let mut first_service = GrpcRateLimitService::new(
            CountingGrpcService {
                calls: calls.clone(),
            },
            limiter.clone(),
        );
        let mut second_service = GrpcRateLimitService::new(
            CountingGrpcService {
                calls: calls.clone(),
            },
            limiter,
        );

        let first = first_service
            .ready()
            .await
            .unwrap()
            .call(Request::new(()))
            .await
            .unwrap();
        assert_eq!(grpc_status_from_response(&first), "0");

        let second = second_service
            .ready()
            .await
            .unwrap()
            .call(Request::new(()))
            .await
            .unwrap();
        assert_eq!(grpc_status_from_response(&second), "8");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[derive(Clone)]
    struct TraceBuf(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for TraceBuf {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn request_id_appears_in_trace_span() {
        use tracing_subscriber::fmt::format::FmtSpan;
        use tracing_subscriber::layer::SubscriberExt;

        let log_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let writer = TraceBuf(log_buf.clone());

        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .with_span_events(FmtSpan::CLOSE);

        let subscriber = tracing_subscriber::registry().with(fmt_layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        let req = Request::builder()
            .uri("/test-path")
            .header("x-request-id", "trace-test-id-12345")
            .body(Empty::<Bytes>::new())
            .unwrap();
        let span = make_request_span(&req);
        drop(span.enter());
        drop(span);

        let output = String::from_utf8(log_buf.lock().unwrap().clone()).unwrap();
        assert!(
            output.contains("trace-test-id-12345"),
            "trace output should contain the request_id recorded in the span, got: {output}"
        );
    }

    #[test]
    fn grpc_method_extracts_last_segment() {
        assert_eq!(
            grpc_method_from_path("/openshell.v1.OpenShell/CreateSandbox"),
            "CreateSandbox"
        );
    }

    #[test]
    fn grpc_method_extracts_inference_service() {
        assert_eq!(
            grpc_method_from_path("/openshell.inference.v1.Inference/GetInferenceBundle"),
            "GetInferenceBundle"
        );
    }

    #[test]
    fn grpc_method_handles_bare_path() {
        assert_eq!(grpc_method_from_path("Health"), "Health");
    }

    #[test]
    fn grpc_method_handles_single_slash() {
        assert_eq!(grpc_method_from_path("/"), "");
    }

    #[test]
    fn grpc_method_handles_empty_string() {
        assert_eq!(grpc_method_from_path(""), "");
    }

    #[test]
    fn normalize_ws_tunnel() {
        assert_eq!(normalize_http_path("/_ws_tunnel"), "/_ws_tunnel");
    }

    #[test]
    fn normalize_ws_tunnel_with_trailing() {
        assert_eq!(normalize_http_path("/_ws_tunnel/foo"), "/_ws_tunnel");
    }

    #[test]
    fn normalize_auth_path() {
        assert_eq!(normalize_http_path("/auth/connect"), "/auth");
    }

    #[test]
    fn normalize_auth_with_query() {
        assert_eq!(
            normalize_http_path("/auth/connect?callback_port=12345&code=AB7-X9KM"),
            "/auth"
        );
    }

    #[test]
    fn normalize_unknown_path_collapses_to_unknown() {
        assert_eq!(normalize_http_path("/random/scanner/probe"), "unknown");
    }

    #[test]
    fn normalize_empty_path() {
        assert_eq!(normalize_http_path(""), "unknown");
    }

    #[test]
    fn normalize_root_path() {
        assert_eq!(normalize_http_path("/"), "unknown");
    }

    mod auth_router {
        use super::*;
        use crate::auth::authenticator::test_support::MockAuthenticator;
        use crate::auth::identity::{Identity, IdentityProvider};
        use crate::auth::principal::{
            Principal, SandboxIdentitySource, SandboxPrincipal, UserPrincipal,
        };
        use http_body_util::Full;
        use std::sync::Arc;
        use std::sync::Mutex;
        use tower::Service;

        type RecordedPrincipal = Arc<Mutex<Option<Principal>>>;

        /// Service that snapshots the `Principal` from request extensions
        /// and returns 200 OK. Used by router-level tests to assert the
        /// chain's effect on the downstream service.
        #[derive(Clone)]
        struct PrincipalRecorder {
            recorded: RecordedPrincipal,
        }

        impl PrincipalRecorder {
            fn new() -> (Self, RecordedPrincipal) {
                let recorded = Arc::new(Mutex::new(None));
                (
                    Self {
                        recorded: recorded.clone(),
                    },
                    recorded,
                )
            }
        }

        impl<B: Send + 'static> Service<Request<B>> for PrincipalRecorder {
            type Response = Response<tonic::body::Body>;
            type Error = std::convert::Infallible;
            type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

            fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
                Poll::Ready(Ok(()))
            }

            fn call(&mut self, req: Request<B>) -> Self::Future {
                let principal = req.extensions().get::<Principal>().cloned();
                *self.recorded.lock().unwrap() = principal;
                Box::pin(async move { Ok(Response::new(tonic::body::Body::empty())) })
            }
        }

        fn empty_request(path: &str) -> Request<Full<Bytes>> {
            Request::builder()
                .uri(path)
                .body(Full::new(Bytes::new()))
                .unwrap()
        }

        fn grpc_status<B>(res: &Response<B>) -> Option<String> {
            res.headers()
                .get("grpc-status")
                .map(|v| v.to_str().unwrap().to_string())
        }

        fn user_principal(subject: &str) -> Principal {
            Principal::User(UserPrincipal {
                identity: Identity {
                    subject: subject.to_string(),
                    display_name: None,
                    roles: vec![],
                    scopes: vec![],
                    provider: IdentityProvider::Oidc,
                },
            })
        }

        fn mtls_identity(subject: &str) -> Identity {
            Identity {
                subject: subject.to_string(),
                display_name: Some(subject.to_string()),
                roles: vec!["openshell-user".to_string()],
                scopes: vec![],
                provider: IdentityProvider::Mtls,
            }
        }

        fn sandbox_principal() -> Principal {
            Principal::Sandbox(SandboxPrincipal {
                sandbox_id: "sandbox-a".to_string(),
                source: SandboxIdentitySource::BootstrapJwt {
                    issuer: "openshell-gateway:test".to_string(),
                },
                trust_domain: Some("openshell".to_string()),
            })
        }

        #[tokio::test]
        async fn mtls_peer_identity_fills_missing_principal_when_enabled() {
            let mock = Arc::new(MockAuthenticator::returning(Ok(None)));
            let chain = AuthenticatorChain::new(vec![mock]);
            let (recorder, seen) = PrincipalRecorder::new();
            let mut router = AuthGrpcRouter::with_peer_identity(
                recorder,
                Some(chain),
                None,
                Some(mtls_identity("openshell-client")),
                true,
                false,
            );

            let res = router
                .call(empty_request("/openshell.v1.OpenShell/ListSandboxes"))
                .await
                .unwrap();

            assert_eq!(res.status(), 200);
            let principal = seen.lock().unwrap().clone().expect("principal");
            match principal {
                Principal::User(u) => {
                    assert_eq!(u.identity.subject, "openshell-client");
                    assert_eq!(u.identity.provider, IdentityProvider::Mtls);
                }
                other => panic!("expected mTLS user principal, got {other:?}"),
            }
        }

        #[tokio::test]
        async fn mtls_peer_identity_authenticates_without_chain_when_enabled() {
            let (recorder, seen) = PrincipalRecorder::new();
            let mut router = AuthGrpcRouter::with_peer_identity(
                recorder,
                None,
                None,
                Some(mtls_identity("openshell-client")),
                true,
                false,
            );

            let res = router
                .call(empty_request("/openshell.v1.OpenShell/ListSandboxes"))
                .await
                .unwrap();

            assert_eq!(res.status(), 200);
            assert!(matches!(
                seen.lock().unwrap().as_ref(),
                Some(Principal::User(_))
            ));
        }

        #[tokio::test]
        async fn mtls_auth_enabled_requires_peer_identity() {
            let (recorder, seen) = PrincipalRecorder::new();
            let mut router =
                AuthGrpcRouter::with_peer_identity(recorder, None, None, None, true, false);

            let res = router
                .call(empty_request("/openshell.v1.OpenShell/ListSandboxes"))
                .await
                .unwrap();

            assert!(seen.lock().unwrap().is_none());
            assert_eq!(grpc_status(&res).as_deref(), Some("16"));
        }

        #[tokio::test]
        async fn unauthenticated_dev_user_fills_missing_principal_when_enabled() {
            let mock = Arc::new(MockAuthenticator::returning(Ok(None)));
            let chain = AuthenticatorChain::new(vec![mock]);
            let (recorder, seen) = PrincipalRecorder::new();
            let mut router =
                AuthGrpcRouter::with_peer_identity(recorder, Some(chain), None, None, false, true);

            let res = router
                .call(empty_request("/openshell.v1.OpenShell/ListSandboxes"))
                .await
                .unwrap();

            assert_eq!(res.status(), 200);
            let principal = seen.lock().unwrap().clone().expect("principal");
            match principal {
                Principal::User(u) => {
                    assert_eq!(u.identity.subject, "unauthenticated-local-dev");
                    assert_eq!(u.identity.provider, IdentityProvider::LocalDev);
                }
                other => panic!("expected dev user principal, got {other:?}"),
            }
        }

        #[tokio::test]
        async fn unauthenticated_dev_user_authenticates_without_chain_when_enabled() {
            let (recorder, seen) = PrincipalRecorder::new();
            let mut router =
                AuthGrpcRouter::with_peer_identity(recorder, None, None, None, false, true);

            let res = router
                .call(empty_request("/openshell.v1.OpenShell/ListSandboxes"))
                .await
                .unwrap();

            assert_eq!(res.status(), 200);
            assert!(matches!(
                seen.lock().unwrap().as_ref(),
                Some(Principal::User(user))
                    if user.identity.subject == "unauthenticated-local-dev"
            ));
        }

        #[tokio::test]
        async fn user_principal_lands_in_request_extensions() {
            let mock = Arc::new(MockAuthenticator::returning(Ok(Some(user_principal(
                "alice",
            )))));
            let chain = AuthenticatorChain::new(vec![mock]);
            let (recorder, seen) = PrincipalRecorder::new();
            let mut router = AuthGrpcRouter::new(recorder, Some(chain), None);
            let _ = router
                .call(empty_request("/openshell.v1.OpenShell/ListSandboxes"))
                .await
                .unwrap();
            let principal = seen.lock().unwrap().clone().expect("principal");
            match principal {
                Principal::User(u) => assert_eq!(u.identity.subject, "alice"),
                _ => panic!("expected user principal"),
            }
        }

        #[tokio::test]
        async fn sandbox_principal_lands_in_request_extensions() {
            let mock = Arc::new(MockAuthenticator::returning(Ok(Some(sandbox_principal()))));
            let chain = AuthenticatorChain::new(vec![mock]);
            let (recorder, seen) = PrincipalRecorder::new();
            let mut router = AuthGrpcRouter::new(recorder, Some(chain), None);
            let _ = router
                .call(empty_request("/openshell.v1.OpenShell/ReportPolicyStatus"))
                .await
                .unwrap();
            let captured = seen.lock().unwrap().clone();
            match captured {
                Some(Principal::Sandbox(p)) => assert_eq!(p.sandbox_id, "sandbox-a"),
                other => panic!("expected sandbox principal, got {other:?}"),
            }
        }

        #[tokio::test]
        async fn sandbox_principal_can_call_allowlisted_method() {
            let mock = Arc::new(MockAuthenticator::returning(Ok(Some(sandbox_principal()))));
            let chain = AuthenticatorChain::new(vec![mock]);
            let (recorder, seen) = PrincipalRecorder::new();
            let mut router = AuthGrpcRouter::new(recorder, Some(chain), None);

            let res = router
                .call(empty_request("/openshell.v1.OpenShell/GetSandboxConfig"))
                .await
                .unwrap();

            assert_eq!(res.status(), 200);
            assert!(matches!(
                seen.lock().unwrap().as_ref(),
                Some(Principal::Sandbox(_))
            ));
        }

        #[tokio::test]
        async fn sandbox_principal_can_fetch_inference_bundle() {
            let mock = Arc::new(MockAuthenticator::returning(Ok(Some(sandbox_principal()))));
            let chain = AuthenticatorChain::new(vec![mock]);
            let (recorder, seen) = PrincipalRecorder::new();
            let mut router = AuthGrpcRouter::new(recorder, Some(chain), None);

            let res = router
                .call(empty_request(
                    "/openshell.inference.v1.Inference/GetInferenceBundle",
                ))
                .await
                .unwrap();

            assert_eq!(res.status(), 200);
            assert!(matches!(
                seen.lock().unwrap().as_ref(),
                Some(Principal::Sandbox(_))
            ));
        }

        /// A user principal — even one carrying `openshell:all` and the
        /// admin role — must not reach a `sandbox`-annotated method. The
        /// router enforces this from the per-handler auth-mode declarations
        /// independent of RBAC.
        #[tokio::test]
        async fn user_principal_is_denied_on_sandbox_only_methods() {
            fn admin_user() -> Principal {
                Principal::User(UserPrincipal {
                    identity: Identity {
                        subject: "admin".to_string(),
                        display_name: None,
                        roles: vec!["openshell-admin".to_string()],
                        scopes: vec!["openshell:all".to_string()],
                        provider: IdentityProvider::Oidc,
                    },
                })
            }

            let policy = AuthzPolicy {
                admin_role: "openshell-admin".to_string(),
                user_role: "openshell-user".to_string(),
                scopes_enabled: true,
            };

            for path in [
                "/openshell.v1.OpenShell/ReportPolicyStatus",
                "/openshell.v1.OpenShell/PushSandboxLogs",
                "/openshell.v1.OpenShell/SubmitPolicyAnalysis",
                "/openshell.v1.OpenShell/GetSandboxProviderEnvironment",
                "/openshell.v1.OpenShell/ConnectSupervisor",
                "/openshell.v1.OpenShell/RelayStream",
                "/openshell.v1.OpenShell/IssueSandboxToken",
                "/openshell.v1.OpenShell/RefreshSandboxToken",
                "/openshell.inference.v1.Inference/GetInferenceBundle",
            ] {
                let mock = Arc::new(MockAuthenticator::returning(Ok(Some(admin_user()))));
                let chain = AuthenticatorChain::new(vec![mock]);
                let (recorder, seen) = PrincipalRecorder::new();
                let mut router = AuthGrpcRouter::new(recorder, Some(chain), Some(policy.clone()));

                let res = router.call(empty_request(path)).await.unwrap();

                assert!(seen.lock().unwrap().is_none(), "{path} reached handler");
                // grpc-status=7 (PERMISSION_DENIED).
                assert_eq!(grpc_status(&res).as_deref(), Some("7"), "{path}");
            }
        }

        #[tokio::test]
        async fn sandbox_principal_is_denied_on_user_and_admin_methods() {
            for path in [
                "/openshell.v1.OpenShell/ListSandboxes",
                "/openshell.v1.OpenShell/DeleteSandbox",
                "/openshell.v1.OpenShell/CreateProvider",
                "/openshell.v1.OpenShell/ApproveDraftChunk",
                "/openshell.inference.v1.Inference/GetClusterInference",
                "/openshell.inference.v1.Inference/SetClusterInference",
            ] {
                let mock = Arc::new(MockAuthenticator::returning(Ok(Some(sandbox_principal()))));
                let chain = AuthenticatorChain::new(vec![mock]);
                let (recorder, seen) = PrincipalRecorder::new();
                let mut router = AuthGrpcRouter::new(recorder, Some(chain), None);

                let res = router.call(empty_request(path)).await.unwrap();

                assert!(seen.lock().unwrap().is_none(), "{path} reached handler");
                assert_eq!(grpc_status(&res).as_deref(), Some("7"), "{path}");
            }
        }

        #[tokio::test]
        async fn missing_principal_returns_unauthenticated() {
            let mock = Arc::new(MockAuthenticator::returning(Ok(None)));
            let chain = AuthenticatorChain::new(vec![mock]);
            let (recorder, seen) = PrincipalRecorder::new();
            let mut router = AuthGrpcRouter::new(recorder, Some(chain), None);
            let res = router
                .call(empty_request("/openshell.v1.OpenShell/ListSandboxes"))
                .await
                .unwrap();
            assert!(seen.lock().unwrap().is_none());
            // tonic sets grpc-status=16 (UNAUTHENTICATED) in trailers.
            assert_eq!(grpc_status(&res).as_deref(), Some("16"));
        }

        #[tokio::test]
        async fn authenticator_error_short_circuits() {
            let mock = Arc::new(MockAuthenticator::returning(Err(
                tonic::Status::unauthenticated("forged"),
            )));
            let chain = AuthenticatorChain::new(vec![mock]);
            let (recorder, seen) = PrincipalRecorder::new();
            let mut router = AuthGrpcRouter::new(recorder, Some(chain), None);
            let res = router
                .call(empty_request("/openshell.v1.OpenShell/ListSandboxes"))
                .await
                .unwrap();
            assert!(seen.lock().unwrap().is_none());
            assert_eq!(grpc_status(&res).as_deref(), Some("16"));
        }

        #[tokio::test]
        async fn health_methods_bypass_chain() {
            // Authenticator is wired to fail-closed; the request still gets
            // through because the path is exempt.
            let mock = Arc::new(MockAuthenticator::returning(Err(
                tonic::Status::unauthenticated("would reject"),
            )));
            let chain = AuthenticatorChain::new(vec![mock.clone()]);
            let (recorder, _) = PrincipalRecorder::new();
            let mut router = AuthGrpcRouter::new(recorder, Some(chain), None);
            let res = router
                .call(empty_request("/openshell.v1.OpenShell/Health"))
                .await
                .unwrap();
            assert_eq!(res.status(), 200);
            assert_eq!(mock.call_count(), 0, "health must not consult the chain");
        }
    }

    // ── W3C trace context extraction ────────────────────────────────────

    #[test]
    fn header_extractor_returns_known_header() {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            "traceparent",
            HeaderValue::from_static("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
        );
        let extractor = HeaderExtractor(&headers);
        assert_eq!(
            opentelemetry::propagation::Extractor::get(&extractor, "traceparent"),
            Some("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
        );
    }

    #[test]
    fn header_extractor_returns_none_for_absent_header() {
        let headers = http::HeaderMap::new();
        let extractor = HeaderExtractor(&headers);
        assert_eq!(
            opentelemetry::propagation::Extractor::get(&extractor, "traceparent"),
            None,
        );
    }

    #[test]
    fn header_extractor_returns_none_for_non_utf8_value() {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            "traceparent",
            HeaderValue::from_bytes(b"\xff\xfe").expect("raw bytes are valid HeaderValue"),
        );
        let extractor = HeaderExtractor(&headers);
        assert_eq!(
            opentelemetry::propagation::Extractor::get(&extractor, "traceparent"),
            None,
            "non-UTF-8 header values should be silently ignored"
        );
    }

    #[test]
    fn header_extractor_keys_lists_all_header_names() {
        let mut headers = http::HeaderMap::new();
        headers.insert("traceparent", HeaderValue::from_static("value1"));
        headers.insert("tracestate", HeaderValue::from_static("value2"));
        let extractor = HeaderExtractor(&headers);
        let keys = opentelemetry::propagation::Extractor::keys(&extractor);
        assert!(keys.contains(&"traceparent"));
        assert!(keys.contains(&"tracestate"));
    }

    #[test]
    fn traceparent_extraction_with_valid_header_produces_remote_context() {
        // Register the W3C propagator for this test.
        opentelemetry::global::set_text_map_propagator(
            opentelemetry_sdk::propagation::TraceContextPropagator::new(),
        );

        let mut headers = http::HeaderMap::new();
        headers.insert(
            "traceparent",
            HeaderValue::from_static("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
        );
        let parent_cx = opentelemetry::global::get_text_map_propagator(|p| {
            p.extract(&HeaderExtractor(&headers))
        });

        let span_ctx = parent_cx.span().span_context().clone();
        assert!(span_ctx.is_remote(), "extracted context should be remote");
        assert!(span_ctx.is_valid(), "extracted context should be valid");
        assert_eq!(
            format!("{:032x}", span_ctx.trace_id()),
            "4bf92f3577b34da6a3ce929d0e0e4736"
        );
    }

    #[test]
    fn traceparent_extraction_with_garbage_produces_empty_context() {
        opentelemetry::global::set_text_map_propagator(
            opentelemetry_sdk::propagation::TraceContextPropagator::new(),
        );

        let mut headers = http::HeaderMap::new();
        headers.insert(
            "traceparent",
            HeaderValue::from_static("not-a-valid-traceparent"),
        );
        let parent_cx = opentelemetry::global::get_text_map_propagator(|p| {
            p.extract(&HeaderExtractor(&headers))
        });

        let span_ctx = parent_cx.span().span_context().clone();
        assert!(
            !span_ctx.is_valid(),
            "garbage traceparent should produce invalid context"
        );
    }

    #[test]
    fn traceparent_extraction_without_header_produces_empty_context() {
        opentelemetry::global::set_text_map_propagator(
            opentelemetry_sdk::propagation::TraceContextPropagator::new(),
        );

        let headers = http::HeaderMap::new();
        let parent_cx = opentelemetry::global::get_text_map_propagator(|p| {
            p.extract(&HeaderExtractor(&headers))
        });

        let span_ctx = parent_cx.span().span_context().clone();
        assert!(
            !span_ctx.is_valid(),
            "absent traceparent should produce invalid context"
        );
    }

    // ── gRPC span naming and the untraced skip-list ──────────────────────

    fn grpc_request(path: &str) -> Request<Empty<Bytes>> {
        Request::builder()
            .uri(path)
            .header("content-type", "application/grpc")
            .body(Empty::<Bytes>::new())
            .expect("valid request")
    }

    /// Capture span close events from `make_request_span` into a string.
    fn span_trace_output(req: &Request<Empty<Bytes>>) -> (String, bool) {
        use tracing_subscriber::fmt::format::FmtSpan;
        use tracing_subscriber::layer::SubscriberExt;

        let log_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let writer = TraceBuf(log_buf.clone());
        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .with_span_events(FmtSpan::CLOSE);
        let subscriber = tracing_subscriber::registry().with(fmt_layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        let span = make_request_span(req);
        let is_none = span.is_none();
        drop(span.enter());
        drop(span);

        let output = String::from_utf8(log_buf.lock().expect("lock").clone()).expect("utf8");
        (output, is_none)
    }

    #[test]
    fn grpc_service_and_method_splits_qualified_path() {
        assert_eq!(
            grpc_service_and_method("/openshell.v1.OpenShell/GetSandbox"),
            Some(("openshell.v1.OpenShell", "GetSandbox"))
        );
        assert_eq!(
            grpc_service_and_method("/openshell.inference.v1.Inference/GetInferenceBundle"),
            Some(("openshell.inference.v1.Inference", "GetInferenceBundle"))
        );
    }

    #[test]
    fn grpc_service_and_method_rejects_malformed_paths() {
        assert_eq!(grpc_service_and_method("Health"), None);
        assert_eq!(grpc_service_and_method("/Health"), None);
        assert_eq!(grpc_service_and_method("/"), None);
        assert_eq!(grpc_service_and_method(""), None);
        assert_eq!(grpc_service_and_method("/openshell.v1.OpenShell/"), None);
    }

    #[test]
    fn is_grpc_request_keys_off_content_type() {
        assert!(is_grpc_request(&grpc_request(
            "/openshell.v1.OpenShell/GetSandbox"
        )));

        let plain = Request::builder()
            .uri("/healthz")
            .body(Empty::<Bytes>::new())
            .expect("valid request");
        assert!(!is_grpc_request(&plain));
    }

    #[test]
    fn grpc_span_is_named_by_rpc_method() {
        let (output, is_none) =
            span_trace_output(&grpc_request("/openshell.v1.OpenShell/GetSandbox"));

        assert!(!is_none, "non-health gRPC calls must be traced");
        assert!(
            output.contains("otel.name=\"openshell.v1.OpenShell/GetSandbox\""),
            "exported span name should follow the OTel RPC semconv, got: {output}"
        );
        assert!(
            output.contains("rpc.service=\"openshell.v1.OpenShell\""),
            "span should carry rpc.service, got: {output}"
        );
        assert!(
            output.contains("rpc.method=\"GetSandbox\""),
            "span should carry rpc.method, got: {output}"
        );
    }

    #[test]
    fn health_rpc_produces_no_span() {
        let (output, is_none) = span_trace_output(&grpc_request("/openshell.v1.OpenShell/Health"));

        assert!(is_none, "Health is on the untraced skip-list");
        assert!(
            output.is_empty(),
            "Health must not emit a span at all, got: {output}"
        );
    }

    #[test]
    fn http_requests_keep_the_generic_request_span() {
        let req = Request::builder()
            .uri("/auth/connect")
            .body(Empty::<Bytes>::new())
            .expect("valid request");
        let (output, is_none) = span_trace_output(&req);

        assert!(!is_none, "HTTP requests are still traced");
        assert!(
            !output.contains("otel.name"),
            "HTTP spans keep their static name, got: {output}"
        );
    }
}
