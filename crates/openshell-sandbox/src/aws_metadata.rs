// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AWS container-credentials emulator for sandbox credential injection.
//!
//! Implements the ECS/EKS-pod-identity container-credentials endpoint so that
//! AWS SDKs can obtain short-lived STS credentials natively inside sandboxes.
//! The AWS SDK discovers it via `AWS_CONTAINER_CREDENTIALS_FULL_URI`, which the
//! sandbox resolver points at the loopback emulator.
//!
//! Unlike the GCP metadata emulator (which serves a placeholder token resolved
//! at egress), this emulator serves the **real** gateway-minted temp creds:
//! AWS `SigV4` signs requests locally inside the sandbox, so the SDK needs the
//! real secret. The creds are short-lived and role-scoped, and the gateway
//! refreshes them before expiry.
//!
//! The emulator runs as a loopback HTTP server inside the sandbox network
//! namespace (see [`metadata_server`](crate::metadata_server)).

use miette::{IntoDiagnostic, Result};
use openshell_core::provider_credentials::ProviderCredentialState;
use openshell_ocsf::{ActivityId, HttpActivityBuilder, SeverityId, StatusId, ocsf_emit};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};

type MetadataResponse = (u16, &'static str, String);

#[derive(Debug, Clone)]
pub struct AwsMetadataContext {
    credentials: ProviderCredentialState,
}

impl AwsMetadataContext {
    pub fn new(credentials: ProviderCredentialState) -> Self {
        Self { credentials }
    }
}

impl crate::metadata_server::MetadataHandler for AwsMetadataContext {
    async fn handle<S: AsyncRead + AsyncWrite + Unpin + Send>(
        &self,
        method: &str,
        path: &str,
        _request: &[u8],
        stream: &mut S,
    ) -> Result<()> {
        let (status, content_type, body) = route_request(self, method, path);
        write_response(stream, status, content_type, &body).await
    }
}

fn route_request(ctx: &AwsMetadataContext, method: &str, path: &str) -> MetadataResponse {
    if method != "GET" {
        emit_event(
            ActivityId::Refuse,
            SeverityId::Low,
            StatusId::Failure,
            &format!("aws-metadata: unsupported method {method}"),
        );
        return (405, "text/plain", "Method Not Allowed".to_string());
    }

    let route = path.split('?').next().unwrap_or(path);
    let route = route.strip_suffix('/').unwrap_or(route);

    // The SDK requests the exact path from AWS_CONTAINER_CREDENTIALS_FULL_URI.
    // Serve creds from that path and the root for resilience to trailing slash.
    if route == openshell_core::aws::CONTAINER_CREDS_PATH || route.is_empty() {
        handle_credentials(ctx)
    } else {
        emit_event(
            ActivityId::Refuse,
            SeverityId::Low,
            StatusId::Failure,
            &format!("aws-metadata: unknown path {route}"),
        );
        (
            404,
            "application/json",
            serde_json::json!({"error": "not_found"}).to_string(),
        )
    }
}

fn handle_credentials(ctx: &AwsMetadataContext) -> MetadataResponse {
    let Some(blob) = ctx.credentials.aws_credentials_json() else {
        emit_event(
            ActivityId::Fail,
            SeverityId::Medium,
            StatusId::Failure,
            "aws-metadata: credentials request but none available",
        );
        return (
            503,
            "application/json",
            serde_json::json!({"error": "credentials_unavailable"}).to_string(),
        );
    };

    // Validate the gateway-minted blob against the shared contract before
    // serving it, so a malformed payload surfaces as a 503 rather than handing
    // the SDK garbage. Re-serializing normalizes the JSON the SDK receives.
    let Ok(creds) = serde_json::from_str::<openshell_core::aws::ContainerCredentials>(&blob) else {
        emit_event(
            ActivityId::Fail,
            SeverityId::Medium,
            StatusId::Failure,
            "aws-metadata: stored credentials blob is malformed",
        );
        return (
            503,
            "application/json",
            serde_json::json!({"error": "credentials_malformed"}).to_string(),
        );
    };

    let Ok(body) = serde_json::to_string(&creds) else {
        return (
            503,
            "application/json",
            serde_json::json!({"error": "credentials_unavailable"}).to_string(),
        );
    };

    emit_event(
        ActivityId::Open,
        SeverityId::Informational,
        StatusId::Success,
        "aws-metadata: temp credentials served",
    );
    (200, "application/json", body)
}

async fn write_response<S>(
    client: &mut S,
    status: u16,
    content_type: &str,
    body: &str,
) -> Result<()>
where
    S: AsyncWrite + Unpin,
{
    let response = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        status_text(status),
        body.len(),
    );
    client
        .write_all(response.as_bytes())
        .await
        .into_diagnostic()?;
    client.flush().await.into_diagnostic()?;
    Ok(())
}

fn status_text(status: u16) -> &'static str {
    match status {
        404 => "Not Found",
        405 => "Method Not Allowed",
        503 => "Service Unavailable",
        _ => "OK",
    }
}

fn emit_event(activity: ActivityId, severity: SeverityId, status: StatusId, message: &str) {
    let event = HttpActivityBuilder::new(crate::ocsf_ctx())
        .activity(activity)
        .severity(severity)
        .status(status)
        .message(message.to_string())
        .build();
    ocsf_emit!(event);
}

#[cfg(test)]
mod tests {
    use super::*;
    use openshell_core::aws;
    use std::collections::HashMap;

    fn make_context(env: HashMap<String, String>) -> AwsMetadataContext {
        let state =
            ProviderCredentialState::from_environment(0, env, HashMap::new(), HashMap::new());
        AwsMetadataContext::new(state)
    }

    fn creds_env() -> HashMap<String, String> {
        HashMap::from([(
            aws::CREDENTIALS_JSON_ENV.to_string(),
            r#"{"AccessKeyId":"ASIA","SecretAccessKey":"secret","Token":"tok","Expiration":"2026-06-24T12:00:00Z"}"#
                .to_string(),
        )])
    }

    #[test]
    fn serves_real_credentials_at_creds_path() {
        let ctx = make_context(creds_env());
        let (status, ct, body) = route_request(&ctx, "GET", aws::CONTAINER_CREDS_PATH);
        assert_eq!(status, 200);
        assert_eq!(ct, "application/json");
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["AccessKeyId"], "ASIA");
        assert_eq!(json["SecretAccessKey"], "secret");
        assert_eq!(json["Token"], "tok");
        assert_eq!(json["Expiration"], "2026-06-24T12:00:00Z");
    }

    #[test]
    fn ignores_query_string() {
        let ctx = make_context(creds_env());
        let path = format!("{}?versionId=1", aws::CONTAINER_CREDS_PATH);
        let (status, _, _) = route_request(&ctx, "GET", &path);
        assert_eq!(status, 200);
    }

    #[test]
    fn no_credentials_returns_503() {
        let ctx = make_context(HashMap::new());
        let (status, _, _) = route_request(&ctx, "GET", aws::CONTAINER_CREDS_PATH);
        assert_eq!(status, 503);
    }

    #[test]
    fn malformed_blob_returns_503() {
        let ctx = make_context(HashMap::from([(
            aws::CREDENTIALS_JSON_ENV.to_string(),
            r#"{"AccessKeyId":"only-one-field"}"#.to_string(),
        )]));
        let (status, _, body) = route_request(&ctx, "GET", aws::CONTAINER_CREDS_PATH);
        assert_eq!(status, 503);
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["error"], "credentials_malformed");
    }

    #[test]
    fn unknown_path_returns_404() {
        let ctx = make_context(creds_env());
        let (status, _, _) = route_request(&ctx, "GET", "/other");
        assert_eq!(status, 404);
    }

    #[test]
    fn post_method_returns_405() {
        let ctx = make_context(creds_env());
        let (status, _, _) = route_request(&ctx, "POST", aws::CONTAINER_CREDS_PATH);
        assert_eq!(status, 405);
    }
}
