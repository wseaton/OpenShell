// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared AWS constants for the container-credentials emulator, provider env
//! injection, and credential resolution.
//!
//! This module is the single source of truth for AWS naming: the emulator
//! endpoint, env var aliases, provider config keys, and the internal
//! credential blob key. `openshell-server`, `openshell-providers`, and
//! `openshell-sandbox` import from here.
//!
//! Unlike the GCP metadata emulator (which serves a *placeholder* token that
//! the egress proxy rewrites at request time), the AWS emulator serves the
//! *real* short-lived STS credentials. AWS `SigV4` signs requests locally
//! inside the sandbox, so there is no bearer token in the outbound request for
//! the proxy to rewrite — the SDK needs the real secret to compute the
//! signature.

use serde::{Deserialize, Serialize};

/// The credential payload served by the container-credentials endpoint.
///
/// This is the exact JSON shape the ECS/EKS-pod-identity endpoint returns, so
/// the field names are dictated by AWS (hence the `rename`s). It is the shared
/// contract between the gateway (which builds it from the STS response and
/// stores it as the credential blob) and the sandbox emulator (which validates
/// the blob and serves it). Centralizing it keeps the field names in one place
/// instead of duplicated string literals across crates.
///
/// The leaf fields are `String` on purpose: an access key, secret, session
/// token, and ISO-8601 expiration are opaque AWS-issued values with no internal
/// structure to validate, so newtypes would add ceremony without safety.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContainerCredentials {
    #[serde(rename = "AccessKeyId")]
    pub access_key_id: String,
    #[serde(rename = "SecretAccessKey")]
    pub secret_access_key: String,
    #[serde(rename = "Token")]
    pub token: String,
    #[serde(rename = "Expiration")]
    pub expiration: String,
}

// ── Container-credentials emulator ──────────────────────────────────────────

/// Hostname advertised to the AWS SDK at provider-injection time.
///
/// Set as `AWS_CONTAINER_CREDENTIALS_FULL_URI`; the sandbox resolver rewrites
/// it to the loopback URI ([`CONTAINER_CREDS_FULL_URI`]) before the child
/// process starts. Its presence in `child_env` also gates emulator startup.
pub const METADATA_HOST: &str = "aws.metadata.openshell.internal";

/// Loopback address the container-credentials emulator binds inside sandbox
/// network namespaces. Distinct from the GCP emulator's `:8174`.
pub const CONTAINER_CREDS_LOOPBACK_ADDR: &str = "127.0.0.1:8175";

/// Path the emulator serves credentials from.
pub const CONTAINER_CREDS_PATH: &str = "/creds";

/// Full loopback URI the AWS SDK uses to fetch container credentials. The AWS
/// SDK allows an unauthenticated `AWS_CONTAINER_CREDENTIALS_FULL_URI` only when
/// it points at loopback, which this does.
pub const CONTAINER_CREDS_FULL_URI: &str = "http://127.0.0.1:8175/creds";

/// Env var the AWS SDK reads to discover the container-credentials endpoint.
pub const CONTAINER_CREDS_FULL_URI_ENV: &str = "AWS_CONTAINER_CREDENTIALS_FULL_URI";

// ── Env var alias arrays ────────────────────────────────────────────────────

/// Env vars that carry the AWS region inside sandboxes. Resolved to real values
/// in the child env so the SDK picks a regional endpoint at startup.
pub const REGION_ENV_VARS: &[&str] = &["AWS_REGION", "AWS_DEFAULT_REGION"];

// ── Provider config keys ────────────────────────────────────────────────────

/// Config key for the role ARN assumed via web identity.
pub const ROLE_ARN_CONFIG_KEY: &str = "role_arn";

/// Config key for the web-identity token file the gateway reads (and re-reads
/// on refresh, since kubelet rotates the projected `ServiceAccount` token).
pub const WEB_IDENTITY_TOKEN_FILE_CONFIG_KEY: &str = "web_identity_token_file";

/// Config key for the AWS region.
pub const REGION_CONFIG_KEY: &str = "region";

/// Config key for the STS role session name (optional).
pub const SESSION_NAME_CONFIG_KEY: &str = "session_name";

// ── Internal credential blob ────────────────────────────────────────────────

/// Internal env key holding the gateway-minted STS response as a JSON blob.
///
/// The blob is `{AccessKeyId, SecretAccessKey, SessionToken, Expiration}`. It
/// is never an AWS SDK env var and is stripped from the child env; only the
/// emulator resolves it (via the `SecretResolver`) and serves it as
/// container-credentials JSON.
pub const CREDENTIALS_JSON_ENV: &str = "AWS_CONTAINER_CREDENTIALS_JSON";

/// AWS SDK static-credential env vars.
///
/// If any of these are present in the child env, the SDK credential chain uses
/// them *before* the container endpoint, so they must be stripped — otherwise
/// the SDK would sign with placeholder garbage. We never set them, but strip
/// defensively in case a profile or user config introduces them.
pub const STATIC_CREDENTIAL_ENV_VARS: &[&str] = &[
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
];

/// Non-secret AWS config vars resolved to real values in the child env so the
/// SDK can read them at process startup. Must equal the union of the region
/// aliases above (enforced by a unit test).
pub const STATIC_CONFIG_KEYS: &[&str] = &["AWS_REGION", "AWS_DEFAULT_REGION"];

/// Keys removed from the child env when AWS provider injection is active.
///
/// Covers the internal JSON blob plus any static credential vars. The emulator
/// resolves the blob out of band; the SDK must fall through to the container
/// endpoint.
pub const CHILD_ENV_KEYS_TO_STRIP: &[&str] = &[
    CREDENTIALS_JSON_ENV,
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn static_config_keys_matches_region_aliases() {
        let expected: HashSet<&str> = REGION_ENV_VARS.iter().copied().collect();
        let actual: HashSet<&str> = STATIC_CONFIG_KEYS.iter().copied().collect();
        assert_eq!(
            expected, actual,
            "STATIC_CONFIG_KEYS must equal the region alias array",
        );
    }

    #[test]
    fn strip_keys_cover_blob_and_static_creds() {
        let strip: HashSet<&str> = CHILD_ENV_KEYS_TO_STRIP.iter().copied().collect();
        assert!(strip.contains(CREDENTIALS_JSON_ENV));
        for var in STATIC_CREDENTIAL_ENV_VARS {
            assert!(strip.contains(var), "{var} must be stripped from child env");
        }
    }

    #[test]
    fn container_credentials_serialize_with_aws_field_names() {
        let creds = ContainerCredentials {
            access_key_id: "ASIA".to_string(),
            secret_access_key: "secret".to_string(),
            token: "tok".to_string(),
            expiration: "2026-06-24T12:00:00Z".to_string(),
        };
        let json = serde_json::to_string(&creds).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["AccessKeyId"], "ASIA");
        assert_eq!(value["SecretAccessKey"], "secret");
        assert_eq!(value["Token"], "tok");
        assert_eq!(value["Expiration"], "2026-06-24T12:00:00Z");

        let round_tripped: ContainerCredentials = serde_json::from_str(&json).unwrap();
        assert_eq!(round_tripped, creds);
    }

    #[test]
    fn full_uri_points_at_loopback_addr_and_path() {
        assert_eq!(
            CONTAINER_CREDS_FULL_URI,
            format!("http://{CONTAINER_CREDS_LOOPBACK_ADDR}{CONTAINER_CREDS_PATH}"),
        );
    }
}
