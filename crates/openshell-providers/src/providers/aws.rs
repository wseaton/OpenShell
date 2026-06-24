// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use openshell_core::aws;

use crate::{
    DiscoveredProvider, Provider, ProviderDiscoverySpec, ProviderError, ProviderPlugin,
    RealDiscoveryContext, discover_with_spec,
};

/// AWS credential provider.
///
/// The gateway runs `AssumeRoleWithWebIdentity` and refreshes the resulting
/// short-lived STS credentials (stored as a single JSON blob under
/// [`aws::CREDENTIALS_JSON_ENV`]). Inside the sandbox, a container-credentials
/// emulator serves those real temp creds to the AWS SDK. `inject_env` sets the
/// discovery env vars; the sandbox resolver later rewrites the full-URI to the
/// loopback emulator and strips the blob from the child env.
pub struct AwsProvider;

const SPEC: ProviderDiscoverySpec = ProviderDiscoverySpec {
    id: "aws",
    // The credential is gateway-minted, not discovered from the local
    // environment, but the registry still needs to know its key.
    credential_env_vars: &[aws::CREDENTIALS_JSON_ENV],
};

impl ProviderPlugin for AwsProvider {
    fn id(&self) -> &'static str {
        SPEC.id
    }

    fn discover_existing(&self) -> Result<Option<DiscoveredProvider>, ProviderError> {
        discover_with_spec(&SPEC, &RealDiscoveryContext)
    }

    fn credential_env_vars(&self) -> &'static [&'static str] {
        SPEC.credential_env_vars
    }

    fn inject_env(&self, provider: &Provider, env: &mut HashMap<String, String>) {
        // Presence of the full-URI var gates emulator startup in the sandbox.
        // The real loopback URI is substituted by the resolver before the child
        // process starts (`child_env_resolved`).
        env.entry(aws::CONTAINER_CREDS_FULL_URI_ENV.to_string())
            .or_insert_with(|| aws::METADATA_HOST.to_string());

        if let Some(region) = provider
            .config
            .get(aws::REGION_CONFIG_KEY)
            .filter(|v| !v.trim().is_empty())
        {
            for var in aws::REGION_ENV_VARS {
                env.entry((*var).to_string())
                    .or_insert_with(|| region.trim().to_string());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_provider(config: HashMap<String, String>) -> Provider {
        Provider {
            config,
            r#type: "aws".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn injects_container_creds_uri_marker() {
        let provider = make_provider(HashMap::new());
        let mut env = HashMap::new();
        AwsProvider.inject_env(&provider, &mut env);

        assert_eq!(
            env.get(aws::CONTAINER_CREDS_FULL_URI_ENV)
                .map(String::as_str),
            Some(aws::METADATA_HOST)
        );
    }

    #[test]
    fn injects_region_aliases() {
        let provider = make_provider(HashMap::from([(
            "region".to_string(),
            "us-east-1".to_string(),
        )]));
        let mut env = HashMap::new();
        AwsProvider.inject_env(&provider, &mut env);

        assert_eq!(env.get("AWS_REGION").map(String::as_str), Some("us-east-1"));
        assert_eq!(
            env.get("AWS_DEFAULT_REGION").map(String::as_str),
            Some("us-east-1")
        );
    }

    #[test]
    fn does_not_overwrite_existing_env() {
        let provider = make_provider(HashMap::from([(
            "region".to_string(),
            "us-west-2".to_string(),
        )]));
        let mut env = HashMap::from([("AWS_REGION".to_string(), "eu-west-1".to_string())]);
        AwsProvider.inject_env(&provider, &mut env);

        assert_eq!(
            env.get("AWS_REGION").map(String::as_str),
            Some("eu-west-1"),
            "should not overwrite existing env"
        );
    }

    #[test]
    fn skips_empty_region() {
        let provider = make_provider(HashMap::from([("region".to_string(), "  ".to_string())]));
        let mut env = HashMap::new();
        AwsProvider.inject_env(&provider, &mut env);

        assert!(!env.contains_key("AWS_REGION"));
    }
}
