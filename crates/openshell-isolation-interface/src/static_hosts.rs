// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Driver-supplied static host entries.
//!
//! A compute driver may know addresses for hostnames that the network
//! supervisor's trusted resolver cannot answer, such as a routable service
//! with no public DNS record. The driver validates the operator's entries
//! once and carries them to the supervisor in the runtime descriptor. The
//! supervisor answers these names from the table in place of its resolver;
//! every policy and destination check still applies to the returned
//! addresses.

use std::collections::BTreeMap;
use std::fmt;
use std::net::IpAddr;

use serde::{Deserialize, Serialize};

/// Names the supervisor already answers itself. A static entry cannot shadow
/// them.
const RESERVED_HOSTNAMES: &[&str] = &[
    "localhost",
    "policy.local",
    "host.openshell.internal",
    "host.containers.internal",
    "host.docker.internal",
];

const MAX_HOSTNAME_LEN: usize = 253;
const MAX_LABEL_LEN: usize = 63;

/// Validated hostname-to-address table, keyed by lowercase hostname without a
/// trailing dot.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    try_from = "BTreeMap<String, Vec<IpAddr>>",
    into = "BTreeMap<String, Vec<IpAddr>>"
)]
pub struct StaticHosts(BTreeMap<String, Vec<IpAddr>>);

/// Why a static host entry was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StaticHostsError {
    /// The hostname is not a valid DNS name.
    InvalidHostname(String),
    /// The hostname is an IP literal; it needs no entry.
    IpLiteralHostname(String),
    /// The hostname is answered by the supervisor itself.
    ReservedHostname(String),
    /// The address is never a valid destination.
    BlockedAddress { hostname: String, address: IpAddr },
    /// The hostname has no addresses.
    NoAddresses(String),
}

impl fmt::Display for StaticHostsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHostname(name) => write!(f, "invalid static hostname {name:?}"),
            Self::IpLiteralHostname(name) => {
                write!(f, "static hostname {name:?} is an IP literal")
            }
            Self::ReservedHostname(name) => write!(f, "static hostname {name:?} is reserved"),
            Self::BlockedAddress { hostname, address } => write!(
                f,
                "static host {hostname:?} maps to blocked address {address}"
            ),
            Self::NoAddresses(name) => write!(f, "static host {name:?} has no addresses"),
        }
    }
}

impl std::error::Error for StaticHostsError {}

impl StaticHosts {
    /// A table with no entries.
    #[must_use]
    pub const fn empty() -> Self {
        Self(BTreeMap::new())
    }

    /// Build a table from `(hostname, address)` pairs. Repeated hostnames
    /// collect their addresses in order, without duplicates.
    pub fn from_entries<I, S>(entries: I) -> Result<Self, StaticHostsError>
    where
        I: IntoIterator<Item = (S, IpAddr)>,
        S: AsRef<str>,
    {
        let mut table: BTreeMap<String, Vec<IpAddr>> = BTreeMap::new();
        for (hostname, address) in entries {
            let hostname = normalize_hostname(hostname.as_ref())?;
            validate_address(&hostname, address)?;
            let addresses = table.entry(hostname).or_default();
            if !addresses.contains(&address) {
                addresses.push(address);
            }
        }
        Ok(Self(table))
    }

    /// Addresses for `host`, matched case-insensitively and ignoring a
    /// trailing dot or IPv6 brackets.
    #[must_use]
    pub fn lookup(&self, host: &str) -> Option<&[IpAddr]> {
        let key = host
            .trim()
            .trim_start_matches('[')
            .trim_end_matches(']')
            .trim_end_matches('.')
            .to_ascii_lowercase();
        self.0.get(&key).map(Vec::as_slice)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Entries in hostname order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &[IpAddr])> {
        self.0
            .iter()
            .map(|(hostname, addresses)| (hostname.as_str(), addresses.as_slice()))
    }
}

impl TryFrom<BTreeMap<String, Vec<IpAddr>>> for StaticHosts {
    type Error = StaticHostsError;

    fn try_from(value: BTreeMap<String, Vec<IpAddr>>) -> Result<Self, Self::Error> {
        let mut entries = Vec::new();
        for (hostname, addresses) in value {
            if addresses.is_empty() {
                return Err(StaticHostsError::NoAddresses(hostname));
            }
            entries.extend(
                addresses
                    .into_iter()
                    .map(|address| (hostname.clone(), address)),
            );
        }
        Self::from_entries(entries)
    }
}

impl From<StaticHosts> for BTreeMap<String, Vec<IpAddr>> {
    fn from(value: StaticHosts) -> Self {
        value.0
    }
}

fn normalize_hostname(raw: &str) -> Result<String, StaticHostsError> {
    let hostname = raw.trim_end_matches('.').to_ascii_lowercase();
    if hostname.parse::<IpAddr>().is_ok() {
        return Err(StaticHostsError::IpLiteralHostname(raw.to_string()));
    }
    let valid_label = |label: &str| {
        !label.is_empty()
            && label.len() <= MAX_LABEL_LEN
            && !label.starts_with('-')
            && !label.ends_with('-')
            && label
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
    };
    if hostname.is_empty()
        || hostname.len() > MAX_HOSTNAME_LEN
        || !hostname.split('.').all(valid_label)
    {
        return Err(StaticHostsError::InvalidHostname(raw.to_string()));
    }
    if RESERVED_HOSTNAMES.contains(&hostname.as_str())
        || openshell_core::net::is_known_metadata_hostname(&hostname)
    {
        return Err(StaticHostsError::ReservedHostname(raw.to_string()));
    }
    Ok(hostname)
}

fn validate_address(hostname: &str, address: IpAddr) -> Result<(), StaticHostsError> {
    if openshell_core::net::is_always_blocked_ip(address) || address.is_multicast() {
        return Err(StaticHostsError::BlockedAddress {
            hostname: hostname.to_string(),
            address,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::static_hosts::{StaticHosts, StaticHostsError};
    use std::collections::BTreeMap;
    use std::net::IpAddr;

    fn ip(value: &str) -> IpAddr {
        value.parse().expect("valid IP")
    }

    #[test]
    fn entries_normalize_and_merge_per_hostname() {
        let hosts = StaticHosts::from_entries([
            ("MAAS.Example.com.", ip("150.239.114.201")),
            ("maas.example.com", ip("150.239.114.202")),
            ("maas.example.com", ip("150.239.114.201")),
            ("db.internal.example", ip("10.0.0.8")),
        ])
        .expect("valid entries");

        assert_eq!(
            hosts.lookup("maas.example.com"),
            Some([ip("150.239.114.201"), ip("150.239.114.202")].as_slice())
        );
        assert_eq!(
            hosts.lookup("db.internal.example"),
            Some([ip("10.0.0.8")].as_slice())
        );
        assert_eq!(hosts.iter().count(), 2);
    }

    #[test]
    fn lookup_ignores_case_trailing_dot_and_brackets() {
        let hosts =
            StaticHosts::from_entries([("maas.example.com", ip("150.239.114.201"))]).expect("ok");
        for query in [
            "maas.example.com",
            "MAAS.EXAMPLE.COM",
            "maas.example.com.",
            "[maas.example.com]",
        ] {
            assert!(hosts.lookup(query).is_some(), "{query}");
        }
        assert!(hosts.lookup("evil.maas.example.com").is_none());
        assert!(hosts.lookup("maas.example.com.evil").is_none());
        assert!(hosts.lookup("").is_none());
    }

    #[test]
    fn invalid_hostnames_are_rejected() {
        for name in [
            "",
            ".",
            "a..b",
            "-lead.example",
            "trail-.example",
            "under_score.example",
            "space .example",
            "*.example.com",
            &format!("{}.example", "a".repeat(64)),
            &format!("{}.example", ["a"; 127].join(".")),
        ] {
            let err = StaticHosts::from_entries([(name, ip("10.0.0.1"))]).unwrap_err();
            assert!(
                matches!(err, StaticHostsError::InvalidHostname(_)),
                "{name}: {err}"
            );
        }
    }

    #[test]
    fn ip_literal_hostnames_are_rejected() {
        for name in ["10.0.0.1", "::1", "2001:db8::1"] {
            let err = StaticHosts::from_entries([(name, ip("10.0.0.2"))]).unwrap_err();
            assert!(
                matches!(err, StaticHostsError::IpLiteralHostname(_)),
                "{name}: {err}"
            );
        }
    }

    #[test]
    fn reserved_hostnames_cannot_be_shadowed() {
        for name in [
            "localhost",
            "LOCALHOST.",
            "policy.local",
            "host.openshell.internal",
            "host.containers.internal",
            "host.docker.internal",
            "metadata.google.internal",
        ] {
            let err = StaticHosts::from_entries([(name, ip("10.0.0.1"))]).unwrap_err();
            assert!(
                matches!(err, StaticHostsError::ReservedHostname(_)),
                "{name}: {err}"
            );
        }
    }

    #[test]
    fn always_blocked_and_multicast_addresses_are_rejected() {
        for address in [
            "127.0.0.1",
            "0.0.0.0",
            "169.254.169.254",
            "::1",
            "::",
            "fe80::1",
            "::ffff:127.0.0.1",
            "224.0.0.1",
            "ff02::1",
        ] {
            let err = StaticHosts::from_entries([("svc.example", ip(address))]).unwrap_err();
            assert!(
                matches!(err, StaticHostsError::BlockedAddress { .. }),
                "{address}: {err}"
            );
        }
    }

    #[test]
    fn private_and_public_addresses_are_accepted() {
        for address in [
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "fd00::1",
            "8.8.8.8",
        ] {
            StaticHosts::from_entries([("svc.example", ip(address))])
                .unwrap_or_else(|err| panic!("{address}: {err}"));
        }
    }

    #[test]
    fn serde_round_trips_and_validates_on_decode() {
        let hosts = StaticHosts::from_entries([
            ("b.example", ip("10.0.0.2")),
            ("a.example", ip("10.0.0.1")),
        ])
        .expect("ok");
        let json = serde_json::to_string(&hosts).expect("encode");
        assert_eq!(
            json,
            r#"{"a.example":["10.0.0.1"],"b.example":["10.0.0.2"]}"#
        );
        let decoded: StaticHosts = serde_json::from_str(&json).expect("decode");
        assert_eq!(decoded, hosts);

        let normalized: StaticHosts =
            serde_json::from_str(r#"{"A.Example.":["10.0.0.1"]}"#).expect("decode");
        assert!(normalized.lookup("a.example").is_some());

        for invalid in [
            r#"{"localhost":["10.0.0.1"]}"#,
            r#"{"a.example":["127.0.0.1"]}"#,
            r#"{"a.example":[]}"#,
            r#"{"bad_name":["10.0.0.1"]}"#,
        ] {
            assert!(
                serde_json::from_str::<StaticHosts>(invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn empty_table_has_no_entries() {
        let hosts = StaticHosts::default();
        assert!(hosts.is_empty());
        assert!(hosts.lookup("anything.example").is_none());
        let decoded: StaticHosts =
            serde_json::from_value(serde_json::Value::Object(serde_json::Map::new()))
                .expect("decode");
        assert_eq!(decoded, hosts);
        let map: BTreeMap<String, Vec<IpAddr>> = hosts.into();
        assert!(map.is_empty());
    }
}
