# Zero Trust Security Specification & IAM Policy Architecture

## 1. Core Principles of Zero Trust Architecture (ZTA)

Zero Trust Architecture operates under the strict assumption that threats exist both inside and outside conventional network perimeters. Perimeter defense models (firewalls and VPNs) are fundamentally deprecated in favor of continuous verification.

### NIST SP 800-207 Tenets
- **Never Trust, Always Verify:** Every access request is fully authenticated, authorized, and encrypted before granting session connectivity.
- **Least Privilege Access:** Users and automated service accounts receive access rights restricted strictly to the explicit resources necessary for their operational role (Just-In-Time and Just-Enough-Access).
- **Assume Breach:** Security telemetry treats every network segment as hostile. Lateral movement is strictly prevented via micro-segmentation, mutual TLS (mTLS), and ephemeral cryptographically signed credentials.

## 2. Cryptographic Identity & Micro-Segmentation

Modern enterprise workloads enforce strict machine-to-machine authentication using SPIFFE/SPIRE identity standards.

### Mutual TLS (mTLS) with Short-Lived Certificates
Services communicating across Kubernetes pods or multi-cloud regions must terminate mTLS with X.509 certificates having validity windows under 24 hours. Automated control planes (such as HashiCorp Vault or cert-manager) rotate private keys transparently without service downtime.

### API Gateways & OAuth 2.0 / OpenID Connect (OIDC)
All ingress API requests must carry cryptographically signed JWT tokens issued by a centralized identity provider (IdP). Claims within the JWT (issuer `iss`, audience `aud`, expiration `exp`, and custom RBAC scopes) are validated locally at the Envoy or Kong gateway before forwarding requests upstream.

## 3. Data Protection & Envelope Encryption

Data at rest within object storage (AWS S3, GCP Cloud Storage) and relational databases is protected using multi-tier envelope encryption.

### Key Management Service (KMS) Hierarchy
1. **Master Key (Customer Managed Key):** Never leaves the Hardware Security Module (HSM).
2. **Data Encryption Key (DEK):** Generated dynamically by KMS to encrypt file blocks using AES-256-GCM.
3. The plaintext DEK is used to encrypt the payload and immediately wiped from volatile memory; only the ciphertext DEK is persisted alongside the encrypted data object.
