# ADR-007: Tenant-Scoped RAG Authorization Boundary

- **Status:** Accepted
- **Decision date:** 2026-08-13
- **Scope:** document ingestion, active-version registry, SQLite retrieval, Qdrant retrieval, HTTP document APIs, and legacy migration

## Context

Version-correct retrieval is still unsafe if a query can score another tenant's vectors or expose a restricted source through metadata. The authorization decision must therefore occur before candidate scoring, not after top-k retrieval. The design must also preserve existing public-only callers and persisted data through an additive migration window.

The service does not yet authenticate callers. Consequently, `tenant_id` and `principals` can only be treated as authorization context supplied by a trusted upstream identity boundary. Exposing those fields to the language model would let the model select its own permissions and is rejected.

## First-principles invariants

1. A document's logical identity is `(tenant_id, source_id)`.
2. The same `source_id` may exist independently in multiple tenants.
3. Empty ACL means tenant-wide visibility; a non-empty ACL requires principal intersection.
4. Unauthorized chunks must be removed before vector scoring or decoding.
5. Unauthorized active versions must not appear in retrieval metadata.
6. Authorization changes are versioned; they cannot silently reuse a stale content-only version.
7. Legacy data remains compatible only in the `public` tenant and with an empty ACL.
8. The model cannot choose tenant or principal context.

## Decision

### Public contracts

Add defaulted fields without removing existing fields:

- ingestion: `tenant_id="public"`, `acl=[]`;
- retrieval: `tenant_id="public"`, `principals=[]`.

The API layer passes the context to the RAG service. Production deployments must overwrite or derive this context from authenticated middleware. The `search_knowledge` function-calling tool remains public-tenant only until trusted request identity is carried through graph state.

### Version and registry identity

The immutable version ID hashes a canonical JSON object containing source ID, content hash, pipeline profile, tenant ID, and sorted ACL. Canonical JSON avoids delimiter-collision ambiguity.

A new `rag_source_scopes` table owns the active pointer under the composite primary key `(tenant_id, source_id)`. `rag_document_versions_v2` references that composite scope, avoiding the original table's global `source_id` foreign-key assumption. Legacy `rag_sources` and `rag_document_versions` remain during the 0.x migration window. Existing public state is copied and reconciled into v2, and new public records are dual-written so a temporary rollback to the prior public-only reader remains recoverable. No destructive rewrite is performed.

### Retrieval enforcement

The RAG service first reads the tenant-scoped active snapshot and removes ACL-ineligible versions. It sends two distinct sets to the vector backend:

- authorized active versions that may be returned;
- all tenant-active source IDs whose legacy unversioned rows must be suppressed.

This distinction prevents a restricted active source from falling back to an older unversioned row for an unauthorized caller.

SQLite applies tenant, ACL, source, and version predicates in SQL before embedding JSON is loaded and decoded. Qdrant applies equivalent payload filters inside `query_points`, before vector candidates are returned to application code.

### Legacy behavior

- SQLite additive columns default legacy chunks to tenant `public` and ACL `[]`.
- Legacy registry rows are copied to `rag_source_scopes` and `rag_document_versions_v2` as public records.
- Re-initialization reconciles newer legacy public pointers/version states after a temporary rollback; private-tenant state remains v2-only.
- Qdrant points missing tenant or ACL payload are visible and deletable only through public-tenant operations during the migration window.

## Rejected alternatives

### Global source ownership

Rejecting reuse of a source ID across tenants simplifies one table but violates normal multitenant identity semantics and creates avoidable naming coupling.

### Post-retrieval ACL filtering

Filtering after top-k can leak timing and metadata, waste candidate slots, and return fewer relevant authorized results. It also violates the requirement that authorization precede scoring.

### Model-controlled tenant/principal tool arguments

An LLM is not an identity provider. Allowing it to set authorization context creates a direct privilege-escalation path.

### Destructive migration

Rewriting or deleting legacy data in place would violate the repository's staged compatibility policy and complicate rollback.

## Consequences

### Positive

- Tenant isolation is enforced consistently in both supported vector adapters.
- Source IDs remain tenant-local names.
- ACL changes are reproducible and visible in version evidence.
- Existing clients remain compatible through public defaults.
- Tests can demonstrate authorization-before-scoring rather than only output filtering.

### Costs and limitations

- ACL-only changes currently re-embed content because authorization is part of immutable version identity. A future policy-version layer may separate content and authorization artifacts if measurements justify the complexity.
- Authentication and trusted identity mapping remain external.
- The chat/tool path is public-tenant only until identity is propagated outside model-controlled arguments.
- Remote Qdrant deployment behavior is not verified; only the in-memory adapter contract is tested.
- Multi-process writers still require a single writer or distributed lease.

## Verification evidence

- `test_cross_tenant_retrieval_isolated_before_scoring`
- `test_same_source_id_is_isolated_per_tenant`
- `test_acl_requires_principal_intersection_and_hides_index_versions`
- `test_restricted_active_version_suppresses_legacy_rows_for_unauthorized_principal`
- `test_sqlite_authorization_filter_runs_before_vector_decoding`
- `test_acl_version_identity_uses_unambiguous_canonical_encoding`
- Qdrant tenant/ACL and legacy-public migration tests
- HTTP additive-contract and validation tests

No new runtime dependency was introduced for this decision.
