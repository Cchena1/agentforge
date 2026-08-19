from __future__ import annotations

import os

# Unit/integration tests deliberately use the deterministic embedding test double.
os.environ.setdefault("AI_AGENT_ENVIRONMENT", "test")
os.environ.setdefault("AI_AGENT_EMBEDDING_PROVIDER", "hash")
os.environ.setdefault("AI_AGENT_EMBEDDING_DIMENSION", "384")
