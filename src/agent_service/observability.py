from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    GCCollector,
    Histogram,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)


class JsonlSpanExporter(SpanExporter):
    """Local-first, content-free trace evidence exporter."""

    def __init__(
        self,
        path: Path,
        on_failure: Callable[[], None] | None = None,
        *,
        max_bytes: int = 20 * 1024 * 1024,
        backup_count: int = 5,
    ) -> None:
        self.path = path
        self._on_failure = on_failure
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                json.dumps(_span_record(span), ensure_ascii=False, sort_keys=True) for span in spans
            ]
            payload = "\n".join(lines) + "\n"
            with self._lock:
                self._rotate_if_needed(len(payload.encode("utf-8")))
                with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(payload)
            return SpanExportResult.SUCCESS
        except OSError:
            if self._on_failure is not None:
                self._on_failure()
            return SpanExportResult.FAILURE

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        if not self.path.exists() or self.path.stat().st_size + incoming_bytes <= self._max_bytes:
            return
        oldest = self.path.with_suffix(f"{self.path.suffix}.{self._backup_count}")
        oldest.unlink(missing_ok=True)
        for index in range(self._backup_count - 1, 0, -1):
            source = self.path.with_suffix(f"{self.path.suffix}.{index}")
            if source.exists():
                source.replace(self.path.with_suffix(f"{self.path.suffix}.{index + 1}"))
        self.path.replace(self.path.with_suffix(f"{self.path.suffix}.1"))


class CountingSpanExporter(SpanExporter):
    """Delegates export while turning exporter failures into a bounded metric."""

    def __init__(self, delegate: SpanExporter, on_failure: Callable[[], None]) -> None:
        self._delegate = delegate
        self._on_failure = on_failure

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            result = self._delegate.export(spans)
        except Exception:  # noqa: BLE001 - exporter plugins may raise arbitrary failures
            self._on_failure()
            return SpanExportResult.FAILURE
        if result is not SpanExportResult.SUCCESS:
            self._on_failure()
        return result

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)


class Observability:
    """Owns bounded-cardinality metrics and OpenTelemetry tracing for one app instance."""

    def __init__(
        self,
        *,
        service_name: str,
        service_version: str,
        environment: str,
        state_dir: Path,
        metrics_enabled: bool = True,
        trace_jsonl_enabled: bool = True,
        trace_jsonl_max_bytes: int = 20 * 1024 * 1024,
        trace_jsonl_backup_count: int = 5,
        otlp_endpoint: str | None = None,
    ) -> None:
        self.metrics_enabled = metrics_enabled
        self.registry = CollectorRegistry(auto_describe=True)
        ProcessCollector(registry=self.registry)
        PlatformCollector(registry=self.registry)
        GCCollector(registry=self.registry)
        self.http_requests = Counter(
            "agentforge_http_requests_total",
            "HTTP requests by stable route template and status.",
            ("method", "route", "status"),
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "agentforge_http_request_duration_seconds",
            "HTTP request duration by stable route template.",
            ("method", "route"),
            registry=self.registry,
        )
        self.rag_operations = Counter(
            "agentforge_rag_operations_total",
            "RAG operations by outcome and degradation state.",
            ("operation", "outcome", "degraded"),
            registry=self.registry,
        )
        self.rag_duration = Histogram(
            "agentforge_rag_operation_duration_seconds",
            "RAG operation duration.",
            ("operation",),
            registry=self.registry,
        )
        self.ingestion_transitions = Counter(
            "agentforge_ingestion_transitions_total",
            "Durable ingestion job state transitions.",
            ("status",),
            registry=self.registry,
        )
        self.tool_runtime_events = Counter(
            "agentforge_tool_runtime_events_total",
            "Tool policy, approval, guard and execution outcomes.",
            ("phase", "outcome", "side_effects"),
            registry=self.registry,
        )
        self.model_requests = Counter(
            "agentforge_model_requests_total",
            "Model route requests by bounded outcome and degradation state.",
            ("route", "outcome", "degraded"),
            registry=self.registry,
        )
        self.model_tokens = Counter(
            "agentforge_model_tokens_total",
            "Model tokens by route and token class.",
            ("route", "token_class"),
            registry=self.registry,
        )
        self.model_estimated_cost = Counter(
            "agentforge_model_estimated_cost_usd_total",
            "Configured estimated model cost in USD by route.",
            ("route",),
            registry=self.registry,
        )
        self.failures = Counter(
            "agentforge_failures_total",
            "Normalized failures by component and bounded category.",
            ("component", "category"),
            registry=self.registry,
        )
        self.memory_operations = Counter(
            "agentforge_memory_operations_total",
            "Memory operations by bounded outcome.",
            ("operation", "outcome"),
            registry=self.registry,
        )
        self.parser_fallbacks = Counter(
            "agentforge_parser_fallbacks_total",
            "Parser fallback transitions.",
            registry=self.registry,
        )
        self.backup_last_success = Gauge(
            "agentforge_backup_last_success_unixtime",
            "Unix time of the last verified local backup.",
            registry=self.registry,
        )
        self.restore_verifications = Counter(
            "agentforge_restore_verifications_total",
            "Restore verification outcomes loaded from durable operations status.",
            ("outcome",),
            registry=self.registry,
        )
        self.trace_export_failures = Counter(
            "agentforge_trace_export_failures_total",
            "Local or OTLP trace export failures.",
            ("exporter",),
            registry=self.registry,
        )
        self._load_operations_status(state_dir)

        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": service_version,
                "deployment.environment.name": environment,
            }
        )
        self.tracer_provider = TracerProvider(resource=resource)
        if trace_jsonl_enabled:
            local_exporter = JsonlSpanExporter(
                state_dir / "telemetry" / "traces.jsonl",
                on_failure=lambda: self.trace_export_failures.labels(exporter="jsonl").inc(),
                max_bytes=trace_jsonl_max_bytes,
                backup_count=trace_jsonl_backup_count,
            )
            self.tracer_provider.add_span_processor(SimpleSpanProcessor(local_exporter))
        if otlp_endpoint:
            exporter = CountingSpanExporter(
                OTLPSpanExporter(endpoint=otlp_endpoint),
                lambda: self.trace_export_failures.labels(exporter="otlp").inc(),
            )
            self.tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        self.tracer = self.tracer_provider.get_tracer("agentforge.service", service_version)

    @contextmanager
    def span(self, name: str, attributes: Mapping[str, Any] | None = None) -> Iterator[trace.Span]:
        safe_attributes = {
            str(key): value
            for key, value in (attributes or {}).items()
            if isinstance(value, (bool, float, int, str))
        }
        with self.tracer.start_as_current_span(name, attributes=safe_attributes) as current:
            yield current

    def record_http(self, method: str, route: str, status: int, duration_seconds: float) -> None:
        if not self.metrics_enabled:
            return
        stable_route = route if route.startswith("/") else "unmatched"
        self.http_requests.labels(method=method, route=stable_route, status=str(status)).inc()
        self.http_duration.labels(method=method, route=stable_route).observe(duration_seconds)

    def record_rag(
        self,
        operation: str,
        outcome: str,
        duration_seconds: float,
        *,
        degraded: bool = False,
    ) -> None:
        if not self.metrics_enabled:
            return
        self.rag_operations.labels(
            operation=operation,
            outcome=outcome,
            degraded=str(degraded).lower(),
        ).inc()
        self.rag_duration.labels(operation=operation).observe(duration_seconds)

    def record_ingestion_transition(self, status: str) -> None:
        if not self.metrics_enabled:
            return
        self.ingestion_transitions.labels(status=status).inc()
        if status == "fallback":
            self.parser_fallbacks.inc()

    def record_tool_runtime_event(
        self,
        phase: str,
        outcome: str,
        side_effects: str,
    ) -> None:
        if not self.metrics_enabled:
            return
        self.tool_runtime_events.labels(
            phase=phase,
            outcome=outcome,
            side_effects=side_effects,
        ).inc()

    def record_model_usage(
        self,
        route: str,
        outcome: str,
        *,
        degraded: bool,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_prompt_tokens: int = 0,
        reasoning_tokens: int = 0,
        estimated_cost_usd: float = 0.0,
    ) -> None:
        if not self.metrics_enabled:
            return
        self.model_requests.labels(
            route=route, outcome=outcome, degraded=str(degraded).lower()
        ).inc()
        token_values = {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "cached_prompt": cached_prompt_tokens,
            "reasoning": reasoning_tokens,
        }
        for token_class, value in token_values.items():
            if value:
                self.model_tokens.labels(route=route, token_class=token_class).inc(value)
        if estimated_cost_usd:
            self.model_estimated_cost.labels(route=route).inc(estimated_cost_usd)

    def record_failure(self, component: str, category: str) -> None:
        if self.metrics_enabled:
            self.failures.labels(component=component, category=category).inc()

    def record_memory_operation(self, operation: str, outcome: str) -> None:
        if self.metrics_enabled:
            self.memory_operations.labels(operation=operation, outcome=outcome).inc()

    def render_metrics(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST

    async def shutdown(self) -> None:
        await asyncio.to_thread(self.tracer_provider.force_flush, 5000)
        await asyncio.to_thread(self.tracer_provider.shutdown)

    def _load_operations_status(self, state_dir: Path) -> None:
        path = state_dir / "operations" / "status.json"
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            backup_time = float(payload.get("last_backup_success_unixtime", 0.0))
            if backup_time > 0:
                self.backup_last_success.set(backup_time)
            restore_outcome = payload.get("last_restore_verification")
            if restore_outcome in {"success", "failure"}:
                self.restore_verifications.labels(outcome=str(restore_outcome)).inc()
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            self.restore_verifications.labels(outcome="failure").inc()


def current_trace_id() -> str | None:
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return f"{context.trace_id:032x}"


def _span_record(span: ReadableSpan) -> dict[str, Any]:
    context = span.context
    parent = span.parent
    return {
        "schema_version": 1,
        "name": span.name,
        "trace_id": f"{context.trace_id:032x}" if context is not None else None,
        "span_id": f"{context.span_id:016x}" if context is not None else None,
        "parent_span_id": f"{parent.span_id:016x}" if parent is not None else None,
        "start_time_unix_nano": span.start_time,
        "end_time_unix_nano": span.end_time,
        "status": span.status.status_code.name,
        "attributes": {
            str(key): _json_value(value) for key, value in (span.attributes or {}).items()
        },
        "events": [
            {
                "name": event.name,
                "timestamp_unix_nano": event.timestamp,
                "attributes": {
                    str(key): _json_value(value) for key, value in (event.attributes or {}).items()
                },
            }
            for event in span.events
        ],
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, (bool, float, int, str)) or value is None:
        return value
    if isinstance(value, Sequence):
        return [_json_value(item) for item in value]
    return str(value)
