import os

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from opentelemetry.instrumentation.langchain import LangchainInstrumentor

def configure_observability(app: FastAPI) -> None:
    """Configure OpenTelemetry tracing and instrument the FastAPI application."""

    resource = Resource.create(
        {
            "service.name": os.getenv("OTEL_SERVICE_NAME", "joke-agent-lab"),
            "service.version": "0.1.0",
            "deployment.environment.name": os.getenv(
                "OTEL_ENVIRONMENT", "local"
            ),
        }
    )

    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(
        endpoint=os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "http://127.0.0.1:4317",
        ),
        insecure=True,
    )

    provider.add_span_processor(
        BatchSpanProcessor(exporter)
    )

    trace.set_tracer_provider(provider)

    # Automatically trace LangChain and LangGraph operations.
    LangchainInstrumentor().instrument()

    FastAPIInstrumentor.instrument_app(app)