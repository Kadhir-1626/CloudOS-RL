# ai_engine/data_pipeline/__init__.py
# Lazy import to prevent module-level failures if optional deps are missing.

def get_orchestrator():
    from ai_engine.data_pipeline.pipeline_orchestrator import DataPipelineOrchestrator
    return DataPipelineOrchestrator


__all__ = ["get_orchestrator"]