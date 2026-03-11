# ai_engine/explainability/__init__.py

def get_explainer(model, background=None):
    from ai_engine.explainability.shap_explainer import SHAPExplainer
    return SHAPExplainer(model, background)


def get_background_generator(config):
    from ai_engine.explainability.background_generator import BackgroundDataGenerator
    return BackgroundDataGenerator(config)


def get_formatter():
    from ai_engine.explainability.explanation_formatter import ExplanationFormatter
    return ExplanationFormatter()


__all__ = ["get_explainer", "get_background_generator", "get_formatter"]