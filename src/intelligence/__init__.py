"""
Intelligence module for Fleming-AI
Self-improving adaptive collection and optimization
"""

from src.intelligence.adaptive_collector import (
    AdaptiveCollector,
    FeedbackLoop,
    MetricsTracker,
    ThresholdOptimizer,
)

__all__ = [
    "AdaptiveCollector",
    "MetricsTracker",
    "ThresholdOptimizer",
    "FeedbackLoop",
]
