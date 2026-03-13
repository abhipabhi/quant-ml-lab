"""
Market Regime Detection Package

Provides tools for detecting global market regimes using
Hidden Markov Models on multi-market financial data.
"""

from .current_world_market_regime import WorldMarketRegimeDetector

__all__ = ["WorldMarketRegimeDetector"]