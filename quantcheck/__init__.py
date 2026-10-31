"""QuantCheck — a statistically honest degradation test for local quantized models.

QuantCheck wraps the established paired per-sample degradation test (exact
one-sided McNemar / binomial sign test on disagreements; see Kübler et al. 2026)
in a one-command CLI for hobbyists comparing a baseline model against a quantized
candidate. The statistical method is not novel; the tool's contribution is
engineering and accessibility. See README.md for full framing and citations.
"""

__version__ = "0.1.0"
