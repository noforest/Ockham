"""Context packs for LLM-based vulnerability detection.

The pipeline has two independent stages: selection decides which repository
functions to keep, representation decides how to encode them. One run crosses
one selector with one representation at a fixed token budget.
"""
