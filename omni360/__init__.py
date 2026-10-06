"""omni360: evaluation framework for 360-degree image coding.

Reimplements the framework of Yu, Lakshman and Girod, "A Framework to Evaluate
Omnidirectional Video Coding Schemes", IEEE ISMAR 2015: sphere-to-plane
projections, viewport PSNR, (weighted) spherical PSNR and BD-rate.
"""
from .bdrate import bd_psnr, bd_rate
from .metrics import Evaluator, psnr, s_psnr, to_luma, viewport_psnr
from .projections import PROJECTIONS, make_projection
from .sampling import render, sample
from .viewport import HeadMotionStats, render_viewport, synthetic_orientations, viewport_directions

__all__ = [
    "PROJECTIONS", "Evaluator", "HeadMotionStats", "bd_psnr", "bd_rate", "make_projection", "psnr",
    "render", "render_viewport", "s_psnr", "sample", "synthetic_orientations", "to_luma",
    "viewport_directions", "viewport_psnr",
]
