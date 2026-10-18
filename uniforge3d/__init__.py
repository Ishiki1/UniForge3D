"""UniForge3D multi-view image-to-3D inference."""

__all__ = ["UniForge3DImageTo3DPipeline"]


def __getattr__(name):
    # Operator-only users must not need the pipeline's CUDA mesh/model stack.
    if name == "UniForge3DImageTo3DPipeline":
        from .pipelines import UniForge3DImageTo3DPipeline

        globals()[name] = UniForge3DImageTo3DPipeline
        return UniForge3DImageTo3DPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
