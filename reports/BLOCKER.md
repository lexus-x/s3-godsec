# Phase 1 Blocker: CUDA/Jax Conflict (Install Hell)

## Evidence of Failure
While attempting to initialize Option A (`Octo-Small` + `LIBERO-Spatial`) for Phase 1, the agent encountered a fatal JAX/CUDA initialization error. 

The `octo` and `libero` libraries are successfully installed in the environment, and `libero_spatial` loads correctly. However, executing the JAX-based `OctoModel.load_pretrained` triggers a CuDNN version mismatch:

```text
E0430 22:32:43.417007 2493694 cuda_dnn.cc:454] Loaded runtime CuDNN library: 9.1.0 but source was compiled with: 9.8.0. CuDNN library needs to have matching major version and equal or higher minor version.
...
jax.errors.JaxRuntimeError: FAILED_PRECONDITION: DNN library initialization failed.
```

This is the exact "install hell" scenario anticipated for the Octo Jax/Flax stack. The model cannot run forward passes in the current environment configuration. 

## Proposed Scope Change / Next Steps
Since the environment's `CuDNN` runtime (`9.1.0`) is older than the compiled version (`9.8.0`), we have two paths forward:

1. **Fix the Environment (Stay on Option A):** Upgrade the system/conda CuDNN runtime to $\ge$ 9.8.0, or downgrade the installed `jax[cuda]` package to a version compiled against CuDNN 9.1.0.
2. **Pivot to Option B (OpenVLA-7B):** OpenVLA is built on PyTorch/HuggingFace and bypasses the JAX/Flax stack entirely. A background job to download/load OpenVLA was initiated but takes significant time (14GB weights) and may encounter VRAM limitations on the 80GB A100 if we attempt to run it concurrently with the SE(3) head without quantization.

**Awaiting user review:** Please advise whether you prefer to resolve the JAX environment issue to stick with Octo-Small, or if I should formally pivot to Option B and proceed with OpenVLA-7B.
