"""
Model Registry - All selected coding models with metadata and inference config.

Excluded: microsoft/Phi-4-mini-instruct (not code-specific)
"""

MODELS = [
    {
        "id": "qwen2.5-coder-1.5b",
        "hf_model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "family": "Qwen2",
        "architecture": "Qwen2 Transformer (GQA, RoPE, SwiGLU)",
        "params": "1.5B",
        "params_active": "1.5B",
        "dense_or_moe": "dense",
        "context_length": 131072,
        "code_specific": True,
        "chat_template": "qwen",
    },
    {
        "id": "qwen2.5-coder-3b",
        "hf_model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "family": "Qwen2",
        "architecture": "Qwen2 Transformer (GQA, RoPE, SwiGLU)",
        "params": "3B",
        "params_active": "3B",
        "dense_or_moe": "dense",
        "context_length": 131072,
        "code_specific": True,
        "chat_template": "qwen",
    },
    {
        "id": "qwen2.5-coder-7b",
        "hf_model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "family": "Qwen2",
        "architecture": "Qwen2 Transformer (GQA, RoPE, SwiGLU)",
        "params": "7.6B",
        "params_active": "7.6B",
        "dense_or_moe": "dense",
        "context_length": 131072,
        "code_specific": True,
        "chat_template": "qwen",
    },
    {
        "id": "deepseek-coder-1.3b",
        "hf_model_id": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "family": "DeepSeek",
        "architecture": "LLaMA-style Transformer (dense)",
        "params": "1.3B",
        "params_active": "1.3B",
        "dense_or_moe": "dense",
        "context_length": 16384,
        "code_specific": True,
        "chat_template": "deepseek",
    },
    {
        "id": "deepseek-coder-6.7b",
        "hf_model_id": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "family": "DeepSeek",
        "architecture": "LLaMA-style Transformer (dense)",
        "params": "6.7B",
        "params_active": "6.7B",
        "dense_or_moe": "dense",
        "context_length": 16384,
        "code_specific": True,
        "chat_template": "deepseek",
    },
    {
        "id": "deepseek-coder-v2-lite",
        "hf_model_id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "family": "DeepSeek-V2",
        "architecture": "DeepSeekMoE + Multi-head Latent Attention",
        "params": "16B",
        "params_active": "2.4B",
        "dense_or_moe": "moe",
        "context_length": 131072,
        "code_specific": True,
        "chat_template": "deepseek-v2",
    },
    {
        "id": "codellama-7b",
        "hf_model_id": "meta-llama/CodeLlama-7b-Instruct-hf",
        "family": "LLaMA2",
        "architecture": "LLaMA 2 Transformer",
        "params": "7B",
        "params_active": "7B",
        "dense_or_moe": "dense",
        "context_length": 16384,
        "code_specific": True,
        "chat_template": "llama",
    },
    {
        "id": "mamba-codestral-7b",
        "hf_model_id": "mistralai/Mamba-Codestral-7B-v0.1",
        "family": "Mamba",
        "architecture": "Mamba2 State Space Model",
        "params": "7B",
        "params_active": "7B",
        "dense_or_moe": "dense",
        "context_length": None,  # SSM - theoretically unlimited
        "code_specific": True,
        "chat_template": "codestral",
        "special_requirements": ["mamba-ssm", "causal-conv1d"],
        # Needs Ampere or newer, and the reason is not performance. Mamba2's scan
        # runs through mamba_chunk_scan_combined, a Triton kernel compiled at
        # submission time, and in bfloat16 it emits .bf16 PTX instructions that
        # ptxas rejects below sm_80. On this cluster that rules out p100 (sm_60),
        # v100 (sm_70) and rtx8000 (sm_75), so roughly half the GPU fleet fails at
        # kernel compilation rather than producing worse output. Submit with
        # --gres=gpu:l40s:1, or a100/a6000/h100, otherwise the job is a coin flip.
        # This is Triton JIT, so TORCH_CUDA_ARCH_LIST at build time has no effect
        # on it -- rebuilding the ahead-of-time kernels more widely was tried first
        # and changed nothing.
        "min_compute_capability": 8.0,
    },
    {
        "id": "starcoder2-3b",
        "hf_model_id": "TechxGenus/starcoder2-3b-instruct",
        "family": "StarCoder2",
        "architecture": "StarCoder2 Transformer (multi-query attention)",
        "params": "3B",
        "params_active": "3B",
        "dense_or_moe": "dense",
        "context_length": 16384,
        "code_specific": True,
        "chat_template": "starcoder",
    },
    {
        "id": "starcoder2-7b",
        "hf_model_id": "TechxGenus/starcoder2-7b-instruct",
        "family": "StarCoder2",
        "architecture": "StarCoder2 Transformer (multi-query attention)",
        "params": "7B",
        "params_active": "7B",
        "dense_or_moe": "dense",
        "context_length": 16384,
        "code_specific": True,
        "chat_template": "starcoder",
    },
    {
        "id": "yi-coder-1.5b",
        "hf_model_id": "01-ai/Yi-Coder-1.5B-Chat",
        "family": "Yi",
        "architecture": "Yi/LLaMA-style Transformer",
        "params": "1.5B",
        "params_active": "1.5B",
        "dense_or_moe": "dense",
        "context_length": 131072,
        "code_specific": True,
        "chat_template": "yi",
    },
    {
        "id": "yi-coder-9b",
        "hf_model_id": "01-ai/Yi-Coder-9B-Chat",
        "family": "Yi",
        "architecture": "Yi/LLaMA-style Transformer",
        "params": "9B",
        "params_active": "9B",
        "dense_or_moe": "dense",
        "context_length": 131072,
        "code_specific": True,
        "chat_template": "yi",
    },
    {
        "id": "granite-3b-code",
        "hf_model_id": "ibm-granite/granite-3b-code-instruct-128k",
        "family": "Granite",
        "architecture": "LLaMA-based Transformer",
        "params": "3B",
        "params_active": "3B",
        "dense_or_moe": "dense",
        "context_length": 131072,
        "code_specific": True,
        "chat_template": "granite",
    },
    {
        "id": "granite-8b-code",
        "hf_model_id": "ibm-granite/granite-8b-code-instruct-128k",
        "family": "Granite",
        "architecture": "LLaMA-based Transformer",
        "params": "8B",
        "params_active": "8B",
        "dense_or_moe": "dense",
        "context_length": 131072,
        "code_specific": True,
        "chat_template": "granite",
    },
]

# Standardized inference settings for fair comparison
INFERENCE_CONFIG = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 2048,
    "repetition_penalty": 1.0,
    "do_sample": False,  # Greedy decoding for deterministic output
    "dtype": "bfloat16",
    "quantization": None,  # FP16/BF16 by default
}

SYSTEM_PROMPT = (
    "You are an expert Python developer. Generate clean, complete Python code "
    "based on the given requirements. Output only the Python code without "
    "explanations or markdown formatting."
)


def get_model_by_id(model_id: str) -> dict:
    for m in MODELS:
        if m["id"] == model_id:
            return m
    raise ValueError(f"Model '{model_id}' not found in registry")


def list_model_ids() -> list:
    return [m["id"] for m in MODELS]
