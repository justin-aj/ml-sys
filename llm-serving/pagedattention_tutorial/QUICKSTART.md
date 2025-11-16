# PagedAttention Quick Start Guide

## Table of Contents
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Common Patterns](#common-patterns)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

---

## Expected Performance

**PagedAttention's memory savings on V100 32GB:**

### Memory Efficiency

1. **Variable-length requests (100 users)**
   - Standard approach wastes ~77% of memory via pre-allocation
   - PagedAttention allocates exactly what's needed (~0% waste)

2. **Throughput scaling on V100**
   - How many concurrent users can a 32GB V100 serve?
   - Standard: ~25 users | PagedAttention: ~100 users (4×!)

3. **Prefix sharing (system prompts)**
   - 100 users sharing a 500-token system prompt
   - PagedAttention saves ~83% memory via block sharing

### Typical Results

On a **V100 32GB**, you should see:

| Metric | Standard | PagedAttention | Improvement |
|--------|----------|----------------|-------------|
| Memory waste | 77% | ~0% | 77% saved |
| Concurrent users | ~25 | ~100 | 4× more |
| With prefix sharing | 60K tokens | 10.5K tokens | 5.7× compression |

**Why this matters:** These memory savings translate directly to serving 4× more users per GPU in production!

---

## Installation

### Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU)
- 16GB+ GPU memory (for 7B models)
- 40GB+ GPU memory (for 13B models)
- 80GB+ GPU memory (for 70B models)

### Install vLLM

```bash
# Install from PyPI (recommended)
pip install vllm

# Or install from source for latest features
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

### Verify Installation

```python
import vllm
print(vllm.__version__)  # Should print version (e.g., 0.2.0)

# Check CUDA availability
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Your GPU name
```

---

## Basic Usage

### Example 1: Single Request

```python
from vllm import LLM, SamplingParams

# Initialize the model with PagedAttention
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    # PagedAttention is enabled by default!
)

# Define generation parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100,
)

# Generate
prompts = ["Hello, my name is"]
outputs = llm.generate(prompts, sampling_params)

# Print result
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
```

**Output:**
```
Prompt: Hello, my name is
Generated: John Smith and I am a software engineer...
```

### Example 2: Batch Requests

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
sampling_params = SamplingParams(temperature=0.8, max_tokens=50)

# Process multiple prompts in one batch
# Continuous batching automatically applied!
prompts = [
    "The future of AI is",
    "Once upon a time",
    "In a galaxy far away",
    "Explain quantum computing:",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    print(f"{prompt} → {generated}\n")
```

### Example 3: Parallel Sampling

```python
from vllm import LLM, SamplingParams

llm = LLM(model="gpt2")

# Generate multiple outputs for each prompt
# All outputs share the prompt's KV cache (thanks to PagedAttention!)
sampling_params = SamplingParams(
    n=5,  # Generate 5 different outputs
    temperature=0.9,
    max_tokens=50,
)

prompts = ["Tell me a joke:"]
outputs = llm.generate(prompts, sampling_params)

# Print all 5 outputs
for i, completion in enumerate(outputs[0].outputs, 1):
    print(f"Joke {i}: {completion.text}\n")
```

---

## Common Patterns

### Pattern 1: Streaming Generation

```python
from vllm import LLM, SamplingParams

llm = LLM(model="gpt2")
sampling_params = SamplingParams(max_tokens=100)

# Stream tokens as they're generated
prompt = "Once upon a time"
for output in llm.generate([prompt], sampling_params, use_tqdm=False):
    # Print each new token
    text = output.outputs[0].text
    if text:
        print(text[-1], end='', flush=True)

print()  # Newline at end
```

### Pattern 2: Custom Sampling

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Different sampling strategies
configs = [
    ("Greedy", SamplingParams(temperature=0)),
    ("Sampling", SamplingParams(temperature=0.8, top_p=0.95)),
    ("Top-K", SamplingParams(temperature=0.8, top_k=40)),
    ("Beam Search", SamplingParams(best_of=5, use_beam_search=True)),
]

prompt = "The meaning of life is"

for name, params in configs:
    output = llm.generate([prompt], params)[0]
    print(f"{name}: {output.outputs[0].text}\n")
```

### Pattern 3: Chat Completion (OpenAI-style)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")

# Format as chat
def format_chat(messages):
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
    prompt += "Assistant: "
    return prompt

messages = [
    {"role": "user", "content": "What is the capital of France?"},
]

prompt = format_chat(messages)
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

output = llm.generate([prompt], sampling_params)[0]
print(output.outputs[0].text)
```

### Pattern 4: Batch Processing with Different Lengths

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Different max_tokens for each prompt
prompts = [
    "Write a short poem:",
    "Explain relativity in detail:",
    "Hi",
]

# Use different sampling params
params_list = [
    SamplingParams(max_tokens=50),   # Short poem
    SamplingParams(max_tokens=200),  # Detailed explanation
    SamplingParams(max_tokens=10),   # Quick response
]

# Process all at once
# PagedAttention allocates exactly what's needed for each sequence
for prompt, params in zip(prompts, params_list):
    output = llm.generate([prompt], params)[0]
    print(f"Prompt: {prompt}")
    print(f"Output: {output.outputs[0].text}\n")
```

---

## Performance Tips

### Tip 1: Configure Block Size

```python
from vllm import LLM

# Default block_size=16 is optimal for most cases
llm = LLM(
    model="meta-llama/Llama-2-13b-hf",
    block_size=16,  # Try 8 or 32 if you have specific needs
)
```

**Guidelines:**
- **block_size=8**: Less fragmentation, more overhead (use for very short sequences)
- **block_size=16**: **Recommended default** (best balance)
- **block_size=32**: More fragmentation, less overhead (use for long sequences)

### Tip 2: Set Max Batch Size

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    max_num_seqs=256,  # Maximum concurrent sequences
    max_num_batched_tokens=4096,  # Maximum tokens in a batch
)
```

**Guidelines:**
- Larger `max_num_seqs` → Higher throughput but more memory
- Smaller `max_num_seqs` → Lower latency
- Tune based on GPU memory and workload

### Tip 3: Enable Optimizations

```python
llm = LLM(
    model="meta-llama/Llama-2-13b-hf",
    # GPU optimizations
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
    # Quantization for more throughput
    quantization="awq",  # or "gptq", "fp8"
    # Tensor parallelism for large models
    tensor_parallel_size=2,  # Split across 2 GPUs
)
```

### Tip 4: Monitor Performance

```python
import time
from vllm import LLM, SamplingParams

llm = LLM(model="gpt2")
sampling_params = SamplingParams(max_tokens=100)

prompts = ["Hello"] * 100  # 100 requests

start = time.time()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.time() - start

print(f"Throughput: {len(prompts) / elapsed:.2f} req/s")
print(f"Latency: {elapsed / len(prompts):.3f} s/req")

# Count total tokens generated
total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
print(f"Tokens/s: {total_tokens / elapsed:.2f}")
```

---

## Troubleshooting

### Issue 1: Out of Memory (OOM)

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**

```python
# 1. Reduce batch size
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    max_num_seqs=64,  # Reduce from default
)

# 2. Reduce GPU memory utilization
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    gpu_memory_utilization=0.8,  # Reduce from 0.9
)

# 3. Use quantization
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",  # 4-bit quantization
)

# 4. Use tensor parallelism (if you have multiple GPUs)
llm = LLM(
    model="meta-llama/Llama-2-13b-hf",
    tensor_parallel_size=2,  # Split across 2 GPUs
)
```

### Issue 2: Slow Generation

**Symptom:** Low throughput, high latency

**Solutions:**

```python
# 1. Increase batch size (if memory allows)
llm = LLM(
    model="gpt2",
    max_num_seqs=512,  # Increase for higher throughput
)

# 2. Ensure you're batching requests
# Bad (one at a time):
for prompt in prompts:
    output = llm.generate([prompt], params)

# Good (batch):
outputs = llm.generate(prompts, params)

# 3. Check if GPU is being used
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.current_device())
```

### Issue 3: Model Not Found

**Error:**
```
OSError: meta-llama/Llama-2-7b-hf is not a local folder and is not a valid model identifier
```

**Solutions:**

```python
# 1. Login to HuggingFace (for gated models like LLaMA)
from huggingface_hub import login
login(token="your_hf_token")

# 2. Or download model first
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "meta-llama/Llama-2-7b-hf"
AutoTokenizer.from_pretrained(model_name)
AutoModelForCausalLM.from_pretrained(model_name)

# 3. Then use with vLLM
llm = LLM(model=model_name)
```

### Issue 4: Block Size Errors

**Error:**
```
ValueError: block_size must be divisible by ...
```

**Solution:**

```python
# Use standard block sizes: 8, 16, 32
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    block_size=16,  # Don't use arbitrary values
)
```

---

## Next Steps

### Learn More

1. **Read detailed concepts**: See [CONCEPTS.md](CONCEPTS.md) for mathematical details
2. **Study implementation**: See [IMPLEMENTATION.md](IMPLEMENTATION.md) for code deep dive
3. **Compare with other systems**: See [COMPARISON.md](COMPARISON.md) for benchmarks
4. **Read main tutorial**: See [README.md](README.md) for comprehensive overview

### Advanced Topics

- **Tensor Parallelism**: Split large models across GPUs
- **Quantization**: Use AWQ, GPTQ for 4-bit inference
- **Custom Kernels**: Integrate with FlashAttention
- **Deployment**: Serve with FastAPI, Docker, Kubernetes

### Resources

- **vLLM GitHub**: https://github.com/vllm-project/vllm
- **vLLM Docs**: https://docs.vllm.ai/
- **Paper**: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)
- **Blog**: https://blog.vllm.ai/

---

## Summary

PagedAttention (via vLLM) makes LLM serving:
- ✅ **6-24× faster** than traditional systems
- ✅ **~85% less memory waste**
- ✅ **Simple to use** (few lines of code)
- ✅ **Production-ready** (used by major deployments)

**Get started in 3 lines:**
```python
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-2-7b-hf")
outputs = llm.generate(["Hello!"], SamplingParams(max_tokens=50))
```

Happy serving! 🚀
