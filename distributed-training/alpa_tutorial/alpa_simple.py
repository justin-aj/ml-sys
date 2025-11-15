"""
Alpa Simple Example: Automatic Model Parallelism

This example demonstrates how Alpa automatically parallelizes a simple MLP
across multiple GPUs without any manual configuration.

Key Concepts:
1. Define model using JAX/Flax (similar to PyTorch)
2. Add @parallelize decorator  
3. Alpa automatically finds best parallelization
4. No manual sharding, pipelining, or configuration needed!

Note: Run with Python 3.10: python3.10 alpa_simple.py
"""

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.training import train_state
import optax

# Try to import Alpa (optional for learning)
try:
    from alpa import parallelize, get_last_executable
    ALPA_AVAILABLE = True
    print("✅ Alpa installed - will use automatic parallelization")
except ImportError:
    print("ℹ️  Alpa not installed - running without @parallelize")
    print("   Install with: pip install alpa")
    print("   (Code will still run and teach JAX/Flax concepts!)")
    ALPA_AVAILABLE = False
    
    # Mock parallelize decorator
    def parallelize(func):
        return func
    
    def get_last_executable(func):
        return None

print()
print("=" * 80)
print("ALPA TUTORIAL: Automatic Model Parallelism")
print("=" * 80)
print()
print("📚 What This Example Shows:")
print("   1. How to define models in JAX/Flax (similar to PyTorch)")
print("   2. How @parallelize decorator enables automatic parallelism")
print("   3. What Alpa decides automatically (no manual tuning!)")
print()
print("💡 Key Insight:")
print("   With Alpa, you write simple model code.")
print("   Alpa figures out how to parallelize across GPUs automatically!")
print()
print("=" * 80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "input_size": 512,
    "hidden_size": 2048,
    "num_layers": 8,
    "output_size": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_steps": 10,
}

# ============================================================================
# 1. DEFINE MODEL (JAX/Flax Style)
# ============================================================================

class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron
    
    This is similar to PyTorch's nn.Module, but using JAX/Flax.
    
    Key differences from PyTorch:
    - Use @nn.compact decorator
    - Layers defined in __call__ (not __init__)
    - Functional style (pass params explicitly)
    """
    
    hidden_size: int = 2048
    num_layers: int = 8
    output_size: int = 10
    
    @nn.compact
    def __call__(self, x):
        # Input layer
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.relu(x)
        
        # Output layer
        x = nn.Dense(self.output_size)(x)
        return x

print("✅ Step 1: Model Defined")
print(f"   - Input size: {CONFIG['input_size']}")
print(f"   - Hidden size: {CONFIG['hidden_size']}")
print(f"   - Num layers: {CONFIG['num_layers']}")
print(f"   - Output size: {CONFIG['output_size']}")
print()

# Calculate model size
params_per_layer = CONFIG['input_size'] * CONFIG['hidden_size']  # First layer
params_per_layer += (CONFIG['num_layers'] - 1) * CONFIG['hidden_size'] * CONFIG['hidden_size']  # Hidden
params_per_layer += CONFIG['hidden_size'] * CONFIG['output_size']  # Output
print(f"📊 Model Statistics:")
print(f"   - Parameters: ~{params_per_layer / 1e6:.1f}M")
print(f"   - Memory (FP32): ~{params_per_layer * 4 / 1e9:.2f} GB")
print()

# ============================================================================
# 2. DEFINE TRAINING FUNCTIONS WITH @parallelize
# ============================================================================

# With Alpa: Just add @parallelize decorator!

@parallelize
def train_step(state, batch, labels):
    """
    Training step - Alpa will parallelize this automatically!
    
    Alpa analyzes this function and decides:
    1. Which operations to split across GPUs (tensor parallelism)
    2. Which operations to pipeline (pipeline parallelism)
    3. How to minimize communication
    4. How to balance load
    
    You don't need to do anything - just write normal JAX code!
    """
    
    def loss_fn(params):
        logits = state.apply_fn(params, batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return jnp.mean(loss)
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

print("✅ Step 2: Training Function Decorated with @parallelize")
print()
if ALPA_AVAILABLE:
    print("   Alpa will automatically:")
    print("   1. ✅ Analyze the computation graph")
    print("   2. ✅ Generate parallelization options for each operation")
    print("   3. ✅ Estimate costs (compute + communication)")
    print("   4. ✅ Find optimal parallelization using dynamic programming")
    print("   5. ✅ Generate efficient GPU kernels")
else:
    print("   (Alpa not installed - showing what it would do)")
print()

# ============================================================================
# 3. INITIALIZE MODEL AND OPTIMIZER
# ============================================================================

# Create model
model = SimpleMLP(
    hidden_size=CONFIG['hidden_size'],
    num_layers=CONFIG['num_layers'],
    output_size=CONFIG['output_size']
)

# Initialize parameters
rng = random.PRNGKey(0)
dummy_input = jnp.ones([CONFIG['batch_size'], CONFIG['input_size']])
params = model.init(rng, dummy_input)

# Create optimizer
tx = optax.adam(CONFIG['learning_rate'])

# Create training state
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx,
)

print("✅ Step 3: Model and Optimizer Initialized")
print()

# ============================================================================
# 4. RUN TRAINING - ALPA PARALLELIZES AUTOMATICALLY!
# ============================================================================

print("🚀 Step 4: Training Started")
print()
if ALPA_AVAILABLE:
    print("   On first run, Alpa will:")
    print("   - Compile the model (may take a few minutes)")
    print("   - Find optimal parallelization")
    print("   - Generate GPU kernels")
    print()
    print("   Subsequent runs are fast!")
else:
    print("   Running without Alpa (standard JAX)")
print()

# Generate fake data
rng, data_rng = random.split(rng)
fake_batch = random.normal(data_rng, [CONFIG['batch_size'], CONFIG['input_size']])
fake_labels = random.randint(data_rng, [CONFIG['batch_size']], 0, CONFIG['output_size'])

print("Training progress:")
print("-" * 80)

for step in range(CONFIG['num_steps']):
    # Training step - Alpa handles parallelization!
    state, loss = train_step(state, fake_batch, fake_labels)
    
    if step % 2 == 0:
        print(f"Step {step:3d} | Loss: {loss:.4f}")

print("-" * 80)
print()
print("✅ Training Complete!")
print()

# ============================================================================
# 5. ANALYZE ALPA'S DECISIONS (if Alpa is installed)
# ============================================================================

if ALPA_AVAILABLE:
    print("=" * 80)
    print("ALPA'S PARALLELIZATION DECISIONS")
    print("=" * 80)
    print()
    
    try:
        # Get the compiled executable
        executable = get_last_executable(train_step)
        
        if executable:
            print("📊 What Alpa Decided:")
            print()
            print("   Alpa analyzed your model and chose optimal parallelization!")
            print("   (Run on multi-GPU system to see full parallelization plan)")
            print()
        else:
            print("   Run on multi-GPU system to see Alpa's parallelization decisions")
            print()
        
    except Exception as e:
        print(f"   Could not retrieve execution plan: {e}")
        print()

# ============================================================================
# 6. COMPARISON: MANUAL VS AUTOMATIC
# ============================================================================

print("=" * 80)
print("COMPARISON: Manual Parallelism vs Alpa")
print("=" * 80)
print()

print("❌ Manual Approach (ZeRO, Megatron, PipeDream):")
print()
print("   You must decide:")
print("   1. Which parallelism strategy? (data/pipeline/tensor)")
print("   2. How many pipeline stages? (2, 4, 8?)")
print("   3. How to split each layer? (by rows? columns? heads?)")
print("   4. How many microbatches? (2, 4, 8?)")
print("   5. How to balance load across GPUs?")
print("   6. How to minimize communication?")
print()
print("   Then implement:")
print("   - 100+ lines of parallelization code")
print("   - Custom communication patterns")
print("   - Load balancing logic")
print("   - Debugging for weeks...")
print()
print("   Total time: Weeks of expert work")
print()

print("✅ Alpa Approach:")
print()
print("   Your code:")
print("   ```python")
print("   @parallelize")
print("   def train_step(state, batch, labels):")
print("       # Just write normal training code!")
print("       loss = compute_loss(state, batch, labels)")
print("       return update(state, loss)")
print("   ```")
print()
print("   Alpa automatically:")
print("   1. ✅ Analyzes your model")
print("   2. ✅ Generates all parallelization options")
print("   3. ✅ Estimates costs for each option")
print("   4. ✅ Finds optimal plan (dynamic programming + ILP)")
print("   5. ✅ Generates efficient GPU code")
print("   6. ✅ Executes with minimal communication")
print()
print("   Total time: 5-30 minutes (first compilation)")
print("   Performance: Often matches or beats manual tuning!")
print()

# ============================================================================
# 7. WHAT YOU LEARNED
# ============================================================================

print("=" * 80)
print("SUMMARY: What You Learned")
print("=" * 80)
print()

print("✅ Key Concepts:")
print()
print("   1. JAX/Flax Basics")
print("      - Models defined with @nn.compact")
print("      - Functional style (explicit params)")
print("      - Similar to PyTorch but more functional")
print()
print("   2. Alpa's @parallelize Decorator")
print("      - Just add to your training function")
print("      - Alpa handles all parallelization automatically")
print("      - Works with any JAX/Flax model")
print()
print("   3. Automatic Parallelism")
print("      - Alpa analyzes computation graph")
print("      - Finds optimal tensor + pipeline parallelism")
print("      - No manual tuning needed!")
print()
print("   4. Comparison with Manual Methods")
print("      - Manual: Weeks of expert work")
print("      - Alpa: Minutes, near-optimal performance")
print()

print("🎯 When to Use Alpa:")
print()
print("   ✅ Large models (1B+ parameters)")
print("   ✅ Want best performance without manual tuning")
print("   ✅ Trying new architectures (Alpa adapts automatically)")
print("   ✅ Complex models (transformers, mixture-of-experts)")
print()
print("   ❌ Small models (< 100M params) - simple data parallel is fine")
print("   ❌ Must use PyTorch - Alpa is JAX-only")
print()

print("📚 Next Steps:")
print()
print("   1. Read README.md - Full tutorial with all concepts")
print("   2. Run alpa_visualize.py - Generate visual diagrams")
print("   3. Read COMPARISON.md - Compare with ZeRO and PipeDream")
print("   4. Try on your own models - Just add @parallelize!")
print()

print("=" * 80)
print("🚀 You've learned Alpa basics! See README.md for deeper dive.")
print("=" * 80)
