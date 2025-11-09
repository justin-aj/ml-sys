"""
Mirage Concepts with PyTorch FX: Graph Rewriting & Cost-Based Selection
========================================================================

This example demonstrates Mirage's core ideas using PyTorch's built-in FX framework:
1. Graph representation (like e-graphs)
2. Rewrite rules (algebraic transformations)
3. Cost-based selection (choose optimal variant)

Unlike our matrix chain example, this uses REAL PyTorch to show how these
concepts apply to production ML frameworks.

Requirements:
    pip install torch

Author: Mirage Tutorial
License: MIT
"""

import torch
import torch.fx as fx
import time
import operator


# === Step 1: Define a simple model ===
class SimpleModel(torch.nn.Module):
    """A simple model with operations that can be rewritten."""
    def forward(self, x, y):
        # Intentionally has redundant ops that can be rewritten
        # (x + y) * 2.0  could also be  (x + y) + (x + y)
        return (x + y) * 2.0


# === Step 2: Define rewrite rules (like Mirage's equality saturation) ===

def rule_mul_by_2_to_add(graph_module):
    """
    Rewrite rule: (x * 2) → (x + x)
    
    This is mathematically equivalent but may have different performance!
    - Multiplication: 1 multiply operation
    - Addition: 1 addition operation
    
    On some hardware, addition is faster than multiplication.
    """
    modified = False
    for node in graph_module.graph.nodes:
        if node.op == 'call_function' and node.target in [operator.mul, torch.mul]:
            args = node.args
            # Check if multiplying by 2.0
            if len(args) >= 2 and (args[1] == 2.0 or args[0] == 2.0):
                # Determine which arg is the tensor
                tensor_arg = args[0] if args[1] == 2.0 else args[1]
                
                with graph_module.graph.inserting_after(node):
                    # Replace (x * 2) with (x + x)
                    new_node = graph_module.graph.call_function(
                        operator.add, (tensor_arg, tensor_arg)
                    )
                
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                modified = True
                break  # Only do one transformation at a time
    
    if modified:
        graph_module.recompile()
    return graph_module


def rule_add_to_mul_by_2(graph_module):
    """
    Reverse rewrite rule: (x + x) → (x * 2)
    
    This shows that rewrites can go both ways!
    Mirage explores ALL mathematically equivalent forms.
    """
    modified = False
    for node in graph_module.graph.nodes:
        if node.op == 'call_function' and node.target in [operator.add, torch.add]:
            args = node.args
            # Check if adding a value to itself
            if len(args) >= 2 and args[0] == args[1]:
                with graph_module.graph.inserting_after(node):
                    # Replace (x + x) with (x * 2)
                    new_node = graph_module.graph.call_function(
                        operator.mul, (args[0], 2.0)
                    )
                
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                modified = True
                break  # Only do one transformation at a time
    
    if modified:
        graph_module.recompile()
    return graph_module


def rule_distributive_law(graph_module):
    """
    Rewrite rule: (a + b) * c → (a * c) + (b * c)
    
    Distributive property - may enable further optimizations!
    """
    modified = False
    for node in graph_module.graph.nodes:
        if node.op == 'call_function' and node.target in [operator.mul, torch.mul]:
            args = node.args
            # Check if first arg is an addition
            if len(args) >= 2 and hasattr(args[0], 'op') and args[0].op == 'call_function' and args[0].target in [operator.add, torch.add]:
                add_node = args[0]
                multiplier = args[1]
                a, b = add_node.args[:2]
                
                with graph_module.graph.inserting_before(node):
                    # Create (a * c) and (b * c) BEFORE the current node
                    left = graph_module.graph.call_function(operator.mul, (a, multiplier))
                    right = graph_module.graph.call_function(operator.mul, (b, multiplier))
                    # Then create the addition
                    new_node = graph_module.graph.call_function(operator.add, (left, right))
                
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                modified = True
                break  # Only do one transformation at a time
    
    if modified:
        graph_module.recompile()
    return graph_module


# === Step 3: Cost model (benchmarking as a proxy for learned cost) ===

def estimate_cost(graph_module, input_shape=(1024,), warmup=100, iterations=1000):
    """
    Estimate runtime cost by benchmarking.
    
    In real Mirage:
    - Would use a learned cost model (neural network predicting runtime)
    - Would consider FLOPs, memory bandwidth, hardware-specific features
    
    Here we just benchmark actual execution time.
    """
    x = torch.randn(*input_shape)
    y = torch.randn(*input_shape)
    
    # Warmup
    for _ in range(warmup):
        _ = graph_module(x, y)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(iterations):
        _ = graph_module(x, y)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    return (time.time() - t0) / iterations * 1000  # Return ms per iteration


# === Step 4: Equality saturation (generate all equivalent graphs) ===

def equality_saturation(model, rewrite_rules, max_iterations=3):
    """
    Mirage-style equality saturation:
    1. Start with original graph
    2. Apply all rewrite rules
    3. Keep all generated variants (like e-graph)
    4. Repeat until no new variants found (saturated)
    
    Returns: Dictionary of {description: graph_module}
    """
    print("\n🔄 Starting Equality Saturation...")
    
    # Track all unique graphs
    graphs = {"original": fx.symbolic_trace(model)}
    graph_strings = {str(graphs["original"].graph)}
    
    for iteration in range(max_iterations):
        initial_count = len(graphs)
        print(f"\n   Iteration {iteration + 1}: {initial_count} graphs")
        
        # Apply each rewrite rule to each existing graph
        for name, gm in list(graphs.items()):
            for rule_idx, rule in enumerate(rewrite_rules):
                try:
                    # Deep copy the graph module
                    import copy
                    new_gm = copy.deepcopy(gm)
                    
                    # Apply the rewrite rule
                    new_gm = rule(new_gm)
                    
                    # Check if this produced a new unique graph
                    graph_str = str(new_gm.graph)
                    if graph_str not in graph_strings:
                        variant_name = f"{name}_r{rule_idx+1}"
                        graphs[variant_name] = new_gm
                        graph_strings.add(graph_str)
                        print(f"      ✨ Found new variant: {variant_name}")
                except Exception as e:
                    # Rule might not apply or error occurred
                    pass
        
        final_count = len(graphs)
        if final_count == initial_count:
            print(f"\n✅ Saturated after {iteration + 1} iterations!")
            break
    
    print(f"   E-graph contains {len(graphs)} equivalent forms")
    return graphs


# === Main execution ===

def main():
    print("=" * 80)
    print("Mirage Concepts with PyTorch FX: Graph Rewriting & Cost-Based Selection")
    print("=" * 80)
    
    # Create model
    model = SimpleModel()
    
    # Trace to FX graph
    gm = fx.symbolic_trace(model)
    print("\n📊 Original Graph:")
    print(gm.graph)
    print("\nOriginal Code:")
    print(gm.code)
    
    # Define rewrite rules
    rewrite_rules = [
        rule_mul_by_2_to_add,
        rule_add_to_mul_by_2,
        rule_distributive_law,
    ]
    
    # Generate candidates using equality saturation
    print("\n" + "=" * 80)
    candidates = equality_saturation(model, rewrite_rules)
    
    # Show all variants
    print("\n" + "=" * 80)
    print("All Generated Variants:")
    print("=" * 80)
    for name, gm in candidates.items():
        print(f"\n{name}:")
        print(gm.code)
    
    # Benchmark each variant
    print("\n" + "=" * 80)
    print("⏱️  Benchmarking All Variants:")
    print("=" * 80)
    
    results = {}
    for name, gmod in candidates.items():
        try:
            cost = estimate_cost(gmod, input_shape=(1024,))
            results[name] = cost
            print(f"   {name:<20} {cost:.6f} ms/iteration")
        except Exception as e:
            print(f"   {name:<20} ERROR: {e}")
    
    # Pick best
    if results:
        best_name, best_cost = min(results.items(), key=lambda kv: kv[1])
        worst_name, worst_cost = max(results.items(), key=lambda kv: kv[1])
        
        print("\n" + "=" * 80)
        print("📈 Results:")
        print("=" * 80)
        print(f"✨ Best variant:  {best_name} ({best_cost:.6f} ms)")
        print(f"⚠️  Worst variant: {worst_name} ({worst_cost:.6f} ms)")
        
        if best_cost < worst_cost:
            speedup = worst_cost / best_cost
            print(f"🚀 Speedup: {speedup:.2f}x")
        
        print("\n📝 Optimal Graph IR:")
        print(candidates[best_name].graph)
        
        print("\n💡 Optimal Python Code:")
        print(candidates[best_name].code)
    
    # Verify correctness
    print("\n" + "=" * 80)
    print("🔍 Verifying Correctness:")
    print("=" * 80)
    
    x = torch.randn(100)
    y = torch.randn(100)
    original_result = candidates["original"](x, y)
    
    all_correct = True
    for name, gmod in candidates.items():
        result = gmod(x, y)
        max_diff = torch.max(torch.abs(result - original_result)).item()
        status = "✅" if max_diff < 1e-5 else "❌"
        print(f"   {status} {name:<20} max_diff={max_diff:.2e}")
        if max_diff >= 1e-5:
            all_correct = False
    
    if all_correct:
        print("\n✅ All variants produce identical results!")
    
    # Key insights
    print("\n" + "=" * 80)
    print("Key Insights:")
    print("=" * 80)
    print("1. PyTorch FX graphs ≈ E-graphs (compact representation)")
    print("2. Rewrite rules = Algebraic equivalences (x*2 ↔ x+x)")
    print("3. Cost model = Benchmarking (could be learned neural network)")
    print("4. Equality saturation = Apply all rules, pick best")
    print("\n💡 This is exactly how Mirage works, but for complex tensor programs!")
    print("=" * 80)


if __name__ == "__main__":
    main()
