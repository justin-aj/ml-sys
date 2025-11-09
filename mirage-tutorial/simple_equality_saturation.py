"""
Mirage Tutorial: Equality Saturation for Matrix Chain Optimization
===================================================================

This tutorial demonstrates the core concept of Mirage (equality saturation)
using a simple matrix chain multiplication example.

Unlike TASO which uses greedy search, equality saturation explores ALL
mathematically equivalent expressions simultaneously using an E-graph.

Author: Mirage Tutorial
License: MIT
"""

import torch
import time
from typing import List, Tuple


class Matrix:
    """Represents a matrix with a name and shape."""
    def __init__(self, name: str, shape: Tuple[int, int]):
        self.name = name
        self.shape = shape
    
    def __repr__(self):
        return self.name
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Matrix):
            return self.name == other.name
        return False


class MatMul:
    """Represents a matrix multiplication: left @ right."""
    def __init__(self, left, right):
        self.left = left
        self.right = right
        
        # Calculate shape immediately (no recursion!)
        left_shape = left.shape if isinstance(left, Matrix) else left.result_shape
        right_shape = right.shape if isinstance(right, Matrix) else right.result_shape
        
        # Result is (left_rows, right_cols)
        self.result_shape = (left_shape[0], right_shape[1])
        
        # Calculate FLOPs for this single operation
        m, k = left_shape
        k2, n = right_shape
        assert k == k2, f"Shape mismatch: {left_shape} @ {right_shape}"
        self.flops = m * k * n
    
    def __repr__(self):
        return f"({self.left} @ {self.right})"
    
    def __hash__(self):
        return hash((self.left, self.right))
    
    def __eq__(self, other):
        if isinstance(other, MatMul):
            return self.left == other.left and self.right == other.right
        return False
    
    def total_cost(self):
        """Calculate total FLOPs including children (simple recursion)."""
        left_cost = self.left.total_cost() if isinstance(self.left, MatMul) else 0
        right_cost = self.right.total_cost() if isinstance(self.right, MatMul) else 0
        return left_cost + right_cost + self.flops


class EGraph:
    """
    E-Graph: Equality Graph
    
    A simple data structure that stores equivalent expressions.
    We just use a set to track all unique expressions we've seen.
    """
    
    def __init__(self):
        self.expressions = set()  # All unique expressions
    
    def add(self, expr):
        """Add an expression to the e-graph."""
        self.expressions.add(expr)
    
    def apply_associativity(self):
        """
        Apply associativity rewrite: (A @ B) @ C = A @ (B @ C)
        
        This is the key optimization for matrix chain multiplication!
        """
        new_exprs = []
        
        for expr in list(self.expressions):
            if isinstance(expr, MatMul):
                # Try left-associative to right-associative
                if isinstance(expr.left, MatMul):
                    # (A @ B) @ C  =>  A @ (B @ C)
                    A = expr.left.left
                    B = expr.left.right
                    C = expr.right
                    new_expr = MatMul(A, MatMul(B, C))
                    if new_expr not in self.expressions:
                        new_exprs.append(new_expr)
                
                # Try right-associative to left-associative
                if isinstance(expr.right, MatMul):
                    # A @ (B @ C)  =>  (A @ B) @ C
                    A = expr.left
                    B = expr.right.left
                    C = expr.right.right
                    new_expr = MatMul(MatMul(A, B), C)
                    if new_expr not in self.expressions:
                        new_exprs.append(new_expr)
        
        # Add all new expressions
        for expr in new_exprs:
            self.expressions.add(expr)
        
        return len(new_exprs) > 0  # Return True if we added new expressions
    
    def saturate(self):
        """
        Equality Saturation: Apply rewrite rules until no new expressions are found.
        
        This is the core of Mirage's approach!
        """
        print("\n🔄 Starting Equality Saturation...")
        iteration = 0
        
        while True:
            iteration += 1
            initial_size = len(self.expressions)
            
            # Apply rewrite rules
            changed = self.apply_associativity()
            
            final_size = len(self.expressions)
            print(f"   Iteration {iteration}: {initial_size} → {final_size} expressions")
            
            if not changed:
                print(f"✅ Saturated after {iteration} iterations!")
                print(f"   E-graph contains {final_size} equivalent expressions")
                break
    
    def extract_best(self):
        """
        Extract the cheapest expression from the e-graph.
        
        This is where we use our cost model to pick the best option!
        """
        best_expr = None
        best_cost = float('inf')
        
        for expr in self.expressions:
            if isinstance(expr, MatMul):
                cost = expr.total_cost()
            else:
                cost = 0
            
            if cost < best_cost:
                best_cost = cost
                best_expr = expr
        
        return best_expr, int(best_cost)


def greedy_optimization(matrices: List[Matrix]):
    """
    TASO-style greedy optimization.
    
    Makes local decisions without exploring all possibilities.
    """
    print("\n🎯 Greedy Optimization (TASO-style)...")
    
    # Start with left-associative: ((A @ B) @ C) @ D
    expr = matrices[0]
    for i in range(1, len(matrices)):
        expr = MatMul(expr, matrices[i])
    
    cost = expr.total_cost()
    print(f"   Result: {expr}")
    print(f"   Cost: {cost:,} FLOPs")
    
    return expr, cost


def equality_saturation_optimization(matrices: List[Matrix]):
    """
    Mirage-style equality saturation.
    
    Explores ALL equivalent expressions, then picks the best!
    """
    print("\n✨ Equality Saturation (Mirage-style)...")
    
    # Create initial expression (left-associative)
    expr = matrices[0]
    for i in range(1, len(matrices)):
        expr = MatMul(expr, matrices[i])
    
    print(f"   Initial: {expr}")
    
    # Create e-graph and add initial expression
    egraph = EGraph()
    egraph.add(expr)
    
    # Saturate: explore all equivalent expressions
    egraph.saturate()
    
    # Extract the best one
    best_expr, best_cost = egraph.extract_best()
    
    print(f"\n   Best Found: {best_expr}")
    print(f"   Cost: {best_cost:,} FLOPs")
    
    return best_expr, best_cost


def verify_correctness(matrices: List[Tuple[str, torch.Tensor]], expr):
    """Verify that an expression computes the correct result."""
    matrix_map = {name: tensor for name, tensor in matrices}
    
    def evaluate(node):
        if isinstance(node, Matrix):
            return matrix_map[node.name]
        elif isinstance(node, MatMul):
            left = evaluate(node.left)
            right = evaluate(node.right)
            return torch.matmul(left, right)
    
    return evaluate(expr)


def benchmark_execution(matrices: List[Tuple[str, torch.Tensor]], expr, device='cpu'):
    """Benchmark the actual execution time of an expression."""
    matrix_map = {name: tensor.to(device) for name, tensor in matrices}
    
    def evaluate(node):
        if isinstance(node, Matrix):
            return matrix_map[node.name]
        elif isinstance(node, MatMul):
            left = evaluate(node.left)
            right = evaluate(node.right)
            return torch.matmul(left, right)
    
    # Warmup
    for _ in range(3):
        result = evaluate(expr)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    num_runs = 10
    start = time.time()
    for _ in range(num_runs):
        result = evaluate(expr)
        if device == 'cuda':
            torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / num_runs * 1000  # Convert to ms
    return avg_time


def main():
    print("=" * 70)
    print("Mirage Tutorial: Equality Saturation for Matrix Chain Optimization")
    print("=" * 70)
    
    # Define a matrix chain: A @ B @ C @ D
    # Shapes chosen to have very different optimal parenthesizations
    A = Matrix("A", (100, 5))
    B = Matrix("B", (5, 50))
    C = Matrix("C", (50, 10))
    D = Matrix("D", (10, 20))
    
    matrices = [A, B, C, D]
    
    print("\n📊 Matrix Chain: A @ B @ C @ D")
    print(f"   A: {A.shape}")
    print(f"   B: {B.shape}")
    print(f"   C: {C.shape}")
    print(f"   D: {D.shape}")
    
    # Create actual tensors for verification
    torch.manual_seed(42)
    A_tensor = torch.randn(A.shape)
    B_tensor = torch.randn(B.shape)
    C_tensor = torch.randn(C.shape)
    D_tensor = torch.randn(D.shape)
    
    tensor_matrices = [
        ("A", A_tensor),
        ("B", B_tensor),
        ("C", C_tensor),
        ("D", D_tensor),
    ]
    
    # Method 1: Greedy (TASO-style)
    greedy_expr, greedy_cost = greedy_optimization(matrices)
    
    # Method 2: Equality Saturation (Mirage-style)
    mirage_expr, mirage_cost = equality_saturation_optimization(matrices)
    
    # Verify correctness
    print("\n🔍 Verifying Correctness...")
    greedy_result = verify_correctness(tensor_matrices, greedy_expr)
    mirage_result = verify_correctness(tensor_matrices, mirage_expr)
    
    max_diff = torch.max(torch.abs(greedy_result - mirage_result)).item()
    print(f"   Max difference: {max_diff:.2e}")
    
    if max_diff < 1e-4:
        print("   ✅ Results match! (within floating-point tolerance)")
    else:
        print("   ❌ Results differ!")
    
    # Calculate speedup
    print("\n📈 Performance Comparison:")
    print(f"   Greedy (TASO):  {greedy_cost:,} FLOPs")
    print(f"   Mirage (E-sat): {mirage_cost:,} FLOPs")
    
    if greedy_cost > mirage_cost:
        speedup = greedy_cost / mirage_cost
        print(f"   🚀 Mirage is {speedup:.2f}x better! (Fewer FLOPs)")
    elif greedy_cost < mirage_cost:
        speedup = mirage_cost / greedy_cost
        print(f"   ⚠️  Greedy is {speedup:.2f}x better (This example favors greedy)")
    else:
        print(f"   ⚖️  Both methods found the same solution!")
    
    # Benchmark actual execution time
    print("\n⏱️  Actual Execution Time (CPU):")
    greedy_time = benchmark_execution(tensor_matrices, greedy_expr, 'cpu')
    mirage_time = benchmark_execution(tensor_matrices, mirage_expr, 'cpu')
    
    print(f"   Greedy: {greedy_time:.3f} ms")
    print(f"   Mirage: {mirage_time:.3f} ms")
    
    if greedy_time > mirage_time:
        time_speedup = greedy_time / mirage_time
        print(f"   🚀 Mirage is {time_speedup:.2f}x faster!")
    elif greedy_time < mirage_time:
        time_speedup = mirage_time / greedy_time
        print(f"   ⚠️  Greedy is {time_speedup:.2f}x faster!")
    else:
        print(f"   ⚖️  Same performance!")
    
    # Show all possible parenthesizations explored
    print("\n🔍 All Equivalent Expressions Found:")
    egraph2 = EGraph()
    initial_expr = matrices[0]
    for i in range(1, len(matrices)):
        initial_expr = MatMul(initial_expr, matrices[i])
    egraph2.add(initial_expr)
    egraph2.saturate()
    
    # Sort by cost
    all_exprs = [(expr, expr.total_cost()) for expr in egraph2.expressions if isinstance(expr, MatMul)]
    all_exprs.sort(key=lambda x: x[1])
    
    for i, (expr, cost) in enumerate(all_exprs, 1):
        marker = "✨ BEST" if cost == mirage_cost else ""
        print(f"   {i}. {expr}")
        print(f"      Cost: {cost:,} FLOPs {marker}")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("=" * 70)
    print("1. Greedy (TASO):   Picks first good option (fast, local optimum)")
    print("2. Mirage (E-sat):  Explores ALL options (slower, global optimum)")
    print("3. E-graphs:        Store exponentially many expressions compactly")
    print("4. Equality Sat:    Apply all rewrites, then extract best")
    print("\n💡 For production: Use TASO (fast, good enough)")
    print("💡 For research:   Use Mirage (slower, optimal)")
    print("=" * 70)


if __name__ == "__main__":
    main()
