# Mirage Concept: Superoptimization via Equality Saturation

## 🎯 The Core Problem

**Every optimizer faces a fundamental challenge:**

Traditional optimizers apply rewrite rules **sequentially**:
```
Input: A @ B + A @ C

Step 1: Apply distributive rule
→ A @ (B + C)

Step 2: Done! (might have missed better options)
```

**The problem:** What if there's a better transformation we could have applied first? What if combining multiple rules in a different order yields something better?

**Mirage's Solution:** Apply **ALL** rules **simultaneously**, explore **ALL** equivalent programs, then pick the best!

---

## 🧮 Concrete Example: The Search Space Explosion

### Simple Expression Optimization

```python
# Input expression
x = a * b + a * c + a * d
```

**Possible equivalent forms:**
```python
# Option 1: Left-to-right distributive
x = (a * b + a * c) + a * d
x = a * (b + c) + a * d
x = a * (b + c + d)  ← Final form

# Option 2: Right-to-left distributive
x = a * b + (a * c + a * d)
x = a * b + a * (c + d)
x = a * (b + c + d)  ← Same final form

# Option 3: Middle-first
x = a * b + a * c + a * d
x = a * b + a * (c + d)
x = a * (b + c + d)  ← Same again!

# Option 4: Factor all at once
x = a * (b + c + d)  ← Direct

# ... and many more paths!
```

**Traditional optimizer (TASO):**
- Picks ONE path (e.g., left-to-right)
- Hopes it leads to optimal
- Might miss better intermediate forms

**Mirage:**
- Explores ALL paths simultaneously
- Represents all equivalent forms in e-graph
- Guaranteed to find optimal (if it exists in the rewrite rules)

---

## 📊 E-Graphs: The Magic Data Structure

### What is an E-Graph?

An **e-graph** (equality graph) is a data structure that:
- Represents multiple equivalent expressions compactly
- Shares common subexpressions
- Grows polynomially while representing exponentially many programs

### Visual Example

**Programs:**
```python
1. a * b + a * c
2. a * (b + c)
3. (b + c) * a
4. b * a + c * a
5. (a * b) + (a * c)
... millions more!
```

**E-Graph (simplified):**
```
E-Class 0: {a}
E-Class 1: {b}
E-Class 2: {c}
E-Class 3: {mul(0,1), mul(1,0)}  // a*b and b*a
E-Class 4: {mul(0,2), mul(2,0)}  // a*c and c*a
E-Class 5: {add(1,2), add(2,1)}  // b+c and c+b
E-Class 6: {mul(0,5)}            // a*(b+c)
E-Class 7: {add(3,4)}            // a*b + a*c
E-Class 8: {6, 7}                // Final e-class: both forms are equal!
```

**Key Insight:** Instead of storing millions of programs, we store equivalence classes!

---

## 🔄 Equality Saturation Algorithm

### Step-by-Step Process

#### **Phase 1: Initialization**
```python
# Start with input program
egraph = EGraph()
egraph.add(a * b + a * c)

# E-graph contains:
# - One e-class for each subexpression
```

#### **Phase 2: Rewriting**
```python
# Apply ALL rewrite rules exhaustively
rules = [
    "commutativity:  a + b → b + a",
    "associativity:  (a + b) + c → a + (b + c)",
    "distributivity: a*b + a*c → a*(b+c)",
    # ... hundreds more
]

while not saturated:
    for rule in rules:
        for match in egraph.find_matches(rule):
            egraph.add_equivalent(match.lhs, match.rhs)
```

#### **Phase 3: Saturation Check**
```python
# Saturated = no new equivalences can be added
saturated = (no rules apply anymore)
```

#### **Phase 4: Extraction**
```python
# Find cheapest program in e-graph
def extract(egraph, cost_model):
    for eclass in egraph:
        for expr in eclass.expressions:
            cost = cost_model(expr, child_costs)
        
        eclass.best = min(cost)
    
    return egraph.root.best
```

---

## 💡 Why This Works

### The Magic: Compact Representation

**Naively storing all programs:**
```
2 equivalents per subexpression
→ 2^n total programs for n operations
→ Exponential explosion! 💥
```

**E-graph representation:**
```
Add new e-class for each unique equivalence
→ Polynomial size (manageable!)
→ But represents exponentially many programs ✨
```

### Example: Matrix Chain

```python
# Input: (A @ B) @ (C @ D)
# How many parenthesizations?

For 4 matrices: 5 ways
For 5 matrices: 14 ways
For 10 matrices: 16,796 ways
For 20 matrices: 1,767,263,190 ways!

# E-graph size: O(n^2) nodes
# Programs represented: Exponential!
```

---

## 🎯 Real Example: Transformer Optimization

### Input Computation

```python
# Standard multi-head attention
def attention(Q, K, V):
    # Q, K, V: [batch, seq, heads, d_head]
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / math.sqrt(d_head)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output
```

### Mirage E-Graph Exploration

#### **E-Class 1: Q @ K.T**
```python
{
    matmul(Q, transpose(K)),
    transpose(matmul(transpose(K), transpose(Q))),
    # ... transpose identities
}
```

#### **E-Class 2: scores / sqrt(d)**
```python
{
    div(e1, sqrt(d)),
    mul(e1, 1/sqrt(d)),
    mul(e1, rsqrt(d)),        # Reciprocal sqrt (faster!)
    # ... algebraic variants
}
```

#### **E-Class 3: softmax(e2)**
```python
{
    softmax(e2),
    div(exp(e2), sum(exp(e2))),
    div(exp(sub(e2, max(e2))), sum(exp(sub(e2, max(e2))))),  # Numerically stable
    fused_softmax(e2),        # Kernel fusion!
    # ... decompositions
}
```

#### **E-Class 4: attn @ V**
```python
{
    matmul(e3, V),
    fused_softmax_matmul(e2, V),  # Flash Attention pattern!
    # ... fused variants
}
```

### What Mirage Discovers

**Traditional optimization (TASO):**
```python
# Applies known fusion rules
output = fused_attention(Q, K, V)  # Good, but predefined
```

**Mirage superoptimization:**
```python
# Discovers optimal combination:
scores = Q @ (K.T * rsqrt(d))      # Fuse scaling into K
attn = fused_stable_softmax(scores) # Numerically stable + fused
output = blockwise_matmul(attn, V)  # Flash Attention pattern

# AND verifies this is optimal for target hardware!
```

**Speedup:** 2.5x vs naive, 1.3x vs TASO (found non-obvious optimization!)

---

## 📊 Cost Models

### How Mirage Picks "Best"

**Cost function example:**
```python
def cost(expr, hardware="v100"):
    if is_matmul(expr):
        M, K, N = get_dims(expr)
        flops = 2 * M * K * N
        memory = M * K + K * N + M * N
        
        if hardware == "v100":
            time = flops / (15.7e12) + memory / (900e9)  # TFLOPS + bandwidth
        
        return time
    
    elif is_fused(expr):
        # Fusion reduces memory traffic
        return cost_unfused(expr) * 0.6
    
    elif is_elementwise(expr):
        return memory_size(expr) / bandwidth(hardware)
    
    # ... hardware-specific costs
```

**Key:** Different hardware → different optimal programs!

---

## 🔬 Advanced: Rewrite Rules

### Types of Rules in Mirage

#### **1. Algebraic Identities**
```python
# Commutativity
a + b → b + a
a * b → b * a

# Associativity
(a + b) + c → a + (b + c)
(a * b) * c → a * (b * c)

# Distributivity
a * b + a * c → a * (b + c)
```

#### **2. Matrix Identities**
```python
# Transpose
(A @ B).T → B.T @ A.T
(A.T).T → A

# Inverse
(A @ B)^-1 → B^-1 @ A^-1
```

#### **3. Operator Fusion**
```python
# Fuse elementwise ops
exp(a) + exp(b) → fused_exp_add(a, b)

# Fuse reduction
sum(exp(a)) → fused_sum_exp(a)

# Fuse matmul + bias
(A @ B) + c → fused_gemm(A, B, c)
```

#### **4. Hardware-Specific**
```python
# Prefer rsqrt on NVIDIA GPUs (single instruction)
1 / sqrt(x) → rsqrt(x)

# Prefer fused multiply-add
a * b + c → fma(a, b, c)
```

### Rule Application Example

```python
# Input: softmax(Q @ K.T / sqrt(d))

# Mirage applies hundreds of rules:
Rule 1: (Q @ K.T) / d → Q @ (K.T / d)       # Move division
Rule 2: 1/sqrt(d) → rsqrt(d)                # Use hardware rsqrt
Rule 3: softmax(x) → div(exp(x), sum(exp))  # Decompose
Rule 4: exp(x - max(x))                     # Numerical stability
Rule 5: fused_softmax(x)                    # Kernel fusion
# ... and explores all combinations!

# Result: Optimal combination for V100 vs A100 vs CPU may differ!
```

---

## 🎓 Key Insights

### 1. Exhaustive Search is Possible

**Counterintuitive but true:**
```
Naive: Can't enumerate 10^15 programs
E-graph: Represents them compactly in polynomial space
Result: Exhaustive search becomes tractable! ✨
```

### 2. Local vs Global Optimality

**TASO (local):**
```
Step 1: a*b + a*c → a*(b+c)  ← Good!
Step 2: Apply next rule on a*(b+c)
Result: Local minimum (might miss global optimum)
```

**Mirage (global):**
```
Explore: a*b + a*c, a*(b+c), (b+c)*a, b*a + c*a, ...
Pick: Globally optimal from all options
Result: Guaranteed best (within rule set and cost model)
```

### 3. Correctness by Construction

**All transformations are mathematically proven equivalences:**
```
Traditional: Trust the optimizer implementer
Mirage: Each rewrite rule is a proven equivalence
Result: Output is GUARANTEED correct! ✅
```

---

## 📈 Performance Characteristics

### Time Complexity

```
Input size: n operations

TASO:    O(n) to O(n^2) (apply rules sequentially)
Mirage:  O(2^n) worst case, O(n^3) typical (e-graph growth)

Practical:
- TASO: Seconds
- Mirage: Minutes (for complex graphs)
```

### Space Complexity

```
Programs: Exponential (2^n)
E-graph: Polynomial (O(n^2) to O(n^3))

Result: Mirage is slower but explores MUCH larger space
```

### Optimality

```
TASO:   Local optimum (greedy search)
Ansor:  Near-optimal (ML-guided sampling)
Mirage: Global optimum (exhaustive enumeration)
        within rewrite rules and cost model
```

---

## 🏆 When Mirage Wins Big

### Best Use Cases

1. **Complex matrix chains**
   ```python
   A @ B @ C @ D @ E
   # Mirage finds optimal parenthesization
   # Can be 100x faster than naive!
   ```

2. **Custom model architectures**
   ```python
   # Novel attention mechanisms
   # Mirage discovers non-obvious fusions
   # 1.5-3x over TASO
   ```

3. **Research/exploration**
   ```python
   # "What's the best possible?"
   # Mirage provides the answer
   ```

### Where TASO/Others Win

1. **Fast iteration** (TASO: seconds vs Mirage: minutes)
2. **Simple graphs** (overhead not worth it)
3. **Production** (TASO more mature)

---

## 🔮 The Future

### Integration with Other Tools

**Mirage + TASO:**
- Use Mirage to discover novel patterns
- Add them as rules to TASO
- Best of both worlds!

**Mirage + Ansor:**
- Mirage optimizes graph
- Ansor optimizes schedules
- Combined optimization

### Research Directions

1. **Faster equality saturation**
   - Incremental e-graph updates
   - Parallelized rule application
   - Make it production-ready!

2. **Better cost models**
   - ML-learned costs (like Ansor)
   - Runtime profiling integration
   - Multi-objective optimization

3. **Broader scope**
   - Include data layout optimization
   - Consider memory hierarchy
   - Full stack optimization

---

## 📚 Summary: The Big Picture

```
┌──────────────────────────────────────────────────────────┐
│ Optimization Approach Spectrum                           │
└──────────────────────────────────────────────────────────┘

Heuristic           Sampling          Exhaustive
(Fast)              (Balanced)        (Optimal)
   │                    │                  │
   ▼                    ▼                  ▼
TASO               Ansor              Mirage
TorchScript                          
   │                    │                  │
   ├─ Predefined rules  ├─ ML-guided      ├─ Equality saturation
   ├─ Local optimum     ├─ Near-optimal   ├─ Global optimum
   ├─ Seconds           ├─ Hours          ├─ Minutes
   └─ Production ✅     └─ Research/Prod  └─ Research 🔬

Trade-off: Speed ←→ Optimality
```

**The Future:** Techniques from Mirage will be integrated into production compilers, bringing exhaustive search to real-world ML optimization!

---

*"If the optimal program exists within your rewrite rules, Mirage will find it."* — The Equality Saturation Guarantee
