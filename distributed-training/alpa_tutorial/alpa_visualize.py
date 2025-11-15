"""
Alpa Visualization: See How Alpa Makes Parallelization Decisions

This script visualizes:
1. Different parallelization strategies
2. How Alpa chooses between them
3. Communication patterns for each strategy
4. Performance comparison
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Create output directory if needed
import os
os.makedirs('.', exist_ok=True)

print("=" * 80)
print("ALPA VISUALIZATION: Parallelization Strategies")
print("=" * 80)
print()

# ============================================================================
# 1. DATA PARALLELISM
# ============================================================================

def visualize_data_parallel():
    """Show data parallelism: same model on each GPU, different data"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # 4 GPUs with full model
    gpu_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for i in range(4):
        # Each GPU has full model
        rect = patches.Rectangle((i*2.5, 0), 2, 4, 
                                 linewidth=2, edgecolor='black', 
                                 facecolor=gpu_colors[i], alpha=0.6)
        ax.add_patch(rect)
        
        # Label
        ax.text(i*2.5 + 1, 2, f'GPU {i}\n[Full Model]\nBatch {i}', 
               ha='center', va='center', fontsize=12, weight='bold')
    
    # Arrow showing gradient sync
    ax.annotate('', xy=(8.5, 4.5), xytext=(1.5, 4.5),
                arrowprops=dict(arrowstyle='<->', lw=2, color='red'))
    ax.text(5, 5, 'Gradient Sync\n(AllReduce)', ha='center', fontsize=10, color='red')
    
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 6)
    ax.axis('off')
    ax.set_title('Data Parallelism: Same Model, Different Data', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig('data_parallel.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: data_parallel.png")
    plt.close()

# ============================================================================
# 2. PIPELINE PARALLELISM  
# ============================================================================

def visualize_pipeline_parallel():
    """Show pipeline parallelism: different layers on each GPU"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # 4 GPUs with different layers
    gpu_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    layer_labels = ['Layers 1-12', 'Layers 13-24', 'Layers 25-36', 'Layers 37-48']
    
    for i in range(4):
        rect = patches.Rectangle((i*2.5, 0), 2, 4,
                                 linewidth=2, edgecolor='black',
                                 facecolor=gpu_colors[i], alpha=0.6)
        ax.add_patch(rect)
        
        ax.text(i*2.5 + 1, 2, f'GPU {i}\n{layer_labels[i]}',
               ha='center', va='center', fontsize=11, weight='bold')
    
    # Arrows showing data flow
    for i in range(3):
        ax.annotate('', xy=((i+1)*2.5 - 0.2, 2), xytext=(i*2.5 + 2.2, 2),
                   arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    
    ax.text(5, 4.5, 'Data flows through pipeline →', ha='center', fontsize=12, color='blue')
    
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 5.5)
    ax.axis('off')
    ax.set_title('Pipeline Parallelism: Different Layers on Each GPU', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig('pipeline_parallel.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: pipeline_parallel.png")
    plt.close()

# ============================================================================
# 3. TENSOR PARALLELISM
# ============================================================================

def visualize_tensor_parallel():
    """Show tensor parallelism: split single operation across GPUs"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    # Show matrix multiply split
    ax.text(5, 6.5, 'Original Operation: Y = X @ W', 
           ha='center', fontsize=14, weight='bold')
    ax.text(5, 6, 'X: [batch, 512]   W: [512, 2048]', 
           ha='center', fontsize=11)
    
    # Tensor parallel split
    gpu_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for i in range(4):
        y_pos = 4 - i*1
        
        # Show W split
        rect = patches.Rectangle((0.5, y_pos-0.3), 1.5, 0.6,
                                 linewidth=2, edgecolor='black',
                                 facecolor=gpu_colors[i], alpha=0.6)
        ax.add_patch(rect)
        ax.text(1.25, y_pos, f'W{i}\n[512, 512]', ha='center', va='center', fontsize=9)
        
        # Show computation
        ax.text(3, y_pos, f'GPU {i}: Y{i} = X @ W{i}', ha='center', va='center', 
               fontsize=10, weight='bold')
        
        # Show result
        rect_result = patches.Rectangle((5.5, y_pos-0.3), 1.5, 0.6,
                                       linewidth=2, edgecolor='black',
                                       facecolor=gpu_colors[i], alpha=0.6)
        ax.add_patch(rect_result)
        ax.text(6.25, y_pos, f'Y{i}\n[batch, 512]', ha='center', va='center', fontsize=9)
    
    # Concat operation
    ax.annotate('', xy=(8, 3), xytext=(7.2, 4),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(8, 3), xytext=(7.2, 3),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(8, 3), xytext=(7.2, 2),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(8, 3), xytext=(7.2, 1),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    rect_final = patches.Rectangle((8.5, 2.5), 1.5, 1,
                                   linewidth=3, edgecolor='green',
                                   facecolor='lightgreen', alpha=0.4)
    ax.add_patch(rect_final)
    ax.text(9.25, 3, 'Concat\nY\n[batch, 2048]', ha='center', va='center', 
           fontsize=10, weight='bold')
    
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Tensor Parallelism: Split Single Operation Across GPUs', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig('tensor_parallel.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: tensor_parallel.png")
    plt.close()

# ============================================================================
# 4. ALPA'S TWO-LEVEL HIERARCHY
# ============================================================================

def visualize_alpa_hierarchy():
    """Show how Alpa combines inter-op and intra-op parallelism"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Level 1: Intra-Operator (Tensor Parallelism)
    ax1.text(7, 4.5, 'Level 1: Intra-Operator Parallelism (Inside Each Operation)',
            ha='center', fontsize=13, weight='bold')
    
    operations = ['Dense 1', 'Dense 2', 'Dense 3', 'Dense 4']
    colors_intra = ['#FF6B6B', '#4ECDC4']
    
    for i, op in enumerate(operations):
        x_base = i * 3.5
        
        # Show operation split across 2 GPUs
        for j in range(2):
            rect = patches.Rectangle((x_base + j*1.5, 2), 1.3, 1.5,
                                     linewidth=2, edgecolor='black',
                                     facecolor=colors_intra[j], alpha=0.6)
            ax1.add_patch(rect)
            ax1.text(x_base + j*1.5 + 0.65, 2.75, f'{op}\nGPU{j}',
                    ha='center', va='center', fontsize=9)
    
    ax1.set_xlim(-0.5, 14.5)
    ax1.set_ylim(0, 5)
    ax1.axis('off')
    
    # Level 2: Inter-Operator (Pipeline Parallelism)
    ax2.text(7, 4.5, 'Level 2: Inter-Operator Parallelism (Between Operations)',
            ha='center', fontsize=13, weight='bold')
    
    stages = ['Stage 0\n(Dense 1-2)', 'Stage 1\n(Dense 3-4)']
    colors_inter = ['#FFB6B6', '#8EDEE4']
    
    for i, stage in enumerate(stages):
        rect = patches.Rectangle((i*7, 1), 6, 2.5,
                                 linewidth=3, edgecolor='black',
                                 facecolor=colors_inter[i], alpha=0.4)
        ax2.add_patch(rect)
        ax2.text(i*7 + 3, 2.25, stage, ha='center', va='center',
                fontsize=11, weight='bold')
        
        # Show 2 GPUs per stage
        for j in range(2):
            mini_rect = patches.Rectangle((i*7 + 0.5 + j*2.5, 1.5), 2, 1.5,
                                         linewidth=1, edgecolor='gray',
                                         facecolor='white', alpha=0.3)
            ax2.add_patch(mini_rect)
            ax2.text(i*7 + 1.5 + j*2.5, 2.25, f'GPU{i*2+j}',
                    ha='center', va='center', fontsize=9)
    
    # Arrow showing pipeline flow
    ax2.annotate('', xy=(7, 2.25), xytext=(6.2, 2.25),
                arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    
    ax2.text(7, 4, 'Result: 2 stages × 2 GPUs/stage = 4 GPUs total',
            ha='center', fontsize=12, color='darkgreen', weight='bold')
    
    ax2.set_xlim(-0.5, 14.5)
    ax2.set_ylim(0, 5)
    ax2.axis('off')
    
    plt.suptitle("Alpa's Two-Level Optimization Hierarchy", fontsize=16, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('alpa_hierarchy.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: alpa_hierarchy.png")
    plt.close()

# ============================================================================
# 5. PERFORMANCE COMPARISON
# ============================================================================

def visualize_performance_comparison():
    """Compare performance of different strategies"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    strategies = ['Data\nParallel', 'Pipeline\nParallel', 'Tensor\nParallel', 
                  'Manual\nCombined', 'Alpa\n(Auto)']
    
    # Simulated performance (relative to single GPU)
    speedups = [3.2, 2.8, 3.5, 4.2, 4.6]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#90EE90']
    
    bars = ax.bar(strategies, speedups, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{speedup}×', ha='center', va='bottom', fontsize=12, weight='bold')
    
    # Horizontal line at 1.0× (baseline)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(4.5, 1.1, 'Single GPU baseline', ha='right', fontsize=10, color='gray')
    
    # Highlight Alpa
    ax.text(4, 5, '✅ Best!', ha='center', fontsize=12, color='green', weight='bold')
    
    ax.set_ylabel('Speedup vs Single GPU', fontsize=13, weight='bold')
    ax.set_title('Performance Comparison: Manual vs Alpa (4 GPUs, GPT-2 Medium)', 
                fontsize=14, weight='bold')
    ax.set_ylim(0, 5.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add note
    ax.text(2.5, 0.3, 'Note: Alpa automatically finds best combination of strategies',
           ha='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: performance_comparison.png")
    plt.close()

# ============================================================================
# 6. COMMUNICATION VOLUME
# ============================================================================

def visualize_communication():
    """Show communication overhead for different strategies"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    strategies = ['Data\nParallel', 'Pipeline\nParallel', 'Tensor\nParallel', 
                  'Manual\nCombined', 'Alpa\n(Auto)']
    
    # Communication volume (GB per iteration, simulated)
    comm_volume = [28, 10, 35, 18, 12]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#90EE90']
    
    bars = ax.bar(strategies, comm_volume, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, vol in zip(bars, comm_volume):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{vol} GB', ha='center', va='bottom', fontsize=11, weight='bold')
    
    # Highlight Alpa
    ax.text(4, 14, '✅ Low comm!', ha='center', fontsize=12, color='green', weight='bold')
    
    ax.set_ylabel('Communication Volume (GB/iteration)', fontsize=13, weight='bold')
    ax.set_title('Communication Overhead Comparison (7B Model, 8 GPUs)', 
                fontsize=14, weight='bold')
    ax.set_ylim(0, 40)
    ax.grid(axis='y', alpha=0.3)
    
    # Add note
    ax.text(2.5, 2, 'Lower is better - Alpa minimizes communication automatically',
           ha='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig('communication_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: communication_comparison.png")
    plt.close()

# ============================================================================
# MAIN: GENERATE ALL VISUALIZATIONS
# ============================================================================

if __name__ == "__main__":
    print("Generating visualizations...")
    print()
    
    visualize_data_parallel()
    visualize_pipeline_parallel()
    visualize_tensor_parallel()
    visualize_alpa_hierarchy()
    visualize_performance_comparison()
    visualize_communication()
    
    print()
    print("=" * 80)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("=" * 80)
    print()
    print("Generated images:")
    print("  1. data_parallel.png - Data parallelism strategy")
    print("  2. pipeline_parallel.png - Pipeline parallelism strategy")
    print("  3. tensor_parallel.png - Tensor parallelism strategy")
    print("  4. alpa_hierarchy.png - Alpa's two-level optimization")
    print("  5. performance_comparison.png - Performance vs manual")
    print("  6. communication_comparison.png - Communication overhead")
    print()
    print("Open these PNG files to see the diagrams!")
    print()
