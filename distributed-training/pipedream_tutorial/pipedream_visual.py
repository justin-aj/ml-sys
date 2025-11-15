"""
PipeDream Visual Comparison: Naive vs Microbatch Pipeline

This script creates a visual comparison showing:
1. Naive pipeline (low GPU utilization)
2. Microbatch pipeline (high GPU utilization)

Author: Educational tutorial
Date: November 15, 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np


def plot_naive_pipeline():
    """
    Visualize naive pipeline parallelism.
    Shows how GPUs are idle most of the time.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Timeline
    num_stages = 4
    total_time = num_stages * 2  # Forward + backward
    
    # Colors
    forward_color = '#3498db'  # Blue
    backward_color = '#e74c3c'  # Red
    idle_color = '#ecf0f1'  # Light gray
    
    # Draw timeline for each GPU
    for gpu_id in range(num_stages):
        y_pos = gpu_id
        
        # Forward pass - only one GPU active at a time
        for t in range(total_time):
            if t == gpu_id:
                # This GPU doing forward
                rect = Rectangle((t, y_pos - 0.4), 1, 0.8, 
                               facecolor=forward_color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(t + 0.5, y_pos, 'FWD', ha='center', va='center', 
                       fontsize=10, fontweight='bold', color='white')
            elif t == (total_time - 1 - gpu_id):
                # This GPU doing backward
                rect = Rectangle((t, y_pos - 0.4), 1, 0.8,
                               facecolor=backward_color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(t + 0.5, y_pos, 'BWD', ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white')
            else:
                # Idle
                rect = Rectangle((t, y_pos - 0.4), 1, 0.8,
                               facecolor=idle_color, edgecolor='gray', linewidth=0.5)
                ax.add_patch(rect)
                ax.text(t + 0.5, y_pos, 'idle', ha='center', va='center',
                       fontsize=8, color='gray', style='italic')
    
    # Formatting
    ax.set_xlim(0, total_time)
    ax.set_ylim(-0.5, num_stages - 0.5)
    ax.set_yticks(range(num_stages))
    ax.set_yticklabels([f'GPU {i}\n(Stage {i})' for i in range(num_stages)])
    ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax.set_title('❌ Naive Pipeline Parallelism\n(Only 1 GPU working at a time - 25% utilization)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Legend
    forward_patch = mpatches.Patch(color=forward_color, label='Forward Pass')
    backward_patch = mpatches.Patch(color=backward_color, label='Backward Pass')
    idle_patch = mpatches.Patch(color=idle_color, label='Idle (Wasted!)')
    ax.legend(handles=[forward_patch, backward_patch, idle_patch], 
             loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('naive_pipeline.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: naive_pipeline.png")
    return fig


def plot_microbatch_pipeline():
    """
    Visualize microbatch pipeline parallelism.
    Shows how all GPUs stay busy.
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Timeline
    num_stages = 4
    num_microbatches = 4
    total_time = num_stages + num_microbatches - 1
    
    # Colors for different microbatches
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']  # Blue, green, orange, purple
    backward_colors = ['#2980b9', '#27ae60', '#e67e22', '#8e44ad']  # Darker versions
    
    # Forward pass
    for gpu_id in range(num_stages):
        y_pos = gpu_id
        
        for t in range(total_time):
            mb = t - gpu_id
            if 0 <= mb < num_microbatches:
                # This GPU processing microbatch
                rect = Rectangle((t, y_pos - 0.4), 1, 0.8,
                               facecolor=colors[mb], edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(t + 0.5, y_pos, f'MB{mb}\nFWD', ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')
            else:
                # Idle
                rect = Rectangle((t, y_pos - 0.4), 1, 0.8,
                               facecolor='#ecf0f1', edgecolor='gray', linewidth=0.5)
                ax.add_patch(rect)
    
    # Formatting
    ax.set_xlim(0, total_time)
    ax.set_ylim(-0.5, num_stages - 0.5)
    ax.set_yticks(range(num_stages))
    ax.set_yticklabels([f'GPU {i}\n(Stage {i})' for i in range(num_stages)])
    ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax.set_title('✅ Microbatch Pipeline (Forward Pass)\n(3-4 GPUs working simultaneously - 75% utilization!)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Legend
    patches = [mpatches.Patch(color=colors[i], label=f'Microbatch {i}') 
              for i in range(num_microbatches)]
    patches.append(mpatches.Patch(color='#ecf0f1', label='Idle'))
    ax.legend(handles=patches, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('microbatch_forward.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: microbatch_forward.png")
    return fig


def plot_microbatch_backward():
    """
    Visualize backward pass in microbatch pipeline.
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Timeline
    num_stages = 4
    num_microbatches = 4
    total_time = num_stages + num_microbatches - 1
    
    # Colors for backward (darker shades)
    backward_colors = ['#2980b9', '#27ae60', '#e67e22', '#8e44ad']
    
    # Backward pass
    for gpu_id in range(num_stages):
        y_pos = gpu_id
        
        for t in range(total_time):
            # Backward flows in reverse (GPU3 first, then GPU2, GPU1, GPU0)
            mb = t - (num_stages - 1 - gpu_id)
            if 0 <= mb < num_microbatches:
                # This GPU processing microbatch backward
                rect = Rectangle((t, y_pos - 0.4), 1, 0.8,
                               facecolor=backward_colors[mb], edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(t + 0.5, y_pos, f'MB{mb}\nBWD', ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')
            else:
                # Idle
                rect = Rectangle((t, y_pos - 0.4), 1, 0.8,
                               facecolor='#ecf0f1', edgecolor='gray', linewidth=0.5)
                ax.add_patch(rect)
    
    # Formatting
    ax.set_xlim(0, total_time)
    ax.set_ylim(-0.5, num_stages - 0.5)
    ax.set_yticks(range(num_stages))
    ax.set_yticklabels([f'GPU {i}\n(Stage {i})' for i in range(num_stages)])
    ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax.set_title('✅ Microbatch Pipeline (Backward Pass)\n(Gradients flow in reverse - all GPUs busy!)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Legend
    patches = [mpatches.Patch(color=backward_colors[i], label=f'Microbatch {i} (backward)') 
              for i in range(num_microbatches)]
    patches.append(mpatches.Patch(color='#ecf0f1', label='Idle'))
    ax.legend(handles=patches, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('microbatch_backward.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: microbatch_backward.png")
    return fig


def plot_weight_versioning():
    """
    Visualize weight versioning concept.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Draw weight versions
    num_microbatches = 4
    
    # Y positions
    y_forward = 3
    y_backward = 2
    y_weights = 1
    
    # Timeline
    for mb in range(num_microbatches):
        x_pos = mb * 3
        
        # Forward pass (uses W0)
        rect = Rectangle((x_pos, y_forward - 0.3), 2.5, 0.6,
                       facecolor='#3498db', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x_pos + 1.25, y_forward, f'MB{mb} Forward\n(uses W₀)', 
               ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        
        # Backward pass (computes grad for W0)
        rect = Rectangle((x_pos, y_backward - 0.3), 2.5, 0.6,
                       facecolor='#e74c3c', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x_pos + 1.25, y_backward, f'MB{mb} Backward\n(grad for W₀)',
               ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        
        # Arrow showing dependency
        ax.annotate('', xy=(x_pos + 1.25, y_backward + 0.35), 
                   xytext=(x_pos + 1.25, y_forward - 0.35),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Weight version box
    rect = Rectangle((0, y_weights - 0.3), num_microbatches * 3 - 0.5, 0.6,
                   facecolor='#2ecc71', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text((num_microbatches * 3 - 0.5) / 2, y_weights,
           'All microbatches use Weight Version 0 (W₀)',
           ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Update arrow
    update_x = num_microbatches * 3 + 1
    ax.annotate('Weight\nUpdate', xy=(update_x, y_weights),
               xytext=(update_x - 2, y_weights),
               arrowprops=dict(arrowstyle='->', lw=3, color='#2ecc71'),
               fontsize=12, fontweight='bold', ha='center')
    
    # Next version
    rect = Rectangle((update_x, y_weights - 0.3), 1.5, 0.6,
                   facecolor='#f39c12', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(update_x + 0.75, y_weights, 'W₁',
           ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Formatting
    ax.set_xlim(-0.5, update_x + 2)
    ax.set_ylim(0.5, 3.5)
    ax.set_yticks([y_weights, y_backward, y_forward])
    ax.set_yticklabels(['Weights', 'Backward', 'Forward'])
    ax.set_xticks([])
    ax.set_title('🔑 PipeDream Weight Versioning\n(All microbatches use same version for consistency)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Add text explanation
    ax.text((num_microbatches * 3 - 0.5) / 2, 0.2,
           '✅ Key Insight: Gradients from all microbatches apply to W₀, creating W₁\n'
           'This ensures correctness - backward uses the same weights as forward!',
           ha='center', va='bottom', fontsize=11, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('weight_versioning.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: weight_versioning.png")
    return fig


def plot_utilization_comparison():
    """
    Bar chart comparing GPU utilization.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = ['Naive\nPipeline', 'Microbatch\nPipeline', 'PipeDream\n(Optimized)']
    utilization = [25, 75, 90]  # Approximate percentages
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    
    bars = ax.bar(strategies, utilization, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, val in zip(bars, utilization):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Formatting
    ax.set_ylabel('GPU Utilization (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_title('GPU Utilization Comparison\n(Higher is better!)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('utilization_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: utilization_comparison.png")
    return fig


def main():
    """Generate all visualizations."""
    print("\n" + "=" * 80)
    print("📊 GENERATING PIPEDREAM VISUALIZATIONS")
    print("=" * 80)
    
    print("\n1. Naive pipeline (low utilization)...")
    plot_naive_pipeline()
    
    print("\n2. Microbatch pipeline - forward pass...")
    plot_microbatch_pipeline()
    
    print("\n3. Microbatch pipeline - backward pass...")
    plot_microbatch_backward()
    
    print("\n4. Weight versioning concept...")
    plot_weight_versioning()
    
    print("\n5. Utilization comparison...")
    plot_utilization_comparison()
    
    print("\n" + "=" * 80)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. naive_pipeline.png - Shows idle GPUs problem")
    print("  2. microbatch_forward.png - Forward pass with microbatches")
    print("  3. microbatch_backward.png - Backward pass with microbatches")
    print("  4. weight_versioning.png - Weight versioning concept")
    print("  5. utilization_comparison.png - Efficiency comparison")
    print("\n💡 Open these images to understand PipeDream visually!")
    print("=" * 80)


if __name__ == "__main__":
    main()
    plt.show()  # Display all plots
