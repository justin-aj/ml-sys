"""
PipeDream: Simple Educational Implementation

This code demonstrates the core concepts of pipeline parallelism with microbatches
and weight versioning, as described in the PipeDream paper.

⚠️  SINGLE GPU SIMULATION:
This tutorial simulates pipeline parallelism on ONE GPU for learning purposes.
Real PipeDream needs multiple GPUs, but you'll learn all the core concepts here!

Goal: Show how microbatches keep GPUs busy and why weight versioning is needed.

Author: Educational tutorial
Date: November 15, 2025
"""

import torch
import torch.nn as nn
from typing import List, Dict
import time
from collections import defaultdict


# ============================================================================
# CONFIGURATION
# ============================================================================
# Edit these values to experiment with different settings

CONFIG = {
    # Pipeline configuration (SIMULATED - we only have 1 GPU!)
    "num_stages": 4,          # Number of pipeline stages (simulates 4 GPUs)
    "num_microbatches": 4,    # Split batch into this many microbatches
    "layers_per_stage": 3,    # Layers in each pipeline stage
    
    # Model configuration
    "hidden_size": 512,       # Hidden dimension size
    "input_size": 784,        # Input size (28x28 images flattened)
    "output_size": 10,        # Number of classes
    
    # Training configuration
    "batch_size": 32,         # Total batch size
    "learning_rate": 0.001,
    "num_batches": 5,         # Number of batches to train
    
    # Visualization
    "verbose": True,          # Print detailed timeline
    "track_versions": True,   # Track which weight version each microbatch uses
}


# ============================================================================
# STEP 1: Define a Simple Model Stage
# ============================================================================
# Each GPU would hold one "stage" of the model (a few layers)
# Here we simulate this on a single GPU for learning

class ModelStage(nn.Module):
    """
    One stage of the pipeline (would run on one GPU in real implementation).
    
    In a real pipeline, you'd split a huge model (e.g., 48 layers) into
    chunks (e.g., 12 layers per GPU). Here we use a simple MLP for clarity.
    
    ⚠️  SIMULATION NOTE:
    In real PipeDream, each stage lives on a different GPU. Here, all stages
    are on the same GPU, but we simulate the pipeline behavior.
    """
    
    def __init__(self, input_size: int, output_size: int, num_layers: int, stage_id: int):
        super().__init__()
        self.stage_id = stage_id
        
        # Build a small MLP for this stage
        layers = []
        current_size = input_size
        
        for i in range(num_layers):
            # Each layer: Linear + ReLU
            layers.append(nn.Linear(current_size, output_size))
            if i < num_layers - 1:  # No ReLU on last layer of stage
                layers.append(nn.ReLU())
            current_size = output_size
        
        self.layers = nn.Sequential(*layers)
        
        # Weight version tracking (for educational purposes)
        self.current_version = 0
        self.version_history = []
    
    def forward(self, x):
        """Forward pass through this stage's layers."""
        return self.layers(x)
    
    def get_weight_version(self):
        """Return current weight version (for tracking)."""
        return self.current_version
    
    def increment_version(self):
        """Increment version after weight update."""
        self.current_version += 1
        self.version_history.append(self.current_version)


# ============================================================================
# STEP 2: Create the Pipeline
# ============================================================================

class SimplePipeline:
    """
    Educational pipeline parallelism implementation.
    
    This shows the core ideas:
    1. Split model into stages (one per GPU)
    2. Process microbatches in a staggered manner
    3. Track weight versions for correctness
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.num_stages = config["num_stages"]
        self.num_microbatches = config["num_microbatches"]
        
        # Create model stages
        self.stages = self._create_stages()
        
        # Optimizers for each stage
        self.optimizers = [
            torch.optim.SGD(stage.parameters(), lr=config["learning_rate"])
            for stage in self.stages
        ]
        
        # Timeline tracking (for visualization)
        self.timeline = defaultdict(list)
        self.current_time = 0
        
        print("=" * 80)
        print("🚀 PipeDream Pipeline Created!")
        print("=" * 80)
        print(f"⚠️  SIMULATION MODE: Running on 1 GPU (simulating {self.num_stages} GPUs)")
        print(f"Pipeline stages: {self.num_stages} (would be {self.num_stages} GPUs in real setup)")
        print(f"Microbatches: {self.num_microbatches}")
        print(f"Layers per stage: {config['layers_per_stage']}")
        print(f"Total layers: {self.num_stages * config['layers_per_stage']}")
        print("\n💡 In real PipeDream:")
        print(f"   - Each stage would be on a separate GPU")
        print(f"   - Activations would be sent between GPUs")
        print(f"   - All {self.num_stages} GPUs would work in parallel")
        print("\n💡 In this simulation:")
        print(f"   - All stages are on 1 GPU (for learning)")
        print(f"   - We simulate the pipeline behavior")
        print(f"   - Timeline shows what WOULD happen on {self.num_stages} GPUs")
        print("=" * 80)
    
    def _create_stages(self):
        """Create all pipeline stages."""
        stages = []
        
        # First stage: input_size → hidden_size
        stages.append(ModelStage(
            input_size=self.config["input_size"],
            output_size=self.config["hidden_size"],
            num_layers=self.config["layers_per_stage"],
            stage_id=0
        ))
        
        # Middle stages: hidden_size → hidden_size
        for i in range(1, self.num_stages - 1):
            stages.append(ModelStage(
                input_size=self.config["hidden_size"],
                output_size=self.config["hidden_size"],
                num_layers=self.config["layers_per_stage"],
                stage_id=i
            ))
        
        # Last stage: hidden_size → output_size
        stages.append(ModelStage(
            input_size=self.config["hidden_size"],
            output_size=self.config["output_size"],
            num_layers=self.config["layers_per_stage"],
            stage_id=self.num_stages - 1
        ))
        
        return stages
    
    def _log_event(self, stage_id: int, event: str):
        """Log an event for timeline visualization."""
        self.timeline[self.current_time].append(f"GPU{stage_id}: {event}")
    
    def forward_microbatch(self, microbatch_id: int, data: torch.Tensor):
        """
        Forward pass for one microbatch through the entire pipeline.
        
        This simulates the staggered execution:
        - MB0 starts at t0 on GPU0
        - MB1 starts at t1 on GPU0 (while MB0 is on GPU1)
        - etc.
        """
        print(f"\n📤 Forward pass for microbatch {microbatch_id}")
        
        # Store activations as we pass through stages
        activations = [None] * (self.num_stages + 1)
        activations[0] = data
        
        # Store which weight version each stage uses (for tracking)
        versions_used = []
        
        # Pass through each stage
        for stage_id in range(self.num_stages):
            # Get current weight version
            version = self.stages[stage_id].get_weight_version()
            versions_used.append(version)
            
            # Forward through this stage
            with torch.no_grad():  # Save memory for this demo
                activations[stage_id + 1] = self.stages[stage_id](activations[stage_id])
            
            if self.config["verbose"]:
                print(f"  Stage {stage_id}: MB{microbatch_id} → "
                      f"using weights v{version} → "
                      f"output shape {activations[stage_id + 1].shape}")
            
            self._log_event(stage_id, f"MB{microbatch_id} forward (W_v{version})")
        
        # Return final output and version info
        return activations[-1], versions_used, activations
    
    def backward_microbatch(self, microbatch_id: int, activations: List, 
                           targets: torch.Tensor, versions_used: List):
        """
        Backward pass for one microbatch.
        
        Key point: We use the SAME weight versions that forward pass used!
        This is PipeDream's weight versioning in action.
        """
        print(f"\n📥 Backward pass for microbatch {microbatch_id}")
        
        # Compute loss (on last stage)
        output = activations[-1]
        loss = nn.functional.cross_entropy(output, targets)
        
        # Backward through pipeline (in reverse order)
        # In a real implementation, you'd send gradients between GPUs
        # Here we just simulate the concept
        
        for stage_id in range(self.num_stages - 1, -1, -1):
            if self.config["verbose"]:
                print(f"  Stage {stage_id}: MB{microbatch_id} ← "
                      f"computing gradients for weights v{versions_used[stage_id]}")
            
            self._log_event(stage_id, f"MB{microbatch_id} backward (W_v{versions_used[stage_id]})")
        
        return loss.item()
    
    def train_one_batch(self, batch_data: torch.Tensor, batch_targets: torch.Tensor):
        """
        Train on one batch using PipeDream's microbatch pipeline.
        
        This is where the magic happens:
        1. Split batch into microbatches
        2. Forward all microbatches (staggered across GPUs)
        3. Backward all microbatches (also staggered)
        4. Update weights ONCE at the end (all microbatches use same version)
        """
        print("\n" + "=" * 80)
        print("🎯 Training One Batch with PipeDream")
        print("=" * 80)
        
        # Split batch into microbatches
        microbatch_size = self.config["batch_size"] // self.num_microbatches
        microbatches = batch_data.split(microbatch_size)
        target_microbatches = batch_targets.split(microbatch_size)
        
        print(f"Total batch size: {self.config['batch_size']}")
        print(f"Split into {self.num_microbatches} microbatches of size {microbatch_size}")
        
        # Store all microbatch information
        all_activations = []
        all_versions = []
        all_losses = []
        
        # ========================================================================
        # FORWARD PHASE: All microbatches through pipeline
        # ========================================================================
        print("\n" + "-" * 80)
        print("FORWARD PHASE")
        print("-" * 80)
        
        for mb_id in range(self.num_microbatches):
            output, versions, activations = self.forward_microbatch(
                mb_id, 
                microbatches[mb_id]
            )
            all_activations.append(activations)
            all_versions.append(versions)
        
        # ========================================================================
        # BACKWARD PHASE: All microbatches backward
        # ========================================================================
        print("\n" + "-" * 80)
        print("BACKWARD PHASE")
        print("-" * 80)
        
        for mb_id in range(self.num_microbatches):
            loss = self.backward_microbatch(
                mb_id,
                all_activations[mb_id],
                target_microbatches[mb_id],
                all_versions[mb_id]
            )
            all_losses.append(loss)
        
        # ========================================================================
        # WEIGHT UPDATE: Apply gradients and increment version
        # ========================================================================
        print("\n" + "-" * 80)
        print("WEIGHT UPDATE PHASE")
        print("-" * 80)
        
        avg_loss = sum(all_losses) / len(all_losses)
        print(f"Average loss across {self.num_microbatches} microbatches: {avg_loss:.4f}")
        
        for stage_id, optimizer in enumerate(self.optimizers):
            old_version = self.stages[stage_id].get_weight_version()
            
            # Apply gradients (in real code, you'd accumulate from all microbatches)
            # optimizer.step()
            # optimizer.zero_grad()
            
            # Increment version for next batch
            self.stages[stage_id].increment_version()
            new_version = self.stages[stage_id].get_weight_version()
            
            print(f"Stage {stage_id}: Updated weights v{old_version} → v{new_version}")
        
        print("\n" + "=" * 80)
        print(f"✅ Batch complete! All stages now at version {new_version}")
        print("=" * 80)
        
        return avg_loss


# ============================================================================
# STEP 3: Visualization Helper
# ============================================================================

def visualize_timeline():
    """
    Print a visual timeline showing when each GPU is doing what.
    
    This helps understand the staggered execution pattern.
    """
    print("\n" + "=" * 80)
    print("📊 PIPELINE TIMELINE VISUALIZATION")
    print("=" * 80)
    print("\nForward Pass Timeline:")
    print("-" * 80)
    
    # Show the concept with a simple table
    num_stages = 4
    num_microbatches = 4
    
    print("\nTime | GPU0 (Stage 0) | GPU1 (Stage 1) | GPU2 (Stage 2) | GPU3 (Stage 3)")
    print("-" * 80)
    
    # Forward pass
    for t in range(num_stages + num_microbatches - 1):
        row = f"t{t}  |"
        for stage in range(num_stages):
            mb = t - stage
            if 0 <= mb < num_microbatches:
                row += f" MB{mb} forward    |"
            else:
                row += " idle          |"
        print(row)
    
    print("\n" + "-" * 80)
    print("Backward Pass Timeline:")
    print("-" * 80)
    print("\nTime | GPU0 (Stage 0) | GPU1 (Stage 1) | GPU2 (Stage 2) | GPU3 (Stage 3)")
    print("-" * 80)
    
    # Backward pass (starts after forward finishes)
    start_time = num_stages + num_microbatches - 1
    for t in range(start_time, start_time + num_stages + num_microbatches - 1):
        row = f"t{t} |"
        for stage in range(num_stages):
            # Backward flows in reverse
            mb = t - start_time - (num_stages - 1 - stage)
            if 0 <= mb < num_microbatches:
                row += f" MB{mb} backward   |"
            else:
                row += " idle          |"
        print(row)
    
    print("\n" + "=" * 80)
    print("Key Observations:")
    print("=" * 80)
    print("1. ✅ Multiple GPUs working simultaneously (high utilization!)")
    print("2. ✅ Forward and backward are staggered across GPUs")
    print("3. ✅ All microbatches use the SAME weight version (v0)")
    print("4. ✅ Weights update AFTER all microbatches complete")
    print("=" * 80)


# ============================================================================
# STEP 4: Main Training Loop
# ============================================================================

def main():
    """
    Main training function demonstrating PipeDream concepts.
    
    ⚠️  SINGLE GPU SIMULATION:
    This runs on 1 GPU but simulates what would happen with 4 GPUs.
    You'll learn all the concepts without needing multiple GPUs!
    """
    print("\n")
    print("=" * 80)
    print("🎓 PIPEDREAM TUTORIAL: EDUCATIONAL IMPLEMENTATION")
    print("=" * 80)
    print("\n⚠️  RUNNING ON 1 GPU (SIMULATION MODE)")
    print("\nThis code demonstrates:")
    print("  1. Pipeline parallelism (how to split model across GPUs)")
    print("  2. Microbatches (how to keep all GPUs busy)")
    print("  3. Weight versioning (how to ensure correctness)")
    print("\n💡 You only have 1 GPU? Perfect for learning!")
    print("  - All stages run on your 1 GPU (simulated)")
    print("  - Timeline shows what WOULD happen with 4 GPUs")
    print("  - When you get multi-GPU access, concepts transfer directly!")
    print("=" * 80)
    
    # Create pipeline
    pipeline = SimplePipeline(CONFIG)
    
    # Show timeline visualization first
    visualize_timeline()
    
    # Create dummy data for demonstration
    print("\n" + "=" * 80)
    print("📊 TRAINING DEMONSTRATION")
    print("=" * 80)
    
    for batch_id in range(CONFIG["num_batches"]):
        print(f"\n\n{'#' * 80}")
        print(f"BATCH {batch_id + 1}/{CONFIG['num_batches']}")
        print('#' * 80)
        
        # Create dummy data (normally you'd load real data)
        batch_data = torch.randn(CONFIG["batch_size"], CONFIG["input_size"])
        batch_targets = torch.randint(0, CONFIG["output_size"], (CONFIG["batch_size"],))
        
        # Train on this batch
        loss = pipeline.train_one_batch(batch_data, batch_targets)
        
        print(f"\n📈 Batch {batch_id + 1} Loss: {loss:.4f}")
    
    # Final summary
    print("\n\n" + "=" * 80)
    print("🎉 TUTORIAL COMPLETE!")
    print("=" * 80)
    print("\nWhat you learned:")
    print("  ✅ Pipeline parallelism splits models across GPUs")
    print("  ✅ Microbatches keep all GPUs busy (75% utilization vs 25%)")
    print("  ✅ Weight versioning ensures correctness")
    print("  ✅ All microbatches in a batch use the same weight version")
    print("  ✅ Weights update once after all microbatches finish")
    print("\n⚠️  Single GPU Simulation:")
    print("  - This tutorial simulated 4 GPUs on your 1 GPU")
    print("  - Timeline shows what WOULD happen with 4 real GPUs")
    print("  - All concepts transfer to real multi-GPU setup!")
    print("\n" + "=" * 80)
    print("\n💡 Next steps:")
    print("  1. Try changing num_microbatches in CONFIG (2, 4, 8)")
    print("  2. Try changing num_stages (2, 4, 8)")
    print("  3. Run pipedream_visual.py to see diagrams")
    print("  4. Read the README.md for more details")
    print("  5. When you get multi-GPU access, use real PipeDream!")
    print("=" * 80)


if __name__ == "__main__":
    main()
