import numpy as np
import matplotlib.pyplot as plt
import math
from abc import ABC, abstractmethod

class LearningRateScheduler(ABC):
    """
    Base class for all learning rate schedulers
    
    Think of schedulers as the "speed controllers" for your model's learning.
    Just like you wouldn't drive at highway speed through a parking lot,
    you don't want your model learning too fast when it's close to the optimal solution.
    """
    
    def __init__(self, initial_lr=0.01):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
    
    @abstractmethod
    def get_lr(self, epoch, **kwargs):
        """Get learning rate for current epoch"""
        pass
    
    def reset(self):
        """Reset scheduler to initial state"""
        self.current_lr = self.initial_lr

class StepDecayScheduler(LearningRateScheduler):
    """
    Reduces learning rate by a factor every few epochs
    
    This is like taking breaks during a long hike - every so often, you slow down
    to catch your breath and reassess your direction. Very popular because it's
    simple and works well in practice.
    """
    
    def __init__(self, initial_lr=0.01, drop_rate=0.5, epochs_drop=10):
        super().__init__(initial_lr)
        self.drop_rate = drop_rate      # How much to multiply LR by (0.5 = half)
        self.epochs_drop = epochs_drop  # How often to drop (every 10 epochs)
    
    def get_lr(self, epoch, **kwargs):
        """
        Calculate learning rate using step decay formula:
        lr = initial_lr * (drop_rate ^ floor(epoch / epochs_drop))
        """
        # How many times have we dropped the learning rate so far?
        num_drops = math.floor(epoch / self.epochs_drop)
        
        # Apply the drop rate that many times
        self.current_lr = self.initial_lr * (self.drop_rate ** num_drops)
        return self.current_lr

class ExponentialDecayScheduler(LearningRateScheduler):
    """
    Smoothly reduces learning rate exponentially
    
    Unlike step decay which has sudden drops, this is like gradually slowing down
    as you approach your destination. The learning rate decreases smoothly,
    which can lead to more stable training.
    """
    
    def __init__(self, initial_lr=0.01, decay_rate=0.95):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate  # Closer to 1 = slower decay
    
    def get_lr(self, epoch, **kwargs):
        """
        Formula: lr = initial_lr * (decay_rate ^ epoch)
        """
        self.current_lr = self.initial_lr * (self.decay_rate ** epoch)
        return self.current_lr

class CosineAnnealingScheduler(LearningRateScheduler):
    """
    Learning rate follows a cosine curve - starts high, smoothly decreases
    
    This mimics the cosine function, giving a smooth, curved decrease.
    Popular in modern deep learning because it provides a nice balance:
    - Fast learning at the beginning
    - Gradual slowdown toward the end
    - No sudden jumps that might destabilize training
    """
    
    def __init__(self, initial_lr=0.01, min_lr=0.0001, max_epochs=100):
        super().__init__(initial_lr)
        self.min_lr = min_lr        # Floor - won't go below this
        self.max_epochs = max_epochs # Total training epochs
    
    def get_lr(self, epoch, **kwargs):
        """
        Formula: lr = min_lr + (initial_lr - min_lr) * (1 + cos(π * epoch / max_epochs)) / 2
        """
        # Cosine annealing formula - creates a smooth curve from initial_lr to min_lr
        cosine_factor = (1 + math.cos(math.pi * epoch / self.max_epochs)) / 2
        self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_factor
        return self.current_lr

class WarmupCosineScheduler(LearningRateScheduler):
    """
    Combines warmup with cosine annealing
    
    Many modern models benefit from "warming up" - starting with a very low
    learning rate and gradually increasing it before the main training.
    Think of it like warming up your car engine before driving.
    """
    
    def __init__(self, initial_lr=0.01, min_lr=0.0001, warmup_epochs=10, max_epochs=100):
        super().__init__(initial_lr)
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
    
    def get_lr(self, epoch, **kwargs):
        if epoch < self.warmup_epochs:
            # Warmup phase: linearly increase from min_lr to initial_lr
            progress = epoch / self.warmup_epochs
            self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * progress
        else:
            # Cosine annealing phase
            adjusted_epoch = epoch - self.warmup_epochs
            adjusted_max_epochs = self.max_epochs - self.warmup_epochs
            cosine_factor = (1 + math.cos(math.pi * adjusted_epoch / adjusted_max_epochs)) / 2
            self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_factor
        
        return self.current_lr

class ReduceLROnPlateauScheduler(LearningRateScheduler):
    """
    Reduces learning rate when a metric stops improving
    
    This is the "smart" scheduler - it actually watches your model's performance
    and only reduces the learning rate when progress stalls. Like a GPS that
    suggests a different route when you hit traffic.
    """
    
    def __init__(self, initial_lr=0.01, factor=0.5, patience=5, min_lr=1e-7):
        super().__init__(initial_lr)
        self.factor = factor        # How much to reduce by (0.5 = half)
        self.patience = patience    # How many epochs to wait before reducing
        self.min_lr = min_lr       # Don't go below this
        
        # Internal tracking
        self.best_metric = float('inf')  # Assume we're minimizing (like loss)
        self.wait_count = 0
        self.mode = 'min'  # 'min' for loss, 'max' for accuracy
    
    def get_lr(self, epoch, metric_value=None, **kwargs):
        """
        This scheduler needs the current metric value (like validation loss)
        """
        if metric_value is None:
            # If no metric provided, just return current learning rate
            return self.current_lr
        
        # Check if we've improved
        if self.mode == 'min':
            improved = metric_value < self.best_metric
        else:  # mode == 'max'
            improved = metric_value > self.best_metric
        
        if improved:
            # We're making progress! Reset the waiting counter
            self.best_metric = metric_value
            self.wait_count = 0
        else:
            # No improvement - increment wait counter
            self.wait_count += 1
            
            # Have we waited long enough? Time to reduce learning rate
            if self.wait_count >= self.patience:
                old_lr = self.current_lr
                self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                self.wait_count = 0  # Reset counter after reduction
                
                if self.current_lr < old_lr:
                    print(f"Epoch {epoch}: Reducing learning rate from {old_lr:.6f} to {self.current_lr:.6f}")
        
        return self.current_lr

class CyclicalLRScheduler(LearningRateScheduler):
    """
    Learning rate cycles between a base and maximum value
    
    This is based on the idea that sometimes you need to "shake things up"
    during training. By cycling the learning rate, you can help the model
    escape local minima and potentially find better solutions.
    """
    
    def __init__(self, base_lr=0.001, max_lr=0.01, step_size=10, mode='triangular'):
        super().__init__(base_lr)
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size  # Half of the cycle length
        self.mode = mode
    
    def get_lr(self, epoch, **kwargs):
        """
        Creates a triangular wave pattern for learning rate
        """
        # Calculate where we are in the current cycle
        cycle = math.floor(1 + epoch / (2 * self.step_size))
        x = abs(epoch / self.step_size - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            # Simple triangular wave
            self.current_lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x)
        elif self.mode == 'triangular2':
            # Triangular wave that decreases in amplitude over time
            self.current_lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x) / (2 ** (cycle - 1))
        
        return self.current_lr

def visualize_schedulers():
    """
    Create visualizations showing how different schedulers behave over time
    This helps you understand which scheduler might work best for your use case
    """
    epochs = 100
    epoch_range = range(epochs)
    
    # Initialize all schedulers
    schedulers = {
        'Step Decay': StepDecayScheduler(initial_lr=0.1, drop_rate=0.5, epochs_drop=20),
        'Exponential Decay': ExponentialDecayScheduler(initial_lr=0.1, decay_rate=0.95),
        'Cosine Annealing': CosineAnnealingScheduler(initial_lr=0.1, min_lr=0.001, max_epochs=epochs),
        'Warmup + Cosine': WarmupCosineScheduler(initial_lr=0.1, min_lr=0.001, warmup_epochs=10, max_epochs=epochs),
        'Cyclical LR': CyclicalLRScheduler(base_lr=0.01, max_lr=0.1, step_size=15)
    }
    
    # For Reduce on Plateau, we'll simulate some metric values
    plateau_scheduler = ReduceLROnPlateauScheduler(initial_lr=0.1, factor=0.5, patience=8)
    
    # Create the plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Learning Rate Schedulers Comparison', fontsize=16, fontweight='bold')
    
    # Plot each scheduler
    for idx, (name, scheduler) in enumerate(schedulers.items()):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        # Reset scheduler and collect learning rates
        scheduler.reset()
        learning_rates = []
        
        for epoch in epoch_range:
            lr = scheduler.get_lr(epoch)
            learning_rates.append(lr)
        
        # Plot
        ax.plot(epoch_range, learning_rates, linewidth=2, color=f'C{idx}')
        ax.set_title(f'{name}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale to better show the differences
    
    # Special handling for Reduce on Plateau (needs metric values)
    ax = axes[1, 2]
    learning_rates = []
    
    # Simulate a training scenario where loss decreases then plateaus
    np.random.seed(42)  # For reproducible "noise"
    simulated_loss = []
    base_loss = 2.0
    
    for epoch in epoch_range:
        # Simulate loss decreasing quickly at first, then plateauing with noise
        if epoch < 30:
            # Good progress phase
            base_loss *= 0.95
            noise = np.random.normal(0, 0.05)
        elif epoch < 60:
            # Slower progress phase
            base_loss *= 0.99
            noise = np.random.normal(0, 0.1)
        else:
            # Plateau phase - no real progress
            noise = np.random.normal(0, 0.1)
        
        current_loss = base_loss + noise
        simulated_loss.append(current_loss)
        
        lr = plateau_scheduler.get_lr(epoch, metric_value=current_loss)
        learning_rates.append(lr)
    
    ax.plot(epoch_range, learning_rates, linewidth=2, color='C5')
    ax.set_title('Reduce LR on Plateau', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Create a second plot showing the simulated loss for the plateau scheduler
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(epoch_range, simulated_loss, color='red', alpha=0.7)
    plt.title('Simulated Validation Loss (for Plateau Scheduler)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(epoch_range, learning_rates, color='blue', linewidth=2)
    plt.title('Learning Rate Response')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def scheduler_recommendations():
    """
    Print practical recommendations for when to use each scheduler
    """
    print("LEARNING RATE SCHEDULER RECOMMENDATIONS")
    print("=" * 50)
    
    recommendations = {
        "Step Decay": {
            "best_for": ["Traditional neural networks", "When you want predictable LR changes"],
            "pros": ["Simple to understand", "Works reliably", "Easy to tune"],
            "cons": ["Sudden jumps can be harsh", "Need to tune drop schedule"],
            "typical_params": "drop_rate=0.1-0.5, epochs_drop=10-30"
        },
        
        "Exponential Decay": {
            "best_for": ["Smooth training", "When you want gradual changes"],
            "pros": ["Smooth decrease", "No sudden jumps", "Mathematically elegant"],
            "cons": ["Can decay too fast or too slow", "Less control over timing"],
            "typical_params": "decay_rate=0.9-0.99 (closer to 1 = slower decay)"
        },
        
        "Cosine Annealing": {
            "best_for": ["Modern deep learning", "Transformer models", "When training time is fixed"],
            "pros": ["Smooth curve", "Popular in SOTA models", "Good final convergence"],
            "cons": ["Need to know total epochs in advance"],
            "typical_params": "min_lr=1e-6 to 1e-4"
        },
        
        "Warmup + Cosine": {
            "best_for": ["Large models", "Transformer architectures", "High learning rates"],
            "pros": ["Stable start", "Smooth progression", "Used in BERT, GPT, etc."],
            "cons": ["More complex", "Need to tune warmup period"],
            "typical_params": "warmup_epochs=5-10% of total epochs"
        },
        
        "Reduce on Plateau": {
            "best_for": ["When you can monitor validation metrics", "Unpredictable training"],
            "pros": ["Adaptive", "Only reduces when needed", "No need to plan schedule"],
            "cons": ["Reactive (not proactive)", "Depends on metric noise"],
            "typical_params": "patience=5-15, factor=0.1-0.5"
        },
        
        "Cyclical LR": {
            "best_for": ["Finding good learning rates", "Escaping local minima"],
            "pros": ["Can escape local minima", "Helps find optimal LR range"],
            "cons": ["Can be unstable", "More experimental"],
            "typical_params": "max_lr = 3-10x base_lr"
        }
    }
    
    for scheduler, info in recommendations.items():
        print(f"\n{scheduler.upper()}")
        print("-" * len(scheduler))
        print(f"Best for: {', '.join(info['best_for'])}")
        print(f"Pros: {', '.join(info['pros'])}")
        print(f"Cons: {', '.join(info['cons'])}")
        print(f"Typical params: {info['typical_params']}")

# Example usage and demonstration
if __name__ == "__main__":
    print("LEARNING RATE SCHEDULERS - Complete Guide")
    print("=" * 50)
    
    # Show visualizations
    print("Generating scheduler visualizations...")
    visualize_schedulers()
    
    # Print recommendations
    scheduler_recommendations()
    
    print("\nQUICK START GUIDE:")
    print("1. For most cases: Start with Cosine Annealing")
    print("2. For large models: Use Warmup + Cosine") 
    print("3. For experimental work: Try Reduce on Plateau")
    print("4. For traditional networks: Step Decay works fine")
    print("5. For finding optimal LR: Use Cyclical LR for exploration")
    
    print("\nTIPS:")
    print("- Always monitor your training/validation curves")
    print("- Learning rate is often the most important hyperparameter")
    print("- When in doubt, start high and let the scheduler bring it down")
    print("- Don't be afraid to experiment - different problems need different approaches")
