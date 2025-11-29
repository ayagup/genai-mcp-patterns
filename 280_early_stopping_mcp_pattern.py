"""
Pattern 280: Early Stopping MCP Pattern

This pattern demonstrates early stopping strategies for training optimization,
including convergence detection, performance monitoring, and resource conservation.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class EarlyStoppingState(TypedDict):
    """State for early stopping workflow"""
    messages: Annotated[List[str], add]
    training_history: Dict[str, Any]
    convergence_analysis: Dict[str, Any]
    stopping_decision: Dict[str, Any]
    optimization_summary: Dict[str, Any]


class TrainingMonitor:
    """Monitors training progress"""
    
    def __init__(self):
        self.metrics = ["loss", "accuracy", "validation_loss", "validation_accuracy"]
    
    def simulate_training(self, epochs: int = 50) -> List[Dict[str, float]]:
        """Simulate training history"""
        import math
        
        history = []
        
        for epoch in range(epochs):
            # Simulate learning curves with plateau
            progress = epoch / epochs
            
            # Training loss decreases exponentially then plateaus
            if epoch < 30:
                train_loss = 2.0 * math.exp(-epoch / 10) + 0.3
            else:
                train_loss = 0.3 + (epoch - 30) * 0.001  # Slight overfitting
            
            # Validation loss decreases then starts increasing (overfitting)
            if epoch < 25:
                val_loss = 2.1 * math.exp(-epoch / 12) + 0.35
            else:
                val_loss = 0.35 + (epoch - 25) * 0.008  # Overfitting starts
            
            # Accuracy increases then plateaus
            if epoch < 25:
                train_acc = 0.5 + 0.45 * (1 - math.exp(-epoch / 8))
            else:
                train_acc = min(0.95, 0.95 + (epoch - 25) * 0.002)
            
            if epoch < 25:
                val_acc = 0.5 + 0.42 * (1 - math.exp(-epoch / 10))
            else:
                val_acc = max(0.85, 0.92 - (epoch - 25) * 0.003)  # Starts decreasing
            
            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
        
        return history


class ConvergenceAnalyzer:
    """Analyzes convergence and stopping criteria"""
    
    def __init__(self):
        self.patience = 5  # Epochs to wait for improvement
        self.min_delta = 0.001  # Minimum improvement threshold
    
    def analyze_convergence(self, history: List[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze training convergence"""
        
        if len(history) < 2:
            return {"converged": False, "reason": "Insufficient data"}
        
        # Check validation loss trend
        val_losses = [h["val_loss"] for h in history]
        val_accuracies = [h["val_accuracy"] for h in history]
        
        # Find best validation loss
        best_val_loss = min(val_losses)
        best_val_loss_epoch = val_losses.index(best_val_loss) + 1
        
        # Find best validation accuracy
        best_val_acc = max(val_accuracies)
        best_val_acc_epoch = val_accuracies.index(best_val_acc) + 1
        
        current_epoch = len(history)
        epochs_since_improvement = current_epoch - best_val_loss_epoch
        
        # Check for convergence
        convergence_signals = []
        
        # Signal 1: No improvement for patience epochs
        if epochs_since_improvement >= self.patience:
            convergence_signals.append(f"No improvement for {epochs_since_improvement} epochs")
        
        # Signal 2: Validation loss increasing (overfitting)
        if len(val_losses) >= 5:
            recent_trend = sum(val_losses[-3:]) / 3 - sum(val_losses[-6:-3]) / 3
            if recent_trend > 0.01:
                convergence_signals.append("Validation loss increasing (overfitting detected)")
        
        # Signal 3: Training plateaued
        if len(history) >= 10:
            recent_improvement = abs(val_losses[-1] - val_losses[-10])
            if recent_improvement < self.min_delta * 10:
                convergence_signals.append("Training plateaued (minimal improvement)")
        
        # Signal 4: Validation accuracy decreasing
        if len(val_accuracies) >= 5:
            recent_acc_trend = sum(val_accuracies[-3:]) / 3 - sum(val_accuracies[-6:-3]) / 3
            if recent_acc_trend < -0.005:
                convergence_signals.append("Validation accuracy decreasing")
        
        return {
            "converged": len(convergence_signals) > 0,
            "signals": convergence_signals,
            "best_val_loss": best_val_loss,
            "best_val_loss_epoch": best_val_loss_epoch,
            "best_val_accuracy": best_val_acc,
            "best_val_accuracy_epoch": best_val_acc_epoch,
            "epochs_since_improvement": epochs_since_improvement,
            "current_val_loss": val_losses[-1],
            "current_val_accuracy": val_accuracies[-1]
        }


class EarlyStoppingStrategy:
    """Implements early stopping decision logic"""
    
    def __init__(self):
        self.strategies = {
            "validation_loss": self._check_validation_loss,
            "overfitting": self._check_overfitting,
            "plateau": self._check_plateau,
            "resource_limit": self._check_resource_limit
        }
    
    def make_decision(self, history: List[Dict[str, float]], 
                     convergence: Dict[str, Any],
                     max_epochs: int = 100) -> Dict[str, Any]:
        """Make early stopping decision"""
        
        current_epoch = len(history)
        
        reasons = []
        should_stop = False
        
        # Check each strategy
        for strategy_name, strategy_func in self.strategies.items():
            result = strategy_func(history, convergence, max_epochs)
            if result["should_stop"]:
                should_stop = True
                reasons.append(result["reason"])
        
        # Calculate resources saved
        epochs_saved = max_epochs - current_epoch if should_stop else 0
        time_saved_hours = epochs_saved * 0.5  # Assume 30 min per epoch
        compute_cost_saved = epochs_saved * 2.5  # Assume $2.5 per epoch
        
        return {
            "should_stop": should_stop,
            "reasons": reasons,
            "current_epoch": current_epoch,
            "max_epochs": max_epochs,
            "epochs_saved": epochs_saved,
            "time_saved_hours": time_saved_hours,
            "cost_saved": compute_cost_saved,
            "best_checkpoint_epoch": convergence["best_val_loss_epoch"],
            "recommendation": self._generate_recommendation(convergence, should_stop)
        }
    
    def _check_validation_loss(self, history, convergence, max_epochs):
        """Check validation loss criteria"""
        if convergence["epochs_since_improvement"] >= 5:
            return {
                "should_stop": True,
                "reason": f"No validation loss improvement for {convergence['epochs_since_improvement']} epochs"
            }
        return {"should_stop": False, "reason": ""}
    
    def _check_overfitting(self, history, convergence, max_epochs):
        """Check for overfitting"""
        if len(history) >= 5:
            recent_val_losses = [h["val_loss"] for h in history[-5:]]
            if all(recent_val_losses[i] > recent_val_losses[i-1] for i in range(1, len(recent_val_losses))):
                return {
                    "should_stop": True,
                    "reason": "Continuous validation loss increase (overfitting)"
                }
        return {"should_stop": False, "reason": ""}
    
    def _check_plateau(self, history, convergence, max_epochs):
        """Check for plateau"""
        if len(history) >= 10:
            recent_val_losses = [h["val_loss"] for h in history[-10:]]
            improvement = max(recent_val_losses) - min(recent_val_losses)
            if improvement < 0.01:
                return {
                    "should_stop": True,
                    "reason": "Training plateaued (< 0.01 change in 10 epochs)"
                }
        return {"should_stop": False, "reason": ""}
    
    def _check_resource_limit(self, history, convergence, max_epochs):
        """Check resource limits"""
        current_epoch = len(history)
        if current_epoch >= max_epochs * 0.9:
            if convergence["epochs_since_improvement"] >= 3:
                return {
                    "should_stop": True,
                    "reason": "Approaching max epochs with no recent improvement"
                }
        return {"should_stop": False, "reason": ""}
    
    def _generate_recommendation(self, convergence, should_stop):
        """Generate recommendation"""
        if should_stop:
            return f"Stop training and use checkpoint from epoch {convergence['best_val_loss_epoch']} (val_loss: {convergence['best_val_loss']:.4f}, val_acc: {convergence['best_val_accuracy']:.4f})"
        else:
            return "Continue training - model still improving"


def simulate_training_agent(state: EarlyStoppingState) -> EarlyStoppingState:
    """Simulate training process"""
    print("\n🏋️ Simulating Training Process...")
    
    monitor = TrainingMonitor()
    history = monitor.simulate_training(epochs=50)
    
    print(f"\n  Training Epochs: {len(history)}")
    print(f"\n  Initial Performance (Epoch 1):")
    print(f"    • Train Loss: {history[0]['train_loss']:.4f}")
    print(f"    • Val Loss: {history[0]['val_loss']:.4f}")
    print(f"    • Val Accuracy: {history[0]['val_accuracy']:.4f}")
    
    print(f"\n  Current Performance (Epoch {len(history)}):")
    print(f"    • Train Loss: {history[-1]['train_loss']:.4f}")
    print(f"    • Val Loss: {history[-1]['val_loss']:.4f}")
    print(f"    • Val Accuracy: {history[-1]['val_accuracy']:.4f}")
    
    # Show progress at key points
    checkpoints = [10, 20, 30, 40]
    print(f"\n  Training Progress:")
    for cp in checkpoints:
        if cp < len(history):
            h = history[cp - 1]
            print(f"    Epoch {cp}: val_loss={h['val_loss']:.4f}, val_acc={h['val_accuracy']:.4f}")
    
    return {
        **state,
        "training_history": {"history": history},
        "messages": [f"✓ Simulated {len(history)} epochs of training"]
    }


def analyze_convergence_agent(state: EarlyStoppingState) -> EarlyStoppingState:
    """Analyze training convergence"""
    print("\n📊 Analyzing Training Convergence...")
    
    analyzer = ConvergenceAnalyzer()
    history = state["training_history"]["history"]
    convergence = analyzer.analyze_convergence(history)
    
    print(f"\n  Best Validation Performance:")
    print(f"    • Best Val Loss: {convergence['best_val_loss']:.4f} (epoch {convergence['best_val_loss_epoch']})")
    print(f"    • Best Val Accuracy: {convergence['best_val_accuracy']:.4f} (epoch {convergence['best_val_accuracy_epoch']})")
    
    print(f"\n  Current Status:")
    print(f"    • Current Val Loss: {convergence['current_val_loss']:.4f}")
    print(f"    • Current Val Accuracy: {convergence['current_val_accuracy']:.4f}")
    print(f"    • Epochs Since Improvement: {convergence['epochs_since_improvement']}")
    
    if convergence["converged"]:
        print(f"\n  ⚠️  Convergence Signals Detected:")
        for signal in convergence["signals"]:
            print(f"    • {signal}")
    else:
        print(f"\n  ✓ Model still improving")
    
    return {
        **state,
        "convergence_analysis": convergence,
        "messages": [f"✓ Convergence analysis complete"]
    }


def make_stopping_decision_agent(state: EarlyStoppingState) -> EarlyStoppingState:
    """Make early stopping decision"""
    print("\n🛑 Making Early Stopping Decision...")
    
    strategy = EarlyStoppingStrategy()
    history = state["training_history"]["history"]
    convergence = state["convergence_analysis"]
    
    decision = strategy.make_decision(history, convergence, max_epochs=100)
    
    if decision["should_stop"]:
        print(f"\n  🛑 DECISION: STOP TRAINING")
        print(f"\n  Reasons:")
        for reason in decision["reasons"]:
            print(f"    • {reason}")
    else:
        print(f"\n  ✓ DECISION: CONTINUE TRAINING")
    
    print(f"\n  Current Epoch: {decision['current_epoch']}")
    print(f"  Max Epochs: {decision['max_epochs']}")
    
    if decision["should_stop"]:
        print(f"\n  Resources Saved:")
        print(f"    • Epochs Saved: {decision['epochs_saved']}")
        print(f"    • Time Saved: {decision['time_saved_hours']:.1f} hours")
        print(f"    • Cost Saved: ${decision['cost_saved']:.2f}")
    
    print(f"\n  Recommendation:")
    print(f"    {decision['recommendation']}")
    
    return {
        **state,
        "stopping_decision": decision,
        "messages": [f"✓ Stopping decision: {'STOP' if decision['should_stop'] else 'CONTINUE'}"]
    }


def generate_stopping_report_agent(state: EarlyStoppingState) -> EarlyStoppingState:
    """Generate early stopping report"""
    print("\n" + "="*70)
    print("EARLY STOPPING REPORT")
    print("="*70)
    
    history = state["training_history"]["history"]
    convergence = state["convergence_analysis"]
    decision = state["stopping_decision"]
    
    print(f"\n🏋️ Training Summary:")
    print(f"  Total Epochs Completed: {len(history)}")
    print(f"  Max Epochs Configured: {decision['max_epochs']}")
    
    print(f"\n  Initial Performance (Epoch 1):")
    print(f"    • Train Loss: {history[0]['train_loss']:.4f}")
    print(f"    • Train Accuracy: {history[0]['train_accuracy']:.4f}")
    print(f"    • Validation Loss: {history[0]['val_loss']:.4f}")
    print(f"    • Validation Accuracy: {history[0]['val_accuracy']:.4f}")
    
    print(f"\n  Final Performance (Epoch {len(history)}):")
    print(f"    • Train Loss: {history[-1]['train_loss']:.4f}")
    print(f"    • Train Accuracy: {history[-1]['train_accuracy']:.4f}")
    print(f"    • Validation Loss: {history[-1]['val_loss']:.4f}")
    print(f"    • Validation Accuracy: {history[-1]['val_accuracy']:.4f}")
    
    print(f"\n📈 Best Performance Achieved:")
    print(f"  Validation Loss:")
    print(f"    • Best: {convergence['best_val_loss']:.4f}")
    print(f"    • Achieved at Epoch: {convergence['best_val_loss_epoch']}")
    print(f"    • Improvement from Initial: {((history[0]['val_loss'] - convergence['best_val_loss']) / history[0]['val_loss'] * 100):.1f}%")
    
    print(f"\n  Validation Accuracy:")
    print(f"    • Best: {convergence['best_val_accuracy']:.4f}")
    print(f"    • Achieved at Epoch: {convergence['best_val_accuracy_epoch']}")
    print(f"    • Improvement from Initial: {((convergence['best_val_accuracy'] - history[0]['val_accuracy']) / history[0]['val_accuracy'] * 100):.1f}%")
    
    print(f"\n📊 Convergence Analysis:")
    print(f"  Epochs Since Last Improvement: {convergence['epochs_since_improvement']}")
    
    if convergence["converged"]:
        print(f"\n  ⚠️  Convergence Signals:")
        for i, signal in enumerate(convergence["signals"], 1):
            print(f"    {i}. {signal}")
    
    print(f"\n🛑 Early Stopping Decision:")
    if decision["should_stop"]:
        print(f"  Status: TRAINING STOPPED ✋")
        print(f"\n  Stopping Reasons:")
        for i, reason in enumerate(decision["reasons"], 1):
            print(f"    {i}. {reason}")
    else:
        print(f"  Status: CONTINUE TRAINING ▶️")
    
    print(f"\n  Best Checkpoint: Epoch {decision['best_checkpoint_epoch']}")
    print(f"  Recommendation: {decision['recommendation']}")
    
    if decision["should_stop"]:
        print(f"\n💰 Resources Saved by Early Stopping:")
        print(f"  Epochs Saved: {decision['epochs_saved']} ({decision['epochs_saved']/decision['max_epochs']*100:.1f}% of total)")
        print(f"  Training Time Saved: {decision['time_saved_hours']:.1f} hours")
        print(f"  Compute Cost Saved: ${decision['cost_saved']:.2f}")
        
        # Calculate ROI
        total_cost = decision['current_epoch'] * 2.5
        print(f"\n  Total Training Cost: ${total_cost:.2f}")
        print(f"  Cost with Full Training: ${decision['max_epochs'] * 2.5:.2f}")
        print(f"  Savings: {decision['cost_saved']/( decision['max_epochs'] * 2.5)*100:.1f}%")
    
    print(f"\n📉 Performance Trends:")
    
    # Calculate trends
    if len(history) >= 10:
        early_val_loss = sum(h["val_loss"] for h in history[:10]) / 10
        late_val_loss = sum(h["val_loss"] for h in history[-10:]) / 10
        trend = "Decreasing" if late_val_loss < early_val_loss else "Increasing"
        
        print(f"  Validation Loss Trend (last 10 epochs): {trend}")
        print(f"    • Early Average (epochs 1-10): {early_val_loss:.4f}")
        print(f"    • Late Average (last 10 epochs): {late_val_loss:.4f}")
        print(f"    • Change: {((late_val_loss - early_val_loss) / early_val_loss * 100):+.1f}%")
    
    print(f"\n💡 Key Insights:")
    insights = []
    
    if convergence['epochs_since_improvement'] >= 5:
        insights.append("Model has not improved for several epochs")
    
    if decision["should_stop"]:
        insights.append(f"Early stopping saved {decision['epochs_saved']} epochs of training")
    
    if convergence['best_val_loss_epoch'] < len(history) * 0.7:
        insights.append("Best performance achieved early in training")
    
    overfitting_gap = history[-1]['train_loss'] - history[-1]['val_loss']
    if overfitting_gap < -0.1:
        insights.append("Model shows signs of overfitting")
    
    for insight in insights:
        print(f"  • {insight}")
    
    print(f"\n💡 Early Stopping Benefits:")
    print("  • Prevents overfitting")
    print("  • Saves computational resources")
    print("  • Reduces training costs")
    print("  • Optimizes time to deployment")
    print("  • Maintains model generalization")
    print("  • Automated optimization")
    
    print("\n="*70)
    print("✅ Early Stopping Analysis Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_early_stopping_graph():
    workflow = StateGraph(EarlyStoppingState)
    workflow.add_node("simulate_training", simulate_training_agent)
    workflow.add_node("analyze_convergence", analyze_convergence_agent)
    workflow.add_node("make_decision", make_stopping_decision_agent)
    workflow.add_node("generate_report", generate_stopping_report_agent)
    workflow.add_edge(START, "simulate_training")
    workflow.add_edge("simulate_training", "analyze_convergence")
    workflow.add_edge("analyze_convergence", "make_decision")
    workflow.add_edge("make_decision", "generate_report")
    workflow.add_edge("generate_report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 280: Early Stopping MCP Pattern")
    print("="*70)
    
    app = create_early_stopping_graph()
    final_state = app.invoke({
        "messages": [],
        "training_history": {},
        "convergence_analysis": {},
        "stopping_decision": {},
        "optimization_summary": {}
    })
    print("\n✅ Early Stopping Pattern Complete!")


if __name__ == "__main__":
    main()
