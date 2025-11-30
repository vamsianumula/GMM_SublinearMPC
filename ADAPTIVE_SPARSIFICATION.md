# Adaptive Sparsification & Peak Hold Estimator

## 1. The Problem: Sublinearity Violation
In dense graphs, the standard GMM algorithm with fixed sparsification ($p=0.5$) violates the sublinear communication limit ($O(S)$ per machine).
*   **Cause**: Too many edges participate in the "Exponentiation" phase, where ball sizes grow exponentially.
*   **Symptom**: Communication volume exceeds the theoretical limit by orders of magnitude (e.g., 500x).

## 2. The Solution: Adaptive Sparsification
We implemented a dynamic throttling mechanism that adjusts $p$ based on the estimated system load.

### Formula
$$ p_{phase} = \min\left(0.5, \frac{\text{Safety} \cdot (P \cdot S)}{\text{Active Edges} \cdot \text{EstBallSize}} \right) $$

### The "Bouncing" Issue
Initially, we estimated `EstBallSize` using the *previous phase's* maximum ball size.
*   **Flaw**: If the algorithm processed a batch of small components (small balls) in Phase $i$, the estimator predicted low load for Phase $i+1$.
*   **Result**: $p$ was reset to 0.5. If Phase $i+1$ hit a giant component, the load spiked, causing the communication volume to "bounce" above the limit.

### The Fix: Peak Hold Estimator
We switched to a **Peak Hold** strategy:
$$ \text{EstBallSize}_{i+1} = \max(\text{EstBallSize}_{i}, \text{CurrentMaxBall}) \times 2 $$
*   **Logic**: The algorithm remembers the largest ball seen so far. It assumes that if we saw a large ball before, we might see it again (or larger) until the graph is fully processed.
*   **Safety Factor**: We set `safety_factor = 0.5` (targeting 50% capacity) to provide a robust buffer against estimation variance.

## 3. Evidence of Success

### A. Communication Volume (Strictly Sublinear)
With Peak Hold + Safety Factor 0.5, the communication volume (blue dots) stays **strictly below the green reference line** ($O(S)$). The "bouncing" is eliminated.

![Communication Volume](docs/images/comm_sublinearity_items.png)

### B. Adaptive Probability (Stable Throttling)
The probability $p$ drops to $\approx 0.005$ and **stays low**. It does not oscillate back to 0.5, proving that the estimator correctly maintains a conservative view of the potential load.

![Adaptive Probability](docs/images/sparsification_p.png)

### C. Convergence
Convergence takes longer (~30 phases vs 12) because we are processing the graph in safe, manageable chunks. This is the necessary trade-off for correctness.

![Convergence](docs/images/convergence.png)
