def gradient_descent(
    func,            # The function to minimize: takes x, returns scalar
    grad_func,       # Its gradient function: takes x, returns gradient (same shape as x)
    initial_x,       # Starting point (float or list/array)
    learning_rate=0.01,
    max_iters=1000,
    tolerance=1e-6
):
    """
    Performs gradient descent to minimize func.

    Args:
        func: Function to minimize.
        grad_func: Gradient of func.
        initial_x: Starting value(s) of x.
        learning_rate: Step size.
        max_iters: Max iterations.
        tolerance: Stop if gradient norm is below this.

    Returns:
        x: Optimized value.
        history: List of function values over iterations.
    """

    x = initial_x
    history = []

    for i in range(max_iters):
        grad = grad_func(x)

        # If gradient is a list/tuple, convert to vector norm
        if isinstance(grad, (list, tuple)):
            grad_norm = sum(g**2 for g in grad) ** 0.5
        else:
            grad_norm = abs(grad)

        # Stop if gradient is very small (close to local minimum)
        if grad_norm < tolerance:
            print(f"Converged after {i} iterations.")
            break

        # Take a step opposite to gradient
        if isinstance(x, (list, tuple)):
            x = [xi - learning_rate * gi for xi, gi in zip(x, grad)]
        else:
            x = x - learning_rate * grad

        current_value = func(x)
        history.append(current_value)

        # Optionally print progress
        if i % 100 == 0:
            print(f"Iter {i}: f(x) = {current_value}, grad norm = {grad_norm}")

    return x, history

# Example: minimize f(x) = (x-3)^2
def f(x):
    return (x - 3) ** 2

def grad_f(x):
    return 2 * (x - 3)

if __name__ == "__main__":
    initial_guess = 0.0
    optimal_x, vals = gradient_descent(f, grad_f, initial_guess, learning_rate=0.1)
    print(f"Optimal x: {optimal_x}")
