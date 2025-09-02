import math
import random

def ternary_search_max(func, left, right, epsilon=1e-9):
    """
    Ternary search to find maximum of unimodal function.
    
    Args:
        func: Function to optimize (should have single peak)
        left, right: Search bounds
        epsilon: Precision threshold
    
    Returns:
        x value where function reaches maximum
    """
    while right - left > epsilon:
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3
        
        if func(m1) < func(m2):
            left = m1  # Maximum is in right 2/3
        else:
            right = m2  # Maximum is in left 2/3
    
    return (left + right) / 2


def ternary_search_min(func, left, right, epsilon=1e-9):
    """

    """
    while right - left > epsilon:
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3
        
        if func(m1) > func(m2):
            left = m1
        else:
            right = m2
    
    return (left + right) / 2



def golden_section_search(func, left, right, epsilon=1e-9):
    """
    Golden section search - optimal for unimodal functions.
    Uses golden ratio to minimize function evaluations.
    """
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio
    
    x1 = right - (right - left) / phi
    x2 = left + (right - left) / phi
    f1, f2 = func(x1), func(x2)
    
    while abs(right - left) > epsilon:
        if f1 > f2:  # Maximum in [left, x2]
            right, x2, f2 = x2, x1, f1
            x1 = right - (right - left) / phi
            f1 = func(x1)
        else:  # Maximum in [x1, right]
            left, x1, f1 = x1, x2, f2
            x2 = left + (right - left) / phi
            f2 = func(x2)
    
    return (left + right) / 2


def fibonacci_search(func, left, right, n=20):
    """
    Fibonacci search - theoretically optimal number of evaluations.
    """
    fib = [1, 1]
    for i in range(2, n + 2):
        fib.append(fib[i-1] + fib[i-2])
    
    k = n
    x1 = left + (fib[k-2] / fib[k]) * (right - left)
    x2 = left + (fib[k-1] / fib[k]) * (right - left)
    f1, f2 = func(x1), func(x2)
    
    for i in range(k-1, 0, -1):
        if f1 > f2:
            right, x2, f2 = x2, x1, f1
            x1 = left + (fib[i-2] / fib[i]) * (right - left)
            f1 = func(x1) if i > 1 else f1
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + (fib[i-1] / fib[i]) * (right - left)
            f2 = func(x2) if i > 1 else f2
    
    return (left + right) / 2

def random_search(func, bounds, n_samples=100):
    """
    Random search - simple but effective for multimodal functions.
    """
    best_x, best_val = None, float('-inf')
    
    for _ in range(n_samples):
        x = random.uniform(bounds[0], bounds[1])
        val = func(x)
        if val > best_val:
            best_x, best_val = x, val
    
    return best_x

def adaptive_random_search(func, bounds, n_samples=100, shrink_factor=0.8):
    """
    Adaptive random search - focuses sampling around best points.
    """
    best_x = random.uniform(bounds[0], bounds[1])
    best_val = func(best_x)
    search_radius = (bounds[1] - bounds[0]) / 4
    
    for _ in range(n_samples - 1):
        x = best_x + random.uniform(-search_radius, search_radius)
        x = max(bounds[0], min(bounds[1], x))  # Clamp to bounds
        
        val = func(x)
        if val > best_val:
            best_x, best_val = x, val
            search_radius *= shrink_factor  # Focus search
    
    return best_x
