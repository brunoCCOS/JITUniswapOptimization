import math
import random

def ternary_search_max(func, low, high, epsilon=1e-6):
    a, b = low, high
    # Evaluate at boundaries initially
    f_low = func(low)
    f_high = func(high)
    while b - a > epsilon:
        mid1 = a + (b - a)/3
        mid2 = b - (b - a)/3
        f1 = func(mid1)
        f2 = func(mid2)
        if f1 > f2:
            b = mid2
        else:
            a = mid1
    # After loop, evaluate at final a, b, and mid
    mid = (a + b) / 2
    f_mid = func(mid)
    f_a = func(a)
    f_b = func(b)
    # Find the maximum among final a, b, mid
    candidates = [(f_low, low), (f_high, high), (f_a, a), (f_b, b), (f_mid, mid)]
    best_f, best_x = max(candidates, key=lambda x: x[0])
    return best_f, best_x



def golden_section_search(func, low, high, epsilon=1e-6):
    a, b = low, high
    # Evaluate at boundaries initially
    f_low = func(low)
    f_high = func(high)
    phi = (1 + math.sqrt(5)) / 2
    while b - a > epsilon:
        mid1 = b - (b - a) / phi
        mid2 = a + (b - a) / phi
        f1 = func(mid1)
        f2 = func(mid2)
        if f1 > f2:
            a = mid1
        else:
            b = mid2
    # After loop, evaluate at final a, b, and mid
    mid = (a + b) / 2
    f_mid = func(mid)
    f_a = func(a)
    f_b = func(b)
    # Find the maximum among final a, b, mid
    candidates = [(f_low, low), (f_high, high), (f_a, a), (f_b, b), (f_mid, mid)]
    best_f, best_x = max(candidates, key=lambda x: x[0])
    return best_f, best_x


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
    
    return func((left + right) / 2),(left + right) / 2

def random_search(func, low, high, n_samples=100):
    """
    Random search - simple but effective for multimodal functions.
    """
    best_x, best_val = None, float('-inf')
    
    for _ in range(n_samples):
        x = random.uniform(low, high)
        val = func(x)
        if val > best_val:
            best_x, best_val = x, val
    
    return best_x

def adaptive_random_search(func, low, high, n_samples=100, shrink_factor=0.8):
    """
    Adaptive random search - focuses sampling around best points.
    """
    best_x = random.uniform(low, high)
    best_val = func(best_x)
    search_radius = (high - low) / 4
    
    for _ in range(n_samples - 1):
        x = best_x + random.uniform(-search_radius, search_radius)
        x = max(low, min(high, x))  # Clamp to bounds
        
        val = func(x)
        if val > best_val:
            best_x, best_val = x, val
            search_radius *= shrink_factor  # Focus search
    
    return best_val, best_x
