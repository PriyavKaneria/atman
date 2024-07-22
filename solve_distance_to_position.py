import sympy as sp

def find_new_coordinates(xA, yA, s1_prime, xB, yB, s2_prime):
    # Define the symbols
    x, y = sp.symbols('x y')
    
    # Define the equations of the circles
    eq1 = (x - xA)**2 + (y - yA)**2 - s1_prime**2
    eq2 = (x - xB)**2 + (y - yB)**2 - s2_prime**2
    
    # Solve the system of equations
    solutions = sp.solve((eq1, eq2), (x, y))
    
    # Convert solutions to numerical values if they exist
    numerical_solutions = []
    for sol in solutions:
        x_val, y_val = sol
        numerical_solutions.append((float(x_val.evalf()), float(y_val.evalf())))
    
    return numerical_solutions