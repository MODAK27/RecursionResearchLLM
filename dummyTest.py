import os
import sys
import numpy as np

def gcd(a: int, b: int) -> int:
    a, b = abs(a), abs(b)
    while b != 0:
        a, b = b, a % b
    return a

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <a> <b>")
        sys.exit(1)
    try:
        a = int(sys.argv[1])
        b = int(sys.argv[2])
    except ValueError:
        print("Error: Both a and b must be integers.")
        sys.exit(1)

    result = gcd(a, b)
    answer = f"GCD of {a} and {b} is {result}"
    print(answer)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "output.txt")
    np.savetxt(output_path, [answer], fmt="%s")

if __name__ == "__main__":
    main()
