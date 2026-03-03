import os
import subprocess
import time
import sys

def run_tests():
    # Get the project root directory assuming run_tests.py is in tests/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    examples_dir = os.path.join(project_root, "examples")
    print(f"Running Interaction Calculus Benchmark Suite...")
    
    # for steps in [10, 100, 200]:
    for steps in [10]:
        log_file = f"test_logs_steps_{steps}.txt"
        print(f"\n--- Benchmarking --steps={steps} (Logging to {log_file}) ---")
        
        with open(log_file, "w") as f:
            f.write(f"=== Benchmarks (Scan Steps: {steps}) ===\n\n")
            
            # 1. Test raw .ic graphs via test_jax.py
            ic_files = [file for file in os.listdir(examples_dir) if file.endswith(".ic")]
            ic_files.sort()
            # ic_files = []
            
            f.write("--- Raw IC Graph Tests (.ic) ---\n")
            for file in ic_files:
                filepath = os.path.join(examples_dir, file)
                msg = f"\n>> Executing: {filepath}\n"
                sys.stdout.write(msg)
                f.write(msg)
                
                start_t = time.time()
                try:
                    test_jax_path = os.path.join(project_root, "tests", "test_jax.py")
                    proc = subprocess.run(
                        ["python3", test_jax_path, filepath, f"--steps={steps}"], 
                        capture_output=True, text=True, timeout=300
                    )
                    elapsed = time.time() - start_t
                    sys.stdout.write(proc.stdout)
                    f.write(proc.stdout)
                    if proc.stderr:
                        sys.stdout.write("\nERRORS:\n" + proc.stderr)
                        f.write("\nERRORS:\n" + proc.stderr)
                        sys.stdout.write(f"FAILED (Time: {elapsed:.2f}s)\n")
                        f.write(f"FAILED (Time: {elapsed:.2f}s)\n")
                    else:
                        sys.stdout.write(f"\nSUCCESS (Time: {elapsed:.2f}s)\n")
                        f.write(f"\nSUCCESS (Time: {elapsed:.2f}s)\n")
                except subprocess.TimeoutExpired:
                    msg = f"FAILED: Timeout\n"
                    sys.stdout.write(msg)
                    f.write(msg)
                except Exception as e:
                    msg = f"FAILED: {e}\n"
                    sys.stdout.write(msg)
                    f.write(msg)
                    
            # 2. Test Lisp algorithms via repl.py
            lisp_files = ["bool.lisp", "math.lisp"] # Explicit order
            msg = "\n\n--- Lisp Compiler Tests (.lisp) ---\n"
            sys.stdout.write(msg)
            f.write(msg)
            
            for file in lisp_files:
                filepath = os.path.join(examples_dir, file)
                msg = f"\n>> Executing: {filepath} (256MB Arrays)\n"
                sys.stdout.write(msg)
                f.write(msg)
                
                start_t = time.time()
                try:
                    repl_path = os.path.join(project_root, "tensoric", "repl.py")
                    proc = subprocess.run(
                        ["python3", repl_path, filepath, f"--steps={steps}"], 
                        capture_output=True, text=True, timeout=300
                    )
                    elapsed = time.time() - start_t
                    sys.stdout.write(proc.stdout)
                    f.write(proc.stdout)
                    if proc.stderr:
                        sys.stdout.write("\nERRORS:\n" + proc.stderr)
                        f.write("\nERRORS:\n" + proc.stderr)
                        sys.stdout.write(f"FAILED (Time: {elapsed:.2f}s)\n")
                        f.write(f"FAILED (Time: {elapsed:.2f}s)\n")
                    else:
                        sys.stdout.write(f"\nSUCCESS (Time: {elapsed:.2f}s)\n")
                        f.write(f"\nSUCCESS (Time: {elapsed:.2f}s)\n")
                except subprocess.TimeoutExpired:
                    msg = f"FAILED: Timeout (300s) exceeded.\n"
                    sys.stdout.write(msg)
                    f.write(msg)
                except Exception as e:
                    msg = f"FAILED: {e}\n"
                    sys.stdout.write(msg)
                    f.write(msg)

if __name__ == "__main__":
    run_tests()
