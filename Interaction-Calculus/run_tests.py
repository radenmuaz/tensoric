import os
import subprocess
import time

def run_tests():
    examples_dir = "examples"
    print(f"Running Interaction Calculus Benchmark Suite...")
    
    for steps in [10, 100, 1000, 10000]:
        log_file = f"test_logs_steps_{steps}.txt"
        print(f"\n--- Benchmarking --steps={steps} (Logging to {log_file}) ---")
        
        with open(log_file, "w") as f:
            f.write(f"=== Benchmarks (Scan Steps: {steps}) ===\n\n")
            
            # 1. Test raw .ic graphs via test_jax.py
            ic_files = [file for file in os.listdir(examples_dir) if file.endswith(".ic")]
            ic_files.sort()
            
            f.write("--- Raw IC Graph Tests (.ic) ---\n")
            for file in ic_files:
                filepath = os.path.join(examples_dir, file)
                f.write(f"\n>> Executing: {filepath}\n")
                
                start_t = time.time()
                try:
                    proc = subprocess.run(
                        ["python3", "test_jax.py", filepath, f"--steps={steps}"], 
                        capture_output=True, text=True, timeout=300
                    )
                    elapsed = time.time() - start_t
                    f.write(proc.stdout)
                    if proc.stderr:
                        f.write("\nERRORS:\n" + proc.stderr)
                        f.write(f"FAILED (Time: {elapsed:.2f}s)\n")
                        print(f"  ✗ {file} (Failed)")
                    else:
                        f.write(f"\nSUCCESS (Time: {elapsed:.2f}s)\n")
                        print(f"  ✓ {file} ({elapsed:.2f}s)")
                except subprocess.TimeoutExpired:
                    f.write(f"FAILED: Timeout\n")
                    print(f"  ✗ {file} (Timeout)")
                except Exception as e:
                    f.write(f"FAILED: {e}\n")
                    
            # 2. Test Lisp algorithms via repl.py
            lisp_files = ["bool.lisp", "math.lisp"] # Explicit order
            f.write("\n\n--- Lisp Compiler Tests (.lisp) ---\n")
            
            for file in lisp_files:
                filepath = os.path.join(examples_dir, file)
                f.write(f"\n>> Executing: {filepath} (256MB Arrays)\n")
                
                start_t = time.time()
                try:
                    proc = subprocess.run(
                        ["python3", "repl.py", filepath, f"--steps={steps}"], 
                        capture_output=True, text=True, timeout=300
                    )
                    elapsed = time.time() - start_t
                    f.write(proc.stdout)
                    if proc.stderr:
                        f.write("\nERRORS:\n" + proc.stderr)
                        f.write(f"FAILED (Time: {elapsed:.2f}s)\n")
                        print(f"  ✗ {file} (Failed)")
                    else:
                        f.write(f"\nSUCCESS (Time: {elapsed:.2f}s)\n")
                        print(f"  ✓ {file} ({elapsed:.2f}s)")
                except subprocess.TimeoutExpired:
                    f.write(f"FAILED: Timeout (300s) exceeded.\n")
                    print(f"  ✗ {file} (Timeout)")
                except Exception as e:
                    f.write(f"FAILED: {e}\n")

if __name__ == "__main__":
    run_tests()
