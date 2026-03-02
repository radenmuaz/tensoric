import os
import subprocess
import time

def run_tests():
    examples_dir = "examples"
    log_file = "test_logs.txt"
    
    print(f"Running Interaction Calculus Test Suite...")
    print(f"Logging output to {log_file}\n")
    
    with open(log_file, "w") as f:
        f.write("=== Interaction Calculus Execution Logs ===\n\n")
        
        # 1. Test raw .ic graphs via test_jax.py
        ic_files = [file for file in os.listdir(examples_dir) if file.endswith(".ic")]
        ic_files.sort()
        
        f.write("--- Raw IC Graph Tests (.ic) ---\n")
        print("Running .ic graph examples...")
        
        for file in ic_files:
            filepath = os.path.join(examples_dir, file)
            f.write(f"\n>> Executing: {filepath}\n")
            
            start_t = time.time()
            try:
                proc = subprocess.run(
                    ["python3", "test_jax.py", filepath], 
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
            except Exception as e:
                f.write(f"FAILED: {e}\n")
                print(f"  ✗ {file} (Failed)")
                
        # 2. Test Lisp algorithms via repl.py
        lisp_files = ["bool.lisp", "math.lisp"] # Explicit order
        
        f.write("\n\n--- Lisp Compiler Tests (.lisp) ---\n")
        print("\nRunning Lisp algorithm examples...")
        
        for file in lisp_files:
            filepath = os.path.join(examples_dir, file)
            f.write(f"\n>> Executing: {filepath} (256MB Arrays)\n")
            
            start_t = time.time()
            try:
                proc = subprocess.run(
                    ["python3", "repl.py", filepath], 
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
                print(f"  ✗ {file} (Error)")

    print(f"\nTest run complete. Check {log_file} for detailed reduction steps.")

if __name__ == "__main__":
    run_tests()
