#!/usr/bin/env python3
"""
RQ2: Fault ratio comparison analysis
"""

import subprocess
import sys
import os

def run_fault_ratio_comparison():
    """
    Run fault ratio comparison experiment
    """
    # Set working directory to current script directory
    workspace_folder = os.path.dirname(os.path.abspath(__file__))
    
    # Build command arguments
    program = os.path.join(workspace_folder, "fault_ratio_comparison.py")
    args = [
        sys.executable,  # Use current Python interpreter
        program,
        "--graph_type", "bc",
        "--n", "7",
        "--max_fault_ratio", "0.5",  
        "--ratio_step", "0.01",      
        "--num_runs", "10", 
        "--num_graphs", "300",
        "--epochs", "300",
        "--n_jobs", "4",
        "--force_regenerate",
        "--output_dir", os.path.join(workspace_folder, "results_RQ2")
    ]
    
    print("=" * 60)
    print("🚀 Starting RQ2 experiment: Fault ratio comparison analysis")
    print("=" * 60)
    print(f"Executing command: {' '.join(args)}")
    print("=" * 60)
    
    try:
        # Execute command
        result = subprocess.run(
            args,
            cwd=workspace_folder,
            check=True,
            text=True,
            env={**os.environ, "PYTHONPATH": workspace_folder}
        )
        
        print("\n" + "=" * 60)
        print("RQ2 experiment completed!")
        print("=" * 60)
        
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"\nError: Command execution failed, return code: {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print(f"\nError: File not found {program}")
        return 1
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = run_fault_ratio_comparison()
    sys.exit(exit_code) 