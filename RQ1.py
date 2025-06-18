#!/usr/bin/env python3
"""
RQ1: Randomly generate fault node counts within theoretical diagnosability range
"""

import subprocess
import sys
import os

def run_theoretical_diagnosability_comparison():
    """
    Run RQ1 experiment
    """
    # Set working directory to current script directory
    workspace_folder = os.path.dirname(os.path.abspath(__file__))
    
    # Build command arguments
    program = os.path.join(workspace_folder, "theoretical_diagnosability.py")
    args = [
        sys.executable,  
        program,
        "--graph_type", "bc",
        "--n", "4",
        "--num_runs", "5",       
        "--num_graphs", "200",   
        "--epochs", "200",       
        "--num_rounds", "5",     
        "--n_jobs", "2",         
        "--force_regenerate",    
        "--random_fault_mode",   
        "--output_dir", os.path.join(workspace_folder, "results_RQ1")
    ]
    
    print("=" * 60)
    print("RQ1: Randomly generate fault node counts within theoretical diagnosability range")
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
        print("RQ1 experiment completed!")
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
    exit_code = run_theoretical_diagnosability_comparison()
    sys.exit(exit_code) 