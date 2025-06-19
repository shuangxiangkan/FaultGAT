#!/usr/bin/env python3
"""
RQ3: Fault diagnosis capability analysis under partial symptoms
Testing symptom missing methods:
- node_disable: Disable a certain proportion of nodes, do not generate symptoms for these nodes as tester and testee
"""

import subprocess
import sys
import os

def run_partial_symptom_comparison():
    """
    Run fault diagnosis experiment under partial symptoms
    """
    # Set working directory to current script directory
    workspace_folder = os.path.dirname(os.path.abspath(__file__))
    
    # Build command arguments
    program = os.path.join(workspace_folder, "partial_symptom_comparison.py")
    args = [
        sys.executable,  # Use current Python interpreter
        program,
        "--graph_type", "bc",
        "--n", "7",
        "--max_missing_ratio", "0.5",   
        "--ratio_step", "0.05",         
        "--num_runs", "10",             
        "--num_graphs", "300",         
        "--epochs", "300",             
        "--n_jobs", "4",               
        "--force_regenerate",          
        "--missing_type", "node_disable",  
        "--output_dir", os.path.join(workspace_folder, "results_RQ3")
    ]
    
    print("=" * 60)
    print("🚀 Starting RQ3 experiment: Fault diagnosis capability analysis under partial symptoms")
    print("=" * 60)
    print(f"Executing command: {' '.join(args)}")
    print("=" * 60)
    
    try:
        # Execute command
        result = subprocess.run(
            args,
            cwd=workspace_folder,
            capture_output=False,
            text=True,
            check=True
        )
        
        print("\n" + "=" * 60)
        print("RQ3 experiment completed!")
        print("=" * 60)
        
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ RQ3 experiment execution failed: {e}")
        print(f"Error code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⏹️ User interrupted the experiment")
        return 130
    except Exception as e:
        print(f"\n💥 Exception occurred during execution: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_partial_symptom_comparison()
    sys.exit(exit_code) 