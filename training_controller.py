"""
Training Program Master Controller - Orchestrates the complete training pipeline.

This script manages the entire training workflow:
1. Deep topology analysis
2. Scenario generation  
3. Test simulation
4. Feedback loop and improvements
5. Dashboard visualization
"""

import sys
import json
import os
from pathlib import Path
import subprocess


class TrainingController:
    """Master controller for the training pipeline."""
    
    def __init__(self):
        """Initialize the training controller."""
        self.steps = [
            {
                "id": 1,
                "name": "Deep Topology Analysis",
                "script": "topology_analyzer.py",
                "input": "Projects/Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf",
                "output": "enhanced_topology.json",
                "description": "Comprehensive extraction of structural elements using GPT-4 Vision and text analysis"
            },
            {
                "id": 2,
                "name": "Scenario Generation",
                "script": "scenario_generator.py",
                "input": "enhanced_topology.json",
                "output": "test_scenarios.json",
                "description": "Generate comprehensive search scenarios covering 90% of knowledge base"
            },
            {
                "id": 3,
                "name": "Test Simulation",
                "script": "test_simulator.py",
                "input": "test_scenarios.json",
                "output": "test_results_*.json",
                "description": "Execute scenarios, validate results, generate feedback"
            },
            {
                "id": 4,
                "name": "Dashboard Review",
                "script": "training_dashboard.html",
                "input": "test_results_*.json",
                "output": "Visual dashboard",
                "description": "Review training results and feedback in web interface"
            }
        ]
        
        self.current_step = 0
    
    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        print("\n" + "="*80)
        print("RAG TRAINING PROGRAM - MASTER CONTROLLER")
        print("="*80)
        
        print("\n📋 Training Pipeline Overview:")
        for step in self.steps:
            print(f"\n  Step {step['id']}: {step['name']}")
            print(f"    📄 Input: {step['input']}")
            print(f"    📤 Output: {step['output']}")
            print(f"    ℹ️  {step['description']}")
        
        print("\n" + "="*80)
        print("\n⚠️  HUMAN-IN-THE-LOOP: Full Pipeline Execution")
        print("\nThis will run all 4 steps sequentially with checkpoints.")
        print("Each step will require your confirmation before proceeding.")
        print("\nEstimated time: 15-30 minutes")
        print("Estimated cost: $3-7 (OpenAI API)")
        print("\nContinue with full pipeline? (yes/no): ", end="")
        
        response = input().strip().lower()
        if response != 'yes':
            print("\n❌ Pipeline cancelled. You can run steps individually instead.")
            self.show_individual_commands()
            return
        
        # Run each step
        for step in self.steps:
            self._run_step(step)
    
    def _run_step(self, step: dict):
        """Run a single pipeline step."""
        print("\n" + "="*80)
        print(f"STEP {step['id']}: {step['name'].upper()}")
        print("="*80)
        
        # Check if input exists (except for first step)
        if step['id'] > 1:
            input_path = Path(step['input'])
            if not input_path.exists() and not input_path.match('*.json'):
                print(f"\n❌ Input file not found: {step['input']}")
                print("   Previous step may not have completed successfully.")
                return False
        
        # Execute step
        if step['script'].endswith('.html'):
            print(f"\n📊 Opening dashboard: {step['script']}")
            print("\n💡 Load the latest test_results_*.json file to view results")
            subprocess.run(['start', step['script']], shell=True)
        else:
            print(f"\n▶️  Executing: python {step['script']}")
            
            if step['id'] == 1:
                # Topology analyzer needs filename argument - use sys.executable for correct Python path
                cmd = f'"{sys.executable}" {step["script"]} "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf"'
                exit_code = os.system(cmd)
                if exit_code != 0:
                    print(f"\n❌ Step {step['id']} failed with error code {exit_code}")
                    return False
            else:
                # Use sys.executable for all scripts to support interactive input
                cmd = f'"{sys.executable}" {step["script"]}'
                exit_code = os.system(cmd)
                if exit_code != 0:
                    print(f"\n❌ Step {step['id']} failed with error code {exit_code}")
                    return False
        
        print(f"\n✓ Step {step['id']} completed successfully")
        return True
    
    def show_individual_commands(self):
        """Show commands to run steps individually."""
        print("\n" + "="*80)
        print("INDIVIDUAL STEP COMMANDS")
        print("="*80)
        
        print("\nRun each step separately:\n")
        
        print('1. Deep Topology Analysis:')
        print('   python topology_analyzer.py "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf"')
        
        print('\n2. Generate Scenarios:')
        print('   python scenario_generator.py')
        
        print('\n3. Run Simulation:')
        print('   python test_simulator.py')
        
        print('\n4. View Dashboard:')
        print('   start training_dashboard.html')
        
        print("\n" + "="*80)
    
    def show_menu(self):
        """Show interactive menu."""
        while True:
            print("\n" + "="*80)
            print("RAG TRAINING PROGRAM - MENU")
            print("="*80)
            
            print("\nOptions:")
            print("  1. Run full training pipeline")
            print("  2. Run individual step")
            print("  3. Show step commands")
            print("  4. Check status")
            print("  5. Exit")
            
            print("\nSelect option (1-5): ", end="")
            choice = input().strip()
            
            if choice == '1':
                self.run_full_pipeline()
            elif choice == '2':
                self.run_individual_step()
            elif choice == '3':
                self.show_individual_commands()
            elif choice == '4':
                self.check_status()
            elif choice == '5':
                print("\n👋 Exiting training program")
                break
            else:
                print("\n❌ Invalid choice. Please select 1-5")
    
    def run_individual_step(self):
        """Run a single step."""
        print("\n" + "="*80)
        print("SELECT STEP TO RUN")
        print("="*80)
        
        for step in self.steps:
            print(f"\n  {step['id']}. {step['name']}")
            print(f"     {step['description']}")
        
        print("\nSelect step (1-4): ", end="")
        choice = input().strip()
        
        if choice.isdigit() and 1 <= int(choice) <= 4:
            step = self.steps[int(choice) - 1]
            self._run_step(step)
        else:
            print("\n❌ Invalid step number")
    
    def check_status(self):
        """Check which steps have been completed."""
        print("\n" + "="*80)
        print("TRAINING PIPELINE STATUS")
        print("="*80)
        
        for step in self.steps:
            output_path = Path(step['output'].replace('*', ''))
            
            if step['output'].endswith('*.json'):
                # Check for any matching files
                matching_files = list(Path('.').glob(step['output']))
                exists = len(matching_files) > 0
                status = f"✅ Complete ({len(matching_files)} files)" if exists else "⏳ Pending"
            elif step['script'].endswith('.html'):
                exists = Path(step['script']).exists()
                status = "✅ Available" if exists else "❌ Missing"
            else:
                exists = output_path.exists() or Path(step['output']).exists()
                status = "✅ Complete" if exists else "⏳ Pending"
            
            print(f"\nStep {step['id']}: {step['name']}")
            print(f"  Status: {status}")
            print(f"  Output: {step['output']}")


def main():
    """Main entry point."""
    controller = TrainingController()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'full':
            controller.run_full_pipeline()
        elif command == 'status':
            controller.check_status()
        elif command == 'commands':
            controller.show_individual_commands()
        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python training_controller.py           # Interactive menu")
            print("  python training_controller.py full      # Run full pipeline")
            print("  python training_controller.py status    # Check status")
            print("  python training_controller.py commands  # Show individual commands")
    else:
        controller.show_menu()


if __name__ == "__main__":
    main()
