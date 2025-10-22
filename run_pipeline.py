import subprocess
import sys

def run_pipeline():
    """Run the complete ML pipeline"""
    
    print("="*50)
    print("HOUSE PRICE PREDICTION PIPELINE")
    print("="*50)
    
    steps = [
        ("Data Preprocessing", "data_preprocessing.py"),
        ("Data Visualization", "data_visualization.py"),
        ("Model Training", "model_training.py"),
        ("Model Evaluation", "model_evaluation.py")
    ]
    
    for step_name, script in steps:
        print(f"\n🔄 Running: {step_name}...")
        result = subprocess.run([sys.executable, script], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {step_name} completed!")
            print(result.stdout)
        else:
            print(f"❌ {step_name} failed!")
            print(result.stderr)
            return
    
    print("\n" + "="*50)
    print("✅ PIPELINE COMPLETE!")
    print("="*50)
    print("\n📱 Run the app with: streamlit run app.py")

if __name__ == "__main__":
    run_pipeline()
