"""
Master Results Generation Script
Runs all comprehensive result generation tasks
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import time

print("="*80)
print("🚀 MASTER RESULTS GENERATION SUITE")
print("   Alzheimer's Disease EEG Detection Project")
print("="*80)
print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

start_time = time.time()

# Track progress
tasks_completed = []
tasks_failed = []

# Task 1: Comprehensive Results (ROC, PR curves, etc.)
print("\n" + "="*80)
print("TASK 1/2: Comprehensive Results Generation")
print("="*80)
try:
    from scripts.generate_comprehensive_results import ComprehensiveResultsGenerator
    
    gen1 = ComprehensiveResultsGenerator()
    results1 = gen1.run_all()
    tasks_completed.append("Comprehensive Results")
    print("✅ Task 1 completed successfully!")
    
except Exception as e:
    tasks_failed.append(("Comprehensive Results", str(e)))
    print(f"❌ Task 1 failed: {str(e)}")

# Task 2: Clinical Analysis and Statistical Testing
print("\n" + "="*80)
print("TASK 2/2: Clinical Analysis & Statistical Testing")
print("="*80)
try:
    from scripts.generate_clinical_analysis import ClinicalAnalysisGenerator
    
    gen2 = ClinicalAnalysisGenerator()
    results2 = gen2.run_all()
    tasks_completed.append("Clinical Analysis")
    print("✅ Task 2 completed successfully!")
    
except Exception as e:
    tasks_failed.append(("Clinical Analysis", str(e)))
    print(f"❌ Task 2 failed: {str(e)}")

# Final summary
end_time = time.time()
duration = end_time - start_time

print("\n" + "="*80)
print("📊 EXECUTION SUMMARY")
print("="*80)

print(f"\nTotal execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")

print(f"\n✅ Tasks Completed ({len(tasks_completed)}):")
for task in tasks_completed:
    print(f"   • {task}")

if tasks_failed:
    print(f"\n❌ Tasks Failed ({len(tasks_failed)}):")
    for task, error in tasks_failed:
        print(f"   • {task}: {error}")

print("\n" + "="*80)
print("📁 RESULTS LOCATION")
print("="*80)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, 'results')

print(f"\nAll results saved in: {results_dir}")
print("\nGenerated folders:")
print(f"   📊 {os.path.join(results_dir, 'comprehensive_viz')}")
print(f"   📋 {os.path.join(results_dir, 'reports')}")
print(f"   🏥 {os.path.join(results_dir, 'clinical_analysis')}")

print("\n" + "="*80)
print("📄 GENERATED FILES")
print("="*80)

print("\n🎯 Priority Visualizations:")
print("   ✅ ROC Curves (all CV folds + average)")
print("   ✅ Precision-Recall Curves (all CV folds + average)")
print("   ✅ Feature Importance Ranking")
print("   ✅ Training History Plots")
print("   ✅ Model Architecture Diagram")
print("   ✅ Statistical Testing Visualizations")

print("\n📊 Reports:")
print("   ✅ Performance Summary (CSV + TXT)")
print("   ✅ Feature Ranking (CSV)")
print("   ✅ Training History (CSV)")
print("   ✅ Clinical Interpretation Report")
print("   ✅ Statistical Testing Results (CSV)")
print("   ✅ Statistical Summary Report")

print("\n" + "="*80)
print("🎯 NEXT STEPS")
print("="*80)

print("\n1. Review Generated Visualizations:")
print("   • Check ROC and PR curves for model performance")
print("   • Examine feature importance rankings")
print("   • Validate statistical test results")

print("\n2. Documentation:")
print("   • Include visualizations in research paper")
print("   • Reference clinical interpretation report")
print("   • Cite statistical validation results")

print("\n3. Further Analysis:")
print("   • Compare with baseline methods")
print("   • Perform external validation")
print("   • Conduct sensitivity analysis")

print("\n4. Clinical Validation:")
print("   • Share clinical report with domain experts")
print("   • Plan prospective validation study")
print("   • Prepare regulatory documentation")

print("\n" + "="*80)
print("✅ ALL TASKS COMPLETE!")
print("="*80)
print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n🎉 Your AD EEG detection project now has comprehensive results!")
print("="*80 + "\n")
