# 📈 Learning Curve Analysis Guide

## Purpose

Analyze your model's training behavior to detect **overfitting**, **underfitting**, or **good fit**.

---

## 🚀 Quick Start

### Option 1: Quick Analysis (30 epochs, ~10-15 minutes)

```powershell
& 'C:/Users/tahle/OneDrive/Documents/SRM/AD project/venv/Scripts/python.exe' scripts/quick_learning_curves.py
```

### Option 2: Full Analysis (50 epochs, ~20-30 minutes)

```powershell
& 'C:/Users/tahle/OneDrive/Documents/SRM/AD project/venv/Scripts/python.exe' scripts/plot_learning_curves.py
```

---

## 📊 What the Analysis Shows

### Four Key Visualizations:

1. **Loss Curves** (Training vs Validation)

   - Shows how loss decreases over epochs
   - Gap indicates overfitting if large

2. **Accuracy Curves** (Training vs Validation)

   - Shows model performance improvement
   - Similar curves = good generalization

3. **Overfitting Indicator**

   - Red zone = Overfitting
   - Green zone = Good fit
   - Tracks the gap between train and validation

4. **Diagnostic Summary**
   - Automated analysis
   - Status determination
   - Recommendations

---

## 🔍 Interpreting Results

### ✅ Good Fit (What You Want)

```
Indicators:
• Training accuracy: 75-85%
• Validation accuracy: 70-80%
• Small gap (<10%) between train and val
• Both curves converge and stabilize

What it means:
✓ Model generalizes well
✓ Not memorizing training data
✓ Ready for deployment
```

### ⚠️ Overfitting (Too Specific)

```
Indicators:
• Training accuracy: 90-100%
• Validation accuracy: 60-70%
• Large gap (>15%) between train and val
• Val curve plateaus or increases while train keeps improving

What it means:
✗ Model memorizing training data
✗ Won't generalize to new data
✗ Needs regularization

Solutions:
1. Increase dropout (0.5 → 0.6)
2. Add L2 regularization
3. Reduce model complexity
4. Get more training data
5. Use data augmentation
6. Stop training earlier
```

### ❌ Underfitting (Too Simple)

```
Indicators:
• Training accuracy: <70%
• Validation accuracy: <65%
• Both curves are low
• Model hasn't learned enough

What it means:
✗ Model too simple
✗ Not capturing patterns
✗ Needs more capacity

Solutions:
1. Increase model size
2. Train for more epochs
3. Reduce regularization
4. Improve features
5. Check learning rate
```

---

## 📈 Example Outputs

### Good Fit Example:

```
Final Metrics:
  Train Accuracy: 0.78 (78%)
  Val Accuracy:   0.75 (75%)
  Accuracy Gap:   0.03

Status: ✅ GOOD FIT!
```

### Overfitting Example:

```
Final Metrics:
  Train Accuracy: 0.95 (95%)
  Val Accuracy:   0.68 (68%)
  Accuracy Gap:   0.27

Status: ⚠️ OVERFITTING DETECTED
```

### Underfitting Example:

```
Final Metrics:
  Train Accuracy: 0.62 (62%)
  Val Accuracy:   0.60 (60%)
  Accuracy Gap:   0.02

Status: ❌ UNDERFITTING DETECTED
```

---

## 🎯 Key Metrics to Watch

### 1. Loss Gap

```
Loss Gap = Validation Loss - Training Loss

• Gap < 0.1:  Excellent (Good Fit)
• Gap 0.1-0.2: Acceptable
• Gap > 0.2:  Warning (Overfitting)
```

### 2. Accuracy Gap

```
Accuracy Gap = Training Acc - Validation Acc

• Gap < 0.05:  Excellent
• Gap 0.05-0.10: Good
• Gap 0.10-0.15: Acceptable
• Gap > 0.15:  Overfitting
```

### 3. Validation Curve Trend

```
Good Signs:
✓ Validation loss decreasing
✓ Validation accuracy increasing
✓ Curves are smooth

Bad Signs:
✗ Validation loss increasing
✗ Validation accuracy decreasing
✗ High variance (zigzag pattern)
```

---

## 🛠️ Fixing Common Issues

### If Overfitting:

1. **Increase Dropout**

   ```python
   # In config.yaml or Config class
   dropout: 0.6  # Increase from 0.5
   ```

2. **Early Stopping**

   ```python
   patience: 5  # Stop if no improvement for 5 epochs
   ```

3. **Reduce Model Size**

   ```python
   hidden_size: 64   # Reduce from 128
   num_layers: 1     # Reduce from 2
   ```

4. **Add Regularization**
   ```python
   # Add L2 regularization
   optimizer = optim.Adam(model.parameters(),
                         lr=0.001,
                         weight_decay=0.01)
   ```

### If Underfitting:

1. **Increase Model Capacity**

   ```python
   hidden_size: 256  # Increase from 128
   num_layers: 3     # Increase from 2
   ```

2. **Train Longer**

   ```python
   epochs: 100  # Increase from 50
   ```

3. **Improve Features**

   - Add more frequency bands
   - Include additional connectivity metrics
   - Extract temporal features

4. **Adjust Learning Rate**
   ```python
   learning_rate: 0.01  # Increase from 0.001
   ```

---

## 📁 Output Files

After running the analysis, you'll find:

```
results/
├── learning_curves_YYYYMMDD_HHMMSS.png
│   └── Comprehensive visualization
│
└── learning_curve_analysis_YYYYMMDD_HHMMSS.txt
    └── Detailed text report
```

---

## 🎓 Understanding the Plots

### Plot 1: Loss Curves

```
What to look for:
- Both curves should decrease
- Should converge (get close to each other)
- Validation shouldn't increase while training decreases
```

### Plot 2: Accuracy Curves

```
What to look for:
- Both curves should increase
- Should be close to each other
- Validation should stabilize, not fluctuate wildly
```

### Plot 3: Overfitting Indicator

```
The gap visualization:
- Positive gap = Overfitting tendency
- Near zero = Good generalization
- Negative gap = Unusual (check regularization)
```

### Plot 4: Diagnostic Text

```
Automated analysis provides:
- Status (Good/Overfitting/Underfitting)
- Key metrics
- Specific recommendations
```

---

## ⚙️ Advanced Options

### Modify Training Duration

Edit the script and change:

```python
num_epochs = 30  # Change to 50, 100, etc.
```

### Modify Train/Val Split

```python
test_size = 0.2  # Change to 0.3 for more validation data
```

### Save Custom Checkpoints

```python
# Save model at specific epochs
if epoch in [10, 20, 30]:
    torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
```

---

## 📞 Troubleshooting

### Issue: Training too slow

**Solution:** Use `quick_learning_curves.py` with fewer epochs

### Issue: Out of memory

**Solution:** Reduce batch size

```python
batch_size: 8  # Reduce from 16
```

### Issue: Can't see overfitting

**Solution:** Train for more epochs (50-100)

### Issue: Results seem random

**Solution:** Set random seed at the start

```python
torch.manual_seed(42)
np.random.seed(42)
```

---

## 🎯 Best Practices

1. **Always run learning curve analysis** before deploying
2. **Compare multiple runs** to ensure consistency
3. **Monitor both loss and accuracy** - they tell different stories
4. **Look at trends**, not just final values
5. **Save the best model** based on validation metrics
6. **Document your findings** for reproducibility

---

## 📚 Further Reading

### What is Overfitting?

When a model learns the training data too well, including noise and outliers, it performs poorly on new data.

### What is Underfitting?

When a model is too simple to capture the underlying patterns in the data.

### What is the Bias-Variance Tradeoff?

- **High Bias (Underfitting):** Model too simple
- **High Variance (Overfitting):** Model too complex
- **Sweet Spot:** Balance between the two

---

## ✅ Quick Checklist

Before deploying your model:

- [ ] Run learning curve analysis
- [ ] Check for overfitting (gap < 15%)
- [ ] Verify validation performance (>70%)
- [ ] Ensure curves have converged
- [ ] Save diagnostic plots
- [ ] Document any issues found
- [ ] Test on independent dataset

---

## 🎉 Summary

**Learning curves are essential for:**

- ✅ Detecting overfitting/underfitting
- ✅ Choosing when to stop training
- ✅ Validating model quality
- ✅ Communicating results
- ✅ Publication-ready figures

**Run the analysis regularly to ensure your model is learning properly!**

---

_For more help, see the main README.md or PROJECT_STATUS_REPORT.md_
