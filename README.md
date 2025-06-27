## MPDD_Challenge
**Author:** drishyau  
**Date:** October 2024  

## MPDD Challenge: Dataset and Evaluation Metrics

### Track 2: Young Adult Depression Detection (MPDD-Young)
This track addresses depression detection among young adults, a high-risk group often understudied. The dataset includes:
- **Participants:** 110 individuals
- **Data Modalities:** Audiovisual recordings
- **Annotations:**
  - Personality labels
  - Demographic information

The goal is to understand how factors like geographic environment, age, gender, and personality traits contribute to depression in young adults.

---

### Evaluation Metrics
The MPDD Challenge employs:
- **Unweighted Accuracy (Acc):** Proportion of correct predictions (class imbalance ignored)
- **Unweighted F1 Score (F1):** Harmonic mean of precision and recall (unweighted)

Final evaluation score for Track2:


## 📊 Model Performance Summary

### Key Results
| Task              | Best Model | UW-Acc | Improvement vs Baseline |
|-------------------|------------|--------|-------------------------|
| Binary 1s         | CNN        | 81.48  | +25.42%                 |
| Ternary 1s        | BiLSTM     | 85.19  | +36.71%                 |
| Binary 5s         | CNN        | 85.19  | +24.58%                 |
| Ternary 5s        | CNN        | 74.07  | +31.65%                 |

### Detailed Results
<details>
<summary>📈 Expand full results table</summary>

**Table 1: Track 2 Binary 1s (Batch=32, Epochs=300, LR=6e-5)**  
| Model              | W-F1  | UW-F1 | W-Acc | UW-Acc |
|--------------------|-------|-------|-------|--------|
| Baseline           | 55.23 | 55.23 | 56.06 | 56.06  |
| BiLSTM             | 73.93 | 73.49 | 73.33 | 74.07  |
| CNN + BiLSTM       | 73.72 | 73.93 | 75.83 | 74.07  |
| Conformer Fusion   | 77.46 | 76.99 | 76.67 | 77.78  |
| **CNN**            | **81.53** | **81.38** | **81.67** | **81.48**  |

**Table 2: Track 2 Ternary 1s (Batch=32, Epochs=300, LR=2.6e-4)**  
| Model              | W-F1  | UW-F1 | W-Acc | UW-Acc |
|--------------------|-------|-------|-------|--------|
| Baseline           | 47.95 | 43.72 | 42.63 | 48.48  |
| **BiLSTM**         | **80.39** | 60.78 | 62.96 | **85.19**  |
| CNN + BiLSTM       | 55.84 | 41.51 | 44.44 | 59.26  |
| Conformer Fusion   | 72.55 | 52.94 | 56.30 | 77.78  |
| CNN                | 76.68 | 57.24 | 60.74 | 81.48  |

**Table 3: Track 2 Binary 5s (Batch=24, Epochs=300, LR=5e-5)**  
| Model              | W-F1  | UW-F1 | W-Acc | UW-Acc |
|--------------------|-------|-------|-------|--------|
| Baseline           | 62.02 | 60.02 | 60.61 | 60.61  |
| BiLSTM             | 77.46 | 76.99 | 76.67 | 77.78  |
| CNN + BiLSTM       | 70.45 | 70.33 | 70.83 | 70.37  |
| Conformer Fusion   | 81.53 | 81.38 | 81.67 | 81.78  |
| **CNN**            | **84.97** | **84.66** | **84.17** | **85.19**  |

**Table 4: Track 2 Ternary 5s (Batch=8, Epochs=500, LR=4e-5)**  
| Model              | W-F1  | UW-F1 | W-Acc | UW-Acc |
|--------------------|-------|-------|-------|--------|
| Baseline           | 42.82 | 39.38 | 41.29 | 42.42  |
| BiLSTM             | 66.14 | 49.84 | 54.07 | 70.37  |
| CNN + BiLSTM       | 52.38 | 39.05 | 42.22 | 55.56  |
| Conformer Fusion   | 73.17 | 53.73 | 57.04 | 77.78  |
| **CNN**            | **73.45** | **67.10** | **65.19** | **74.07**  |
</details>

---

---

## 🛠️ Setup & Execution

### 1. Project Structure
MPDD_Code/
├── models/
│   └── our/
│       └── our_model.py      # <--- Your custom models live here!
├── scripts/
│   ├── Track2/
│   │   └── binary_5s_track2.yaml   # <--- Training YAML configs
│   └── test.yaml                   # <--- Testing YAML configs
├── train.py                        # <--- Training entry point
├── test.py                         # <--- Testing entry point


### 2. Installation
Clone repository
git clone https://github.com/yourusername/alzheimer-prediction.git
cd alzheimer-prediction
### 3. Configure Your Model
Edit the YAML configuration under MPDD_Code/scripts/Track2/ for training, or MPDD_Code/scripts/ for testing.
Specify your model in the code, for example:
model = CNN(opt)
Adjust other settings (batch size, epochs, learning rate, etc.) as needed in your YAML file.

### 4. Training

Run the training script with your chosen configuration:
python3 train.py --config=MPDD_Code/scripts/Track2/binary_5s_track2.yaml
 Results will be saved in the output folder specified in your YAML file.

### 5. Testing
In test.py, specify the model you want to evaluate.
Use the relevant YAML file for testing:
python3 test.py --config=MPDD_Code/scripts/test.yaml
Test results will also be saved as per the YAML configuration.

### Tips & Best Practices
Ensure your data paths and output directories are correctly set in the YAML files.
For experimenting with different models, simply change the model assignment in your code and update the YAML config as needed.
Check logs and saved results for performance metrics and model checkpoints.



### Key Observations
1. CNN models consistently outperform other architectures across tasks
2. Binary classification yields better results than ternary classification
3. Longer sequence length (5s vs 1s) improves performance in binary tasks
4. Conformer Fusion shows competitive results in ternary classification


### **9. Contact**
If you have any questions or suggestions about the project, feel free to create an issue in the repository or contact the project maintainers at drishyau@iiitd.ac.in.

---
