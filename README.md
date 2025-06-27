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


### Performance Results

**Table 1: Track 2 Binary 1s (Batch size=32, Epochs=300, LR=0.000060)**  
| Model              | W-F1  | UW-F1 | W-Acc | UW-Acc |
|--------------------|-------|-------|-------|--------|
| Baseline           | 55.23 | 55.23 | 56.06 | 56.06  |
| BiLSTM             | 73.93 | 73.49 | 73.33 | 74.07  |
| CNN + BiLSTM       | 73.72 | 73.93 | 75.83 | 74.07  |
| Conformer Fusion   | 77.46 | 76.99 | 76.67 | 77.78  |
| CNN                | 81.53 | 81.38 | 81.67 | 81.48  |

**Table 2: Track 2 Ternary 1s (Batch size=32, Epochs=300, LR=0.000260)**  
| Model              | W-F1  | UW-F1 | W-Acc | UW-Acc |
|--------------------|-------|-------|-------|--------|
| Baseline           | 47.95 | 43.72 | 42.63 | 48.48  |
| BiLSTM             | 80.39 | 60.78 | 62.96 | 85.19  |
| CNN + BiLSTM       | 55.84 | 41.51 | 44.44 | 59.26  |
| Conformer Fusion   | 72.55 | 52.94 | 56.30 | 77.78  |
| CNN                | 76.68 | 57.24 | 60.74 | 81.48  |

**Table 3: Track 2 Binary 5s (Batch size=24, Epochs=300, LR=0.000050)**  
| Model              | W-F1  | UW-F1 | W-Acc | UW-Acc |
|--------------------|-------|-------|-------|--------|
| Baseline           | 62.02 | 60.02 | 60.61 | 60.61  |
| BiLSTM             | 77.46 | 76.99 | 76.67 | 77.78  |
| CNN + BiLSTM       | 70.45 | 70.33 | 70.83 | 70.37  |
| Conformer Fusion   | 81.53 | 81.38 | 81.67 | 81.78  |
| CNN                | 84.97 | 84.66 | 84.17 | 85.19  |

**Table 4: Track 2 Ternary 5s (Batch size=8, Epochs=500, LR=0.000040)**  
| Model              | W-F1  | UW-F1 | W-Acc | UW-Acc |
|--------------------|-------|-------|-------|--------|
| Baseline           | 42.82 | 39.38 | 41.29 | 42.42  |
| BiLSTM             | 66.14 | 49.84 | 54.07 | 70.37  |
| CNN + BiLSTM       | 52.38 | 39.05 | 42.22 | 55.56  |
| Conformer Fusion   | 73.17 | 53.73 | 57.04 | 77.78  |
| CNN                | 73.45 | 67.10 | 65.19 | 74.07  |

---


### **9. Contact**
If you have any questions or suggestions about the project, feel free to create an issue in the repository or contact the project maintainers at drishyau@iiitd.ac.in.

---
