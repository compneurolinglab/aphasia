
# Aphasia-Inspired Model Lesioning

This repository contains the code used in our research on **lesioning large language models (LLMs) to study aphasia-like behaviors**. Our study explores how selectively disrupting model components can replicate linguistic deficits similar to those observed in human aphasia patients.

---

## 🔍 Overview
- We use **Visual-Chinese-LLaMA-Alpaca (VisualCLA；Yang et al., 2023; Cui et al., 2023)** as the base model for our experiments.
- We implement a **lesioning framework** to systematically disable specific components of the model and analyze its linguistic performance.
- We provide code to **evaluate the lesioned models** on controlled linguistic tasks inspired by aphasia studies.
- **No model weights are shared** in compliance with licensing restrictions.

---

## 📂 Dataset
We use an **existing aphasia dataset** from:
- **Bi et al. (2015), Han et al. (2013)**:  
🔹 **Data Ethics & Privacy**:
- **The dataset was originally approved from the Institutional Review Board of the State Key Laboratory of Cognitive Neuroscience and Learning, Beijing Normal University.).
- **All patient data has been anonymized**.
- **We do not share any data from this dataset**. Interested researchers should refer to the original papers and researchers for access.

---

## ⚙️ Features
- **Lesioning framework**: Methods for selectively disabling attention heads, feedforward layers, and embedding components.
- **Aphasia-inspired benchmarks**: Code for testing **semantic** and **syntactic impairments** in LLMs.
- **Performance evaluation**: Scripts to measure model degradation post-lesioning.


