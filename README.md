# Hand_crafted_linguistic_feature_extraction
This repository contains the implementation and resources for the research study ## ** "Statistical Analysis of Interpretable Linguistic Features for MCI Detection in Bilingual Speech" **.

### **1. ANOVA & Post-hoc Analysis**

One-way ANOVA and Tukey's HSD post-hoc tests with visualizations to compare statistical differences across groups.

### **2. Feature Cleaning**

Removes non-numeric features and filters out features with excessive missing or invalid values (preprocessing step).

### **3. Linguistic Feature Extraction**

* **Elfen Extractor** – Surface, emotion, lexical richness, and POS features.
* **LFTK Extractor** – Broad general linguistic features.

### **4. Visualization**

Feature distribution and boxplot plots for comparison across groups or tasks.


```
├── ANOVA_post_houc_test.py        # One-way ANOVA + Tukey HSD with visualizations
├── Feature_normalizer.py      # Clean datasets, remove non-numeric/high-missing features
├── custom_extractor.py       # Elfen extractor: surface, emotion, lexical richness, POS, and LFTK extractor: broad linguistic feature set
├── kernel_etimation_distribution_and_task_box_plot.py    # Distributions & boxplots across groups
```
