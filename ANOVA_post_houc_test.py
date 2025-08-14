import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def perform_anova(data, features, group_col):
    """
    Perform one-way ANOVA for each numeric feature across unique groups.
    
    Skips non-numeric features automatically.

    Returns:
    - pd.DataFrame: DataFrame containing ANOVA results with p-values.
    """
    anova_results = []

    for feature in features:
        if not pd.api.types.is_numeric_dtype(data[feature]):
            print(f"⚠️ Skipping non-numeric feature: '{feature}'")
            continue

        groups = [
            data[data[group_col] == group][feature].dropna()
            for group in data[group_col].unique()
        ]

        # Ensure all groups have at least 2 values
        if all(len(group) > 1 for group in groups):
            try:
                anova_stat, p_value = stats.f_oneway(*groups)
                anova_results.append({"Feature": feature, "P-value": p_value})
            except Exception as e:
                print(f"⚠️ Error running ANOVA on '{feature}': {e}")
                anova_results.append({"Feature": feature, "P-value": None})
        else:
            print(f"⚠️ Skipping '{feature}' due to insufficient data in some groups.")
            anova_results.append({"Feature": feature, "P-value": None})

    return pd.DataFrame(anova_results)










def analyze_semantic_features(data, semantic_features, group_column):
    """
    Performs Tukey's HSD post-hoc test, visualizes feature distributions with boxplots,
    and compares feature means between the English and Chinese languages.
    
    Parameters:
    - data (pd.DataFrame): The dataset containing semantic features.
    - semantic_features (list): List of feature names to analyze.
    - group_column (str): Column name for grouping (default is "task_index").
    """
    tukey_results = {}
    
    # Perform Tukey's HSD test for each feature
    for feature in semantic_features:
        try:
            tukey = pairwise_tukeyhsd(
                endog=data[feature], 
                groups=data[group_column], 
                alpha=0.05
            )
            tukey_results[feature] = tukey
            print(f"Post-hoc test for {feature}:\n", tukey, "\n")
        except Exception as e:
            print(f"Error in Tukey test for {feature}: {e}")

    # Ensure subplot layout is appropriate
    num_features = len(semantic_features)
    num_rows = int(np.ceil(num_features / 3))  # Dynamic row calculation
    plt.figure(figsize=(15, 5 * num_rows))  # Adjust figure size

    
    for i, feature in enumerate(semantic_features, 1):
        if not pd.api.types.is_numeric_dtype(data[feature]):
            print(f"⚠️ Skipping non-numeric feature for boxplot: '{feature}'")
            continue

        plt.subplot(num_rows, 3, i)
        sns.boxplot(x=group_column, y=feature, data=data, palette="coolwarm")
        plt.xticks(rotation=45)
        plt.title(f"Boxplot of {feature} across task types")


    plt.tight_layout()
    plt.show()

    # Ensure 'language' column exists
    if "Languages" not in data.columns:
        data["Languages"] = data[group_column].apply(lambda x: "Chinese" if "zh" in str(x) else "English")

    # Compute feature means for different groups
    #feature_means = data.groupby("Languages")[semantic_features].mean()
    # Filter only numeric features
    numeric_features = [f for f in semantic_features if pd.api.types.is_numeric_dtype(data[f])]

# Compute feature means
    feature_means = data.groupby("Languages")[numeric_features].mean()
    
    # Visualize language differences
    plt.figure(figsize=(12, 6))
    #feature_means.T.plot(kind="bar", figsize=(12, 6), colormap="viridis")
    feature_means.T.plot(kind="bar", figsize=(12, 6), colormap="viridis")

    plt.title("Comparison of Semantic Features between Languages")
    plt.ylabel("Feature Mean Value")
    plt.xticks(rotation=45)
    plt.legend(title="Languages")
    plt.show()

    return tukey_results











