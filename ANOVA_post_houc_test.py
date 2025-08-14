import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def perform_anova(data, features, group_col="task_index"):
    """
    Perform one-way ANOVA for each feature across unique groups in the specified column.
    
    Parameters:
    - data (pd.DataFrame): The dataset containing the features and grouping column.
    - features (list of str): List of feature names to analyze.
    - group_col (str): The column name used for grouping (default: "task_index").
    
    Returns:
    - pd.DataFrame: DataFrame containing ANOVA results with p-values.
    """
    anova_results = []
    
    for feature in features:
        groups = [data[data[group_col] == task][feature].dropna() for task in data[group_col].unique()]
        if all(len(group) > 1 for group in groups):  # Ensure each group has enough data
            anova_stat, p_value = stats.f_oneway(*groups)
            anova_results.append({"Feature": feature, "P-value": p_value})
        else:
            anova_results.append({"Feature": feature, "P-value": None})  # Handle insufficient data
    
    return pd.DataFrame(anova_results)









def analyze_semantic_features(data, semantic_features, group_column):
    """
    Performs Tukey's HSD post-hoc test, visualizes feature distributions with boxplots,
    and compares feature means between English and Chinese languages.
    
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

    # Visualize boxplots
    # for i, feature in enumerate(semantic_features, 1):
    #     plt.subplot(num_rows, 3, i)  # Dynamically adjust subplots
    #     sns.boxplot(x=group_column, y=feature, data=data, palette="coolwarm")
    #     plt.xticks(rotation=45)
    #     plt.title(f"Boxplot of {feature} across task types")
        # Visualize boxplots
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












# semantic_features_simple = [
#    'word_repetition_rate', 'idea_repetition_rate',
#        'semantic_outlier_rate', 'referential_ambiguity',
#        'abstract_vs_concrete_ratio', 'lexical_diversity',
#        'sentence_length_variation', 'pause_frequency', 'lexical_redundancy',
#        'content_density', 'syntactic_simplicity', 'semantic_drift',
#        'pronoun_to_noun_ratio', 'lexical_sophistication'
# ]

# Remove the folllowing features
# ⚠️ Feature 'Gender' is not numeric.
# ⚠️ Feature 'Source' is not numeric.
# ⚠️ Feature 'text' is not numeric.
# ⚠️ Feature 'Media' is not numeric.
# ⚠️ Feature 'Continents' is not numeric.
# ⚠️ Feature 'PID' is not numeric.
# ⚠️ Feature 'Countries' is not numeric.
# ⚠️ Feature 'Duration' is not numeric.
# ⚠️ Feature 'Location' is not numeric.
# ⚠️ Feature 'Diagnosis' is not numeric.
# ⚠️ Feature 'lemmas' is not numeric.
# ⚠️ Feature 'Languages' is not numeric.
# ⚠️ Feature 'Dataset' is not numeric.
# ⚠️ Feature 'Task' is not numeric.
# ⚠️ Feature 'Text_interviewer' is not numeric.
# ⚠️ Feature 'Date' is not numeric.
# ⚠️ Feature 'Moca' is not numeric.
# ⚠️ Feature 'Text_participant' is not numeric.
# ⚠️ Feature 'Transcriber' is not numeric.
# ⚠️ Feature 'tokens' is not numeric.
# ⚠️ Feature 'Setting' is not numeric.
# ⚠️ Feature 'File_ID' is not numeric.
# ⚠️ Feature 'Participants' is not numeric.
# ⚠️ Feature 'Modality' is not numeric.
# ⚠️ Feature 'Comment' is not numeric.
# ⚠️ Feature 'Education' is not numeric.


# Adding task index column
#combined_df_elfen["task_index"]= combined_df_elfen["Languages"]+combined_df_elfen["File_ID"].apply(lambda x: x.split("-")[2])


# Filtering
# filter=result_elfen[result_elfen['P-value'] < 0.05]['Feature'].reset_index(drop=True)
# new=combined_df_elfen[filter]