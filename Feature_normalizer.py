import pandas as pd

def non_numeric_feature_removal(df,features_list):
    non_numeric_feature_list = []
    for feature in features_list:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            print(f"⚠️ Feature '{feature}' is not numeric.")
            non_numeric_feature_list.append(feature)
    df = df.drop(non_numeric_feature_list, axis=1)
    return df,non_numeric_feature_list

def get_coulumn_with_high_missing_value(df, threshold):
    missing_percentage = df.apply(lambda col: ((col.isna())| (col==0)|(col=="Unknown")).mean())
    cols_above_threshold = missing_percentage[missing_percentage>threshold]
    df_cleaned = df.drop(columns=cols_above_threshold.index)
    for col in df_cleaned.select_dtypes(include= "number").columns:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
    #print(f"Columns with more than {int(threshold*100)}% missing values: ")
    #print(cols_above_threshold)
    return df_cleaned, cols_above_threshold