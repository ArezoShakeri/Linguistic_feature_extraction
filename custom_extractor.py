import pandas as pd
import spacy
spacy.load("zh_core_web_sm")
import zh_core_web_sm
spacy.load("en_core_web_sm")
import en_core_web_sm
import lftk
import polars as pl
from elfen.extractor import Extractor
import json


# This function only extractes feature in the following areas: "surface", "emotion","lexical_richness","pos"
def extract_feature_elfen(df,language,model,name):
    extractor = Extractor(data = df,language = language,model = model)#"zh_core_web_sm"
    extractor.extract_feature_group(feature_group = ["surface", "emotion","lexical_richness","pos"]) #Extractable features for zh
    # we have to remove the column with pl.object type to be able to save features
    df_data = extractor.data

    # Drop the 'nlp' column
    df_data = df_data.drop("nlp")

    # Ensure 'token_freqs' (object column) is converted to a JSON string
    df_data = df_data.with_columns(
        pl.col("token_freqs").map_elements(json.dumps).alias("token_freqs")
    )

    # Write the cleaned DataFrame to a JSON Lines file
    df_data.write_ndjson(name+"_elfen_spacy.jsonl")







# Agnistic feature exraction using LFTK library
def lftk_feature_extractor(df,model,name_output_file):
    """
    Extracts features from a DataFrame using the LFTK library.
    
    Parameters:
    - df: Polars DataFrame containing the text data."""
    nlp = spacy.load(model)
    text_list= df["text"].to_list()
    docs = list(nlp.pipe(text_list, batch_size=1000))
    LFTK=lftk.Extractor(docs=docs)
    searched_features_general = lftk.search_features(language="general", return_format="list_key")
    extracted_features = LFTK.extract(features=searched_features_general)
    features_df = pd.DataFrame(extracted_features)
    features_pl_df = pl.from_pandas(features_df)
    df_with_features = pl.concat([df, features_pl_df], how="horizontal")
    # Save the updated DataFrame to a JSONL file
    df_with_features.write_ndjson(name_output_file+"_lftk.jsonl")
