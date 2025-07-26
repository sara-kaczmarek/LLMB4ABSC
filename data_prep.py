import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_xml_to_df(path):

    tree = ET.parse(path)
    root = tree.getroot()

    data = []

    for sentence in root.iter("sentence"):
        text = sentence.find("text").text.strip()

        aspect_terms = sentence.find("aspectTerms")
        if aspect_terms is not None:
            for at in aspect_terms.findall("aspectTerm"):
                aspect = at.attrib.get("term", "").strip()
                polarity = at.attrib.get("polarity", "").strip().lower()
                if polarity != "conflict" and aspect.lower() != "null":
                    data.append({
                        "sentence": text,
                        "aspect": aspect.lower(),
                        "polarity": polarity
                    })

        opinions = sentence.find("Opinions")
        if opinions is not None:
            for op in opinions.findall("Opinion"):
                aspect = op.attrib.get("target", "").strip()
                polarity = op.attrib.get("polarity", "").strip().lower()
                if polarity != "conflict" and aspect.lower() != "null":
                    data.append({
                        "sentence": text,
                        "aspect": aspect.lower(),
                        "polarity": polarity
                    })

    return pd.DataFrame(data)

def load_and_sample(path, rename_cols, n, seed=42):
    df = pd.read_csv(path).rename(columns=rename_cols)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)

def load_and_encode(path):
    df = pd.read_csv(path)
    df = df[df["polarity"].isin(["positive", "neutral", "negative"])].copy()
    df["label"] = LabelEncoder().fit(["positive", "neutral", "negative"]).transform(df["polarity"])
    return df

def clean_sentences(df):
    return df[df['sentence'].apply(lambda x: isinstance(x, str))].copy()
