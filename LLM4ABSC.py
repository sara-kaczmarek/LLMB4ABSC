
from sentence_transformers.util import cos_sim
import pandas as pd
import numpy as np
import torch
import re

def sc_prompt(sentence, aspect, model, tokenizer, max_tokens=10):
    prompt = (
        "Classify the sentiment expressed towards the given aspect term in the following sentence. \n"
        "Choose either 'positive', 'neutral', or 'negative'. \n"
        f"Sentence: {sentence} \n"
        f"Aspect: {aspect} \n"
        "Sentiment:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return decoded

label_mapping = {
    "good": "positive",
    "great": "positive",
    "excellent": "positive",
    "amazing": "positive",
    "bad": "negative",
    "terrible": "negative",
    "awful": "negative",
    "horrible": "negative",
    "fine": "neutral",
    "okay": "neutral",
    "neutral": "neutral",
    "positive": "positive",
    "negative": "negative"
}

def normalize_and_map_label(output):
    if not isinstance(output, str):
        return "error"

    first_word = output.strip().split()[0].lower()

    first_word = unicodedata.normalize("NFKC", first_word)
    first_word = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', first_word)
    first_word = first_word.translate(str.maketrans('', '', string.punctuation))

    if first_word in label_mapping:
        return label_mapping[first_word]

    return "error"

def run_inference(df_test, inference_function, **inference_kwargs):
    results = []

    for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Running inference"):
        sentence = row['sentence']
        aspect = row['aspect']

        try:
            output = inference_function(sentence, aspect, **inference_kwargs)
            sentiment = normalize_and_map_label(output)

            if sentiment == "error":
                print(f"\nPARSE ERROR at index {idx}:")
                print("Raw model output:", output)

        except Exception as e:
            sentiment = "error"
            print(f"\nGENERATION ERROR at index {idx}: {e}")

        results.append({
            "sentence": sentence,
            "aspect": aspect,
            "predicted_sentiment": sentiment,
            "true_sentiment": row["polarity"]
        })

    return pd.DataFrame(results)

def compute_simcse_embeddings(df):
    if "generated_sentence" in df.columns:
        df["sentence"] = df["generated_sentence"]
    elif "sentence" in df.columns:
        df["sentence"] = df["sentence"]
    else:
        raise ValueError("No column named 'sentence' or 'generated_sentence' found in dataframe.")

    df["sentence"] = df["sentence"].astype(str)
    df["sentence_embedding"] = df["sentence"].apply(lambda s: model_sbert.encode(s, convert_to_tensor=True))
    return df

def sc_fewshot_prompt(sentence, aspect, df, model, tokenizer, k, scenario, max_tokens=10):
    input_embedding = model_sbert.encode(sentence, convert_to_tensor=True)
    df_same_aspect = df[df['original_aspect'] == aspect].copy()

    if len(df_same_aspect) == 0:
        print("Fallback: No matches found for aspect. Using full dataset.")
        df_same_aspect = df.copy()

    classes = ["positive", "neutral", "negative"]
    per_class_k = k // len(classes)

    if scenario == "random":
        retrieved_examples = df_same_aspect.sample(n=min(k, len(df_same_aspect)), random_state=42)

    elif scenario == "random_equal":
        retrieved_examples = pd.concat([
            df_same_aspect[df_same_aspect["sentiment"] == c].sample(
                n=per_class_k, replace=True, random_state=42
            ) if len(df_same_aspect[df_same_aspect["sentiment"] == c]) >= per_class_k
            else df_same_aspect.sample(n=per_class_k, replace=True, random_state=42)
            for c in classes
        ])

    elif scenario == "simcse":
        embeddings = torch.stack(df_same_aspect["sentence_embedding"].tolist())
        similarities = cos_sim(input_embedding, embeddings)[0].cpu().numpy()
        top_k_idx = np.argsort(similarities)[::-1][:k]
        retrieved_examples = df_same_aspect.iloc[top_k_idx]

    elif scenario == "simcse_equal":
        retrieved_examples = []

        for cls in classes:
            df_cls = df_same_aspect[df_same_aspect["sentiment"] == cls]
            if len(df_cls) == 0:
                df_cls = df_same_aspect

            embeddings = torch.stack(df_cls["sentence_embedding"].tolist())
            similarities = cos_sim(input_embedding, embeddings)[0].cpu().numpy()
            top_k_idx = np.argsort(similarities)[::-1][:per_class_k]
            retrieved_examples.extend(df_cls.iloc[top_k_idx].to_dict("records"))

        retrieved_examples = pd.DataFrame(retrieved_examples)

    else:
        raise ValueError("Invalid scenario")

    prompt = (
        "Classify the sentiment expressed towards the given aspect in the following sentence.\n"
        "Choose either 'positive', 'neutral', or 'negative'.\n"
        "Below are examples of sentences with their corresponding sentiment labels:\n\n"
    )

    for _, row in retrieved_examples.iterrows():
        example = (
            f"### Sentence: {row['generated_sentence']}\n"
            f"### Aspect: {row['original_aspect']}\n"
            f"### Sentiment: {row['sentiment']}\n"
        )
        prompt += example + "\n\n"

    prompt += f"Now classify the following sentence\n\nSentence: {sentence}\nAspect: {aspect}\nSentiment:"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    sentiment = decoded.strip().split()[0].lower()

    return sentiment