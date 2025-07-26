import re
import random
import warnings
from collections import defaultdict
import pandas as pd
from tqdm import tqdm


###### ASPECT EXPANSION (AX) ######
def ax_prompt(aspect, domain, model, tokenizer, max_tokens=30, device="cuda"):
    prompt = (
        f"You are an expert in Aspect-Based Sentiment Analysis for the {domain} domain.\n"
        "You will perform an Aspect Expansion task. In this task, you are given an aspect term and your goal is to generate related terms that customers may use to refer to the same aspect.\n"
        "Generate 3 to 5 related terms, including synonyms, homonyms or alternative expressions.\n"
        "Output format: [term1, term2, term3, term4, term5]\n\n"
        "If fewer than 5 related terms exist, output as many as applicable.\n\n"
        f"Aspect term: {aspect}\n"
        "Output:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    expanded_text = decoded.replace(prompt, "").strip()
    return expanded_text

def extract_and_parse_list(generated_text):
    match = re.search(r"\[(.*?)\]", generated_text)
    if not match:
        return None
    items = match.group(1).split(",")
    return [item.strip().strip('"').strip("'") for item in items if item.strip()]

def generate_aspect_expansions(df, model, tokenizer, domain="restaurant", device="cuda"):

    unique_aspects = df["aspect"].dropna().unique().tolist()

    records = []

    for aspect in tqdm(unique_aspects, desc="Generating expansions"):
        expanded_text = ax_prompt(aspect, domain, model, tokenizer, device=device)

        expanded_list = extract_and_parse_list(expanded_text)
        if expanded_list is None:
            expanded_list = []

        expanded_list.append(aspect)

        seen = set()
        cleaned = []
        for x in expanded_list:
            x = x.strip().lower()
            if x not in seen:
                seen.add(x)
                cleaned.append(x)

        records.append((aspect, cleaned))

    df = pd.DataFrame(records, columns=["original_aspect", "expanded_aspects"])

    df["expanded_aspects"] = df["expanded_aspects"].apply(lambda x: ", ".join(x))

    return df

def filter_expanded_aspects_to_nouns(df, nlp):
    def is_valid_noun(phrase):
        phrase = phrase.strip()
        if not phrase:
            return False
        doc = nlp(phrase)
        return any(token.pos_ in {"NOUN", "PROPN"} for token in doc)

    def filter_nouns(row):
        original = row["original_aspect"].strip()
        terms = [term.strip() for term in row["expanded_aspects"].split(",")]
        valid = [term for term in terms if is_valid_noun(term)]

        if original not in valid:
            valid.append(original)

        return ", ".join(sorted(set(valid), key=valid.index))

    df["expanded_aspects"] = df.apply(filter_nouns, axis=1)
    return df

###### DATA GENERATION (DG) ######
def dg_prompt(aspect, sentiment, domain, model, tokenizer, max_tokens=50, device="cuda"):
    neutral_hint = (
        " For neutral sentiment, do not express any opinion, use factual or descriptive language, and avoid emotional or judgmental words."
        if sentiment.lower() == "neutral" else ""
    )

    prompt = (
        f"You are generating a single {domain} review sentence.\n"
        "The sentence must mention the given aspect and reflect the given sentiment.\n"
        "Use natural human-like language, realistic and rich in vocabulary.\n"
        "Output format: one sentence of maximum 20 words. Stop generating after 1 sentence." +
        neutral_hint + "\n" +
        f"Aspect: {aspect}\n"
        f"Sentiment: {sentiment}\n"
        "Output:"
    )

    encoded = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in encoded.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.9,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = decoded.replace(prompt, "").strip().split("\n")[0].strip()
    return clean_generated_sentence(generated_text)

def clean_generated_sentence(text):
    if pd.isna(text):
        return None

    text = str(text).split('\n')[0].strip()
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s.,!?\'()-]', '', text)
    text = re.sub(r'^[\'"“”‘’]+|[\'"“”‘’]+$', '', text)

    match = re.split(r'(?<=[.!?])\s', text, maxsplit=1)
    text = match[0]

    return text if text else None

def generate_synthetic_sentences_from_expansions(df_expansions, dataset_choice, model, tokenizer, n_per_sentiment=1, device="cuda"):
    sentiments = ["positive", "neutral", "negative"]
    domain = "laptop" if dataset_choice == "Lapt14" else "restaurant"
    results = []

    for _, row in tqdm(df_expansions.iterrows(), total=len(df_expansions), desc="Generating Synthetic Sentences"):
        original = row["original_aspect"]
        expansions = [term.strip() for term in row["expanded_aspects"].split(",") if term.strip()]

        for sentiment in sentiments:
            for _ in range(n_per_sentiment):
                used = random.choice(expansions) if expansions else original

                gen_text = dg_prompt(
                    aspect=used,
                    sentiment=sentiment,
                    domain=domain,
                    model=model,
                    tokenizer=tokenizer,
                    device=device
                )

                results.append({
                    "original_aspect": original,
                    "used_extended_aspect": used,
                    "sentiment": sentiment,
                    "generated_sentence": gen_text
                })

    return pd.DataFrame(results)
  

########## QUALITY FILTERING PIPELINE (QFP) ###############

warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")

def manual_filter(df):
    df = df.copy()
    df["reason_dropped"] = ""

    df["generated_sentence"] = df["generated_sentence"].fillna("").astype(str)

    subjective_words = [
        "good", "great", "delicious", "amazing", "fantastic", "excellent",
        "bad", "awful", "horrible", "terrible", "lovely", "awesome",
        "tasty", "perfect", "nice", "worst", "wonderful", "love", "hate",
        "disgusting", "incredible", "favorite", "boring", "unpleasant",
        "yummy", "satisfying", "flavorful", "perfection", "perfectly",
        "cozy", "welcoming", "inviting", "authentic", "bold", "rich", "original",
        "interesting", "relaxed", "appealing", "iconic", "classic",
        "well seasoned", "well cooked", "popular", "skilled", "recommended",
        "recommend", "bland", "quiet", "clean", "quick service", "steep prices",
        "discerning palate", "would order again", "famous","perfection","perfectly"
    ]
    pattern = r'\b(?:' + '|'.join(map(re.escape, subjective_words)) + r')\b'

    mask_missing = df["generated_sentence"].str.strip() == ""
    df.loc[mask_missing, "reason_dropped"] = "missing_or_empty"

    dup_mask = df.duplicated(subset=["generated_sentence"]) & df["reason_dropped"].eq("")
    df.loc[dup_mask, "reason_dropped"] = "duplicate"

    punct_mask = ~df["generated_sentence"].str.strip().str.endswith(tuple(".!?")) & df["reason_dropped"].eq("")
    df.loc[punct_mask, "reason_dropped"] = "no_end_punctuation"

    short_mask = df["generated_sentence"].str.split().str.len() < 2
    short_mask = short_mask & df["reason_dropped"].eq("")
    df.loc[short_mask, "reason_dropped"] = "too_short"

    is_neutral = df["sentiment"].str.lower() == "neutral"
    contains_subjective = df["generated_sentence"].str.lower().str.contains(pattern, na=False, regex=True)
    sub_mask = is_neutral & contains_subjective & df["reason_dropped"].eq("")
    df.loc[sub_mask, "reason_dropped"] = "subjective_in_neutral"

    dropped_df = df[df["reason_dropped"] != ""].reset_index(drop=True)
    filtered_df = df[df["reason_dropped"] == ""].drop(columns=["reason_dropped"]).reset_index(drop=True)

    return filtered_df, dropped_df


def llm_filter(df, model, tokenizer, domain, device="cuda"):
    from collections import defaultdict
    import pandas as pd
    from tqdm import tqdm
    import re

    def extract_decision(text, label=""):
        match = re.search(r"(?i)answer:\s*(yes|no)", text, re.DOTALL)
        if match:
            decision = match.group(1).capitalize()

            return decision
        else:

            return None

    drop_counts = defaultdict(int)
    results = []

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Applying filters")):
        sentence = row["generated_sentence"]
        aspect = row["used_extended_aspect"]
        sentiment = row["sentiment"]
        original_aspect = row.get("original_aspect", None)


        row_result = {
            "generated_sentence": sentence,
            "original_aspect": original_aspect,
            "used_extended_aspect": aspect,
            "sentiment": sentiment,
            "aspect_sentiment_match": None,
            "domain_match": None,
            "english_fluent": None
        }

        combined_prompt = (
            f"Does this sentence express a {sentiment} sentiment towards the aspect '{aspect}'?\n"
            f"Sentence: \"{sentence}\"\n"
            f"Respond with either Yes or No. No explanations.\nAnswer:"
        )
        inputs = tokenizer(combined_prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=10, temperature=0.0,
                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
        combined_response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        combined_decision = extract_decision(combined_response, "Aspect+Sentiment")
        row_result["aspect_sentiment_match"] = combined_decision

        if combined_decision is None:
            drop_counts["aspect_sentiment_no_answer"] += 1
            continue
        elif combined_decision != "Yes":
            drop_counts["aspect_sentiment_answered_no"] += 1
            continue

        if sentiment.lower() == "neutral":
            neutral_prompt = (
                f"Is there any emotional or opinionated expression toward the aspect '{aspect}' in this sentence?\n"
                f"Sentence: \"{sentence}\"\n"
                f"Respond with either Yes or No. No explanations.\nAnswer:"
            )
            inputs = tokenizer(neutral_prompt, return_tensors="pt").to(device)
            output = model.generate(**inputs, max_new_tokens=10, temperature=0.0,
                                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
            neutral_response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

            neutral_decision = extract_decision(neutral_response, "Neutral Emotion Check")

            if neutral_decision is None:
                drop_counts["neutral_emotion_no_answer"] += 1
                continue
            elif neutral_decision == "Yes":
                drop_counts["neutral_emotion_expressed"] += 1
                continue

        df_prompt = (
            f"Does the following sentence sound like something a customer would write in a {domain} review?\n"
            f"Sentence: \"{sentence}\"\n"
            f"Respond with either Yes or No. No explanations.\nAnswer:"
        )
        inputs = tokenizer(df_prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=10, temperature=0.0,
                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
        df_response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        df_decision = extract_decision(df_response, "Domain")
        row_result["domain_match"] = df_decision

        if df_decision is None:
            drop_counts["domain_no_answer"] += 1
            continue
        elif df_decision != "Yes":
            drop_counts["domain_answered_no"] += 1
            continue

        ef_prompt = (
            f"Is the following sentence fluent and grammatically correct English?\n"
            f"Sentence: \"{sentence}\"\n"
            f"Respond with either Yes or No. No explanations.\nAnswer:"
        )
        inputs = tokenizer(ef_prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=10, temperature=0.0,
                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
        ef_response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        ef_decision = extract_decision(ef_response, "English Fluency")
        row_result["english_fluent"] = ef_decision

        if ef_decision is None:
            drop_counts["fluency_no_answer"] += 1
            continue
        elif ef_decision != "Yes":
            drop_counts["fluency_answered_no"] += 1
            continue

        results.append(row_result)

    df_filtered = pd.DataFrame(results).reset_index(drop=True)
    df_final = df_filtered.drop(columns=[
        "aspect_sentiment_match",
        "domain_match",
        "english_fluent"
    ])

    df_drop_stats = pd.DataFrame(list(drop_counts.items()), columns=["Stage", "Dropped"])
    return df_final, df_drop_stats

########## ITERATIVE DATA GENERATION (IDG) ###############

def iterative_DG(
    df_synthetic_filtered,
    df_expansions_filtered,
    model,
    tokenizer,
    dataset_choice,
    manual_filter,
    llm_filter,
    n_per_sentiment=10,
    max_attempts=10,
    device="cuda"
):
    sentiments = ["positive", "neutral", "negative"]
    domain = "laptop" if dataset_choice == "Lapt14" else "restaurant"
    total_target = len(df_expansions_filtered) * len(sentiments) * n_per_sentiment

    icl_pool = df_synthetic_filtered.drop_duplicates(subset=["generated_sentence"]).copy()
    results = [icl_pool.copy()]
    global_counter = len(icl_pool)

    for _, row in tqdm(df_expansions_filtered.iterrows(), total=len(df_expansions_filtered), desc="Generating Balanced Data"):
        original_aspect = row["original_aspect"]
        expanded_list = [x.strip() for x in row["expanded_aspects"].split(",") if x.strip()]
        if not expanded_list:
            expanded_list = [original_aspect]

        for sentiment in sentiments:
            current_count = len(icl_pool[(icl_pool["original_aspect"] == original_aspect) & (icl_pool["sentiment"] == sentiment)])
            attempts = 0

            while current_count < n_per_sentiment and attempts < max_attempts:
                demo_pool = icl_pool[
                    ((icl_pool["original_aspect"] != original_aspect) & (icl_pool["sentiment"] == sentiment)) |
                    ((icl_pool["original_aspect"] == original_aspect) & (icl_pool["sentiment"] != sentiment))
                ]
                if demo_pool.empty:
                    break

                demos = demo_pool.sample(n=min(5, len(demo_pool)), random_state=random.randint(0, 10000))

                prompt = (
                    f"You are generating a single {domain} review sentence.\n"
                    "Each sentence must mention the given aspect and reflect the given sentiment.\n"
                    "Use natural human-like language, realistic and rich in vocabulary.\n"
                )

                if sentiment.lower() == "neutral":
                    prompt += "The sentence must express no clear opinion or emotion. It should be factual or descriptive only.\n"

                prompt += "Output format: one sentence of maximum 20 words. Stop generating after 1 sentence.\n\n"

                for _, demo in demos.iterrows():
                    prompt += (
                        f"Aspect: {demo['used_extended_aspect']}\n"
                        f"Sentiment: {demo['sentiment']}\n"
                        f"Output: {demo['generated_sentence']}\n\n"
                    )

                aspect_used = random.choice(expanded_list)
                prompt += (
                    f"Aspect: {aspect_used}\n"
                    f"Sentiment: {sentiment}\n"
                    "Output:"
                )

                try:
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.9,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3
                    )

                    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated = decoded.replace(prompt, "").strip().split("\n")[0].strip()
                    cleaned = clean_generated_sentence(generated)

                    if not cleaned:
                        attempts += 1
                        continue

                    candidate_df = pd.DataFrame([{
                        "original_aspect": original_aspect,
                        "used_extended_aspect": aspect_used,
                        "sentiment": sentiment,
                        "generated_sentence": cleaned
                    }])

                    candidate_df, _ = manual_filter(candidate_df)
                    if len(candidate_df) == 0:
                        attempts += 1
                        continue

                    filtered_df, _ = llm_filter(candidate_df, model, tokenizer, domain, device)
                    if not filtered_df.empty:
                        results.append(filtered_df)
                        icl_pool = pd.concat([icl_pool, filtered_df], ignore_index=True)
                        current_count += 1
                        global_counter += 1
                        print(f"[Accepted: {global_counter}/{total_target}] — ({original_aspect}, {sentiment})")

                except Exception as e:
                    print(f"[ERROR] {e}")
                attempts += 1

    final_df = pd.concat(results).reset_index(drop=True)
    return final_df