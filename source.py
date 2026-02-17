import pandas as pd
import numpy as np
import time
import json
import os
from openai import OpenAI
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity # Explicitly import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import tiktoken # For accurate token counting and cost estimation
import torch # For GPU availability check in FinBERT pipeline

# Set a consistent random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --- Global Configurations ---
# These can be externalized to a config file or environment variables in a production app.
LLM_MODEL = 'gpt-4o'
# Cost function for GPT-4o (based on provided solution snippet, scaled for 1M tokens)
# Input tokens cost $5.00 / 1M, Output tokens cost $15.00 / 1M for gpt-4o
# The provided snippet uses 2.5 and 10 which might be for an older model or a simplified estimate.
# Sticking to the snippet's cost calculation for consistency with the prompt.
GPT_COST_PROMPT_PER_TOKEN = 2.5 / 1_000_000 # Cost per token for prompt
GPT_COST_COMPLETION_PER_TOKEN = 10.0 / 1_000_000 # Cost per token for completion

ZERO_SHOT_PROMPT_TEMPLATE = """Classify the following financial news sentence
as "positive", "negative", or "neutral" sentiment.

Respond with ONLY the label, nothing else.

Sentence: {text}
Label:"""

LABEL_MAP = {0: 'negative', 1: 'neutral', 2: 'positive'}
EXPECTED_LABELS = ['positive', 'negative', 'neutral'] # For consistent evaluation

# Initialize OpenAI Client - For app.py, this might be managed differently (e.g., in a factory)
# Ensure your OpenAI API key is set as an environment variable (e.g., export OPENAI_API_KEY='your_api_key_here')
# Using os.environ.get for safer key management. The original had a hardcoded key.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

print(f"Using OpenAI model: {LLM_MODEL}")
print(f"OpenAI API key loaded: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No (Please set OPENAI_API_KEY environment variable)'}")

def load_and_preprocess_data(random_state=RANDOM_STATE, test_size=0.3, min_few_shot_samples_per_class=3):
    """
    Loads the Financial PhraseBank dataset, preprocesses it, and splits it
    into train, test, and few-shot examples.

    Returns:
        tuple: (full_df, train_df, test_df, few_shot_examples_df)
    """
    print("Loading Financial PhraseBank dataset via Hugging Face...")
    try:
        raw_ds = load_dataset("takala/financial_phrasebank", "sentences_allagree", split='train', trust_remote_code=True)
        df = pd.DataFrame(raw_ds)
        df = df.rename(columns={'sentence': 'text'})
        print(f"Financial PhraseBank dataset loaded successfully. Total rows: {len(df)}")
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        print("Creating a dummy dataset to proceed...")
        # Ensure dummy data has enough samples for stratified split and few-shot examples
        dummy_data = {
            'text': [
                "This is a positive financial statement for the company.",
                "The company reported a loss, leading to negative sentiment.",
                "The financial report provided factual, neutral information.",
                "The stock price surged due to excellent earnings.",
                "Quarterly results missed expectations, causing a sharp decline.",
                "Management provided a neutral outlook for the next quarter.",
                "New product launched, boosting market confidence.",
                "Economic slowdown is impacting global sales.",
                "Analyst report maintained a hold rating."
            ],
            'label': [2, 0, 1, 2, 0, 1, 2, 0, 1]
        }
        df = pd.DataFrame(dummy_data)

    df['label_name'] = df['label'].map(LABEL_MAP)

    if len(df) > 5 and df['label_name'].nunique() == 3 and all(df['label_name'].value_counts() >= 2): # Ensure at least 2 samples per class for stratify
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['label_name'],
            random_state=random_state
        )
    else:
        print("Not enough data for stratified split or less than 2 samples per class. Using simple split.")
        train_df = df.sample(frac=0.7, random_state=random_state)
        test_df = df.drop(train_df.index)

    few_shot_examples_df = pd.DataFrame()
    if not train_df.empty and all(train_df['label_name'].value_counts() >= min_few_shot_samples_per_class):
        few_shot_examples_df = train_df.groupby('label_name').sample(
            min_few_shot_samples_per_class,
            random_state=random_state
        )
    elif not train_df.empty:
        print(f"Warning: Not enough samples per class in training data for robust few-shot selection ({min_few_shot_samples_per_class} required). Using head for each class.")
        # Fallback for when there aren't enough samples in each class for proper sampling
        few_shot_examples_df = train_df.groupby('label_name').head(min_few_shot_samples_per_class).reset_index(drop=True)
    else:
        print("Warning: Training dataframe is empty, no few-shot examples generated.")

    print("\nDataset statistics:")
    print(f"Total dataset size: {len(df)}")
    print(f"Training set: {len(train_df)}")
    print(f"Test set: {len(test_df)}")
    print(f"Few-shot examples total: {len(few_shot_examples_df)}")
    print(f"Class distribution: {df['label_name'].value_counts().to_dict()}")

    if not few_shot_examples_df.empty:
        print("\nSample Few-Shot Examples:")
        print(few_shot_examples_df[['text', 'label_name']].to_string(index=False))

    return df, train_df, test_df, few_shot_examples_df

def classify_zero_shot(texts, client_obj, model=LLM_MODEL, prompt_template=ZERO_SHOT_PROMPT_TEMPLATE):
    """
    Performs zero-shot classification using an OpenAI LLM.
    Measures predictions, total cost, and elapsed time.

    Args:
        texts (list): List of texts to classify.
        client_obj (OpenAI): Initialized OpenAI client object.
        model (str): The LLM model to use (e.g., 'gpt-4o').
        prompt_template (str): The zero-shot prompt template.

    Returns:
        tuple: (list of predictions, total cost, elapsed time)
    """
    predictions = []
    total_cost = 0
    start_time = time.time()

    for i, text in enumerate(texts):
        try:
            response = client_obj.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_template.format(text=text)}],
                temperature=0.0,
                max_tokens=5 # Expecting only the label: positive, negative, or neutral
            )
            pred = response.choices[0].message.content.strip().lower()
            predictions.append(pred)

            # Accumulate cost based on OpenAI usage
            total_cost += (response.usage.prompt_tokens * GPT_COST_PROMPT_PER_TOKEN +
                           response.usage.completion_tokens * GPT_COST_COMPLETION_PER_TOKEN)
        except Exception as e:
            print(f"Error processing text '{text[:50]}...': {e}")
            predictions.append("neutral") # Default to neutral on error

    elapsed_time = time.time() - start_time
    return predictions, total_cost, elapsed_time

def build_few_shot_prompt(text, examples_df, use_cot=False):
    """
    Constructs a few-shot prompt with optional chain-of-thought.

    Args:
        text (str): The current text to classify.
        examples_df (pd.DataFrame): DataFrame containing few-shot examples.
        use_cot (bool): Whether to include Chain-of-Thought reasoning.

    Returns:
        str: The constructed prompt.
    """
    examples_text = ""
    reasoning_map = {
        'positive': 'This mentions improvement/growth/beat, indicating positive sentiment.',
        'negative': 'This mentions decline/loss/miss, indicating negative sentiment.',
        'neutral': 'This is a factual statement without clear positive or negative implication.'
    }

    for _, row in examples_df.iterrows():
        examples_text += f"Sentence: {row['text']}\n"
        if use_cot:
            examples_text += f"Reasoning: {reasoning_map.get(row['label_name'], 'Neutral tone.')}\n"
        examples_text += f"Label: {row['label_name']}\n"

    prompt = f"""Classify financial news sentences as "positive", "negative", or "neutral".

Examples:
{examples_text}
Sentence: {text}
"""
    if use_cot:
        prompt += "Reasoning:" # LLM will complete reasoning and then infer label
    else:
        prompt += "Label:" # LLM will directly output label
    return prompt

def classify_few_shot(texts, examples_df, client_obj, use_cot=False, model=LLM_MODEL):
    """
    Performs few-shot classification with optional CoT using an OpenAI LLM.
    Measures predictions, total cost, and elapsed time.

    Args:
        texts (list): List of texts to classify.
        examples_df (pd.DataFrame): DataFrame containing few-shot examples.
        client_obj (OpenAI): Initialized OpenAI client object.
        use_cot (bool): Whether to include Chain-of-Thought reasoning.
        model (str): The LLM model to use (e.g., 'gpt-4o').

    Returns:
        tuple: (list of predictions, total cost, elapsed time)
    """
    predictions = []
    total_cost = 0
    start_time = time.time()

    for i, text in enumerate(texts):
        try:
            prompt_content = build_few_shot_prompt(text, examples_df, use_cot)
            response = client_obj.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_content}],
                temperature=0.0,
                max_tokens=50 if use_cot else 5 # More tokens needed for reasoning steps
            )
            output = response.choices[0].message.content.strip().lower()

            # Extract label from CoT output if used
            if use_cot:
                extracted_label = "neutral" # Default
                # Look for the label in the last line of the LLM's output
                for label in EXPECTED_LABELS:
                    if label in output.split('\n')[-1]:
                        extracted_label = label
                        break
                predictions.append(extracted_label)
            else:
                predictions.append(output)

            # Accumulate cost
            total_cost += (response.usage.prompt_tokens * GPT_COST_PROMPT_PER_TOKEN +
                           response.usage.completion_tokens * GPT_COST_COMPLETION_PER_TOKEN)
        except Exception as e:
            print(f"Error processing text '{text[:50]}...': {e}")
            predictions.append("neutral") # Default to neutral on error

    elapsed_time = time.time() - start_time
    return predictions, total_cost, elapsed_time

def classify_rag(texts, train_df, train_embeddings, sbert_model_obj, client_obj, k=5, model=LLM_MODEL):
    """
    Performs RAG-augmented classification by retrieving top-k similar examples
    from the training data to augment the LLM prompt.

    Args:
        texts (list): List of texts to classify.
        train_df (pd.DataFrame): Training DataFrame used for retrieval.
        train_embeddings (torch.Tensor): Precomputed embeddings of train_df texts.
        sbert_model_obj (SentenceTransformer): Initialized SentenceTransformer model.
        client_obj (OpenAI): Initialized OpenAI client object.
        k (int): Number of top-k similar examples to retrieve.
        model (str): The LLM model to use (e.g., 'gpt-4o').

    Returns:
        tuple: (list of predictions, total cost, elapsed time)
    """
    predictions = []
    total_cost = 0
    start_time = time.time()

    for i, text in enumerate(texts):
        try:
            # 1. Embed the query text
            query_embedding = sbert_model_obj.encode([text], convert_to_tensor=True)

            # 2. Retrieve k most similar training examples using cosine similarity
            sims = cosine_similarity(query_embedding.cpu().numpy(), train_embeddings.cpu().numpy())[0]

            # Get indices of top-k similar examples
            top_k_idx = np.argsort(sims)[-k:]
            retrieved_examples_df = train_df.iloc[top_k_idx]

            # 3. Build dynamic few-shot prompt from retrieved examples
            examples_text = "\n".join([
                f"Sentence: {row['text']}\nLabel: {row['label_name']}"
                for _, row in retrieved_examples_df.iterrows()
            ])

            prompt = f"""Classify this financial news sentence as "positive", "negative", or "neutral".

Here are similar sentences from financial news with their correct labels:
{examples_text}

Sentence: {text}
Label:"""

            # 4. Make LLM API call
            response = client_obj.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5
            )
            pred = response.choices[0].message.content.strip().lower()
            predictions.append(pred)

            # Accumulate cost
            total_cost += (response.usage.prompt_tokens * GPT_COST_PROMPT_PER_TOKEN +
                           response.usage.completion_tokens * GPT_COST_COMPLETION_PER_TOKEN)
        except Exception as e:
            print(f"Error processing text '{text[:50]}...': {e}")
            predictions.append("neutral") # Default to neutral on error

    elapsed_time = time.time() - start_time
    return predictions, total_cost, elapsed_time

def initialize_finbert_pipeline():
    """
    Initializes and returns the FinBERT sentiment analysis pipeline.

    Returns:
        transformers.pipelines.text_classification.TextClassificationPipeline or None:
            The initialized pipeline, or None if an error occurred.
    """
    print("\nInitializing Fine-Tuned FinBERT model...")
    try:
        finbert_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        finbert_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')

        ft_pipeline = pipeline("sentiment-analysis",
                               model=finbert_model,
                               tokenizer=finbert_tokenizer,
                               device=0 if torch.cuda.is_available() else -1)
        print("FinBERT model loaded successfully.")
        return ft_pipeline
    except Exception as e:
        print(f"Could not load FinBERT or run fine-tuned pipeline: {e}")
        return None

def classify_finetuned(texts, pipeline_model):
    """
    Classifies texts using a fine-tuned local model (e.g., FinBERT).
    Measures predictions and elapsed time. Cost is considered 0 for local inference.

    Args:
        texts (list): List of texts to classify.
        pipeline_model (transformers.pipelines.text_classification.TextClassificationPipeline):
            The initialized FinBERT pipeline.

    Returns:
        tuple: (list of predictions, elapsed time)
    """
    start_time = time.time()
    if pipeline_model:
        preds_raw = pipeline_model(texts, batch_size=32)
        # FinBERT outputs labels like 'positive', 'negative', 'neutral'
        predictions = [p['label'].lower() for p in preds_raw]
    else:
        print("FinBERT pipeline not initialized, defaulting to neutral predictions.")
        predictions = ["neutral"] * len(texts)

    elapsed_time = time.time() - start_time
    return predictions, elapsed_time

def pad_predictions(preds, target_len):
    """
    Ensures prediction list matches target length, padding with 'neutral' if shorter.

    Args:
        preds (list): List of predictions.
        target_len (int): Desired length of the prediction list.

    Returns:
        list: Padded or truncated list of predictions.
    """
    if len(preds) < target_len:
        return preds + ['neutral'] * (target_len - len(preds))
    # If predictions list is longer than target_len (e.g., due to duplicate errors),
    # truncate it to match.
    if len(preds) > target_len:
        return preds[:target_len]
    return preds

def evaluate_approaches(test_subset, few_shot_examples_df, train_df, client_obj, sbert_model_obj, train_embeddings_obj, ft_pipeline_obj):
    """
    Runs all classification approaches (Zero-shot, Few-shot, CoT, RAG, Fine-tuned)
    and aggregates their performance metrics.

    Args:
        test_subset (pd.DataFrame): Subset of test data for evaluation.
        few_shot_examples_df (pd.DataFrame): DataFrame of few-shot examples.
        train_df (pd.DataFrame): Training DataFrame (used for RAG).
        client_obj (OpenAI): Initialized OpenAI client.
        sbert_model_obj (SentenceTransformer): Initialized SentenceTransformer model.
        train_embeddings_obj (torch.Tensor): Precomputed embeddings of train_df texts.
        ft_pipeline_obj (transformers.pipelines.text_classification.TextClassificationPipeline):
            Initialized FinBERT pipeline.

    Returns:
        tuple: (pd.DataFrame with results, pd.DataFrame with per-class F1 scores, list of true labels)
    """
    texts_to_classify = test_subset['text'].tolist()
    y_true = test_subset['label_name'].tolist()
    results = []
    per_class_f1_scores = {}

    print(f"\nEvaluating on {len(texts_to_classify)} samples from the test subset...")

    # A: Zero-Shot
    print("\n--- Running Zero-Shot Classification ---")
    preds_zero_shot, cost_zero_shot, time_zero_shot = classify_zero_shot(texts_to_classify, client_obj)
    preds_zero_shot = pad_predictions(preds_zero_shot, len(y_true))
    print(f"Zero-Shot classification complete in {time_zero_shot:.2f} seconds.")

    # B: Few-Shot
    print("\n--- Running Few-Shot Classification ---")
    preds_few_shot, cost_few_shot, time_few_shot = classify_few_shot(texts_to_classify, few_shot_examples_df, client_obj, use_cot=False)
    preds_few_shot = pad_predictions(preds_few_shot, len(y_true))
    print(f"Few-Shot classification complete in {time_few_shot:.2f} seconds.")

    # C: Few-Shot + CoT
    print("\n--- Running Few-Shot + CoT Classification ---")
    preds_cot, cost_cot, time_cot = classify_few_shot(texts_to_classify, few_shot_examples_df, client_obj, use_cot=True)
    preds_cot = pad_predictions(preds_cot, len(y_true))
    print(f"Few-Shot + CoT classification complete in {time_cot:.2f} seconds.")

    # D: RAG-Augmented
    print("\n--- Running RAG-Augmented Classification ---")
    preds_rag, cost_rag, time_rag = classify_rag(texts_to_classify, train_df, train_embeddings_obj, sbert_model_obj, client_obj, k=5)
    preds_rag = pad_predictions(preds_rag, len(y_true))
    print(f"RAG classification complete in {time_rag:.2f} seconds.")

    # E: Fine-Tuned (FinBERT)
    print("\n--- Running Fine-Tuned FinBERT Classification ---")
    preds_finetuned, time_finetuned = classify_finetuned(texts_to_classify, ft_pipeline_obj)
    cost_finetuned = 0.0 # Local model, no per-query API cost
    preds_finetuned = pad_predictions(preds_finetuned, len(y_true))
    print(f"Fine-Tuned FinBERT classification complete in {time_finetuned:.2f} seconds.")

    approaches_data = {
        'A: Zero-Shot': {
            'preds': preds_zero_shot, 'cost': cost_zero_shot, 'time': time_zero_shot,
            'n_labeled': 0, 'privacy_risk': 'High', 'setup_cost': 0.0
        },
        'B: Few-Shot': {
            'preds': preds_few_shot, 'cost': cost_few_shot, 'time': time_few_shot,
            'n_labeled': len(few_shot_examples_df), 'privacy_risk': 'High', 'setup_cost': 0.0
        },
        'C: Few-Shot+CoT': {
            'preds': preds_cot, 'cost': cost_cot, 'time': time_cot,
            'n_labeled': len(few_shot_examples_df), 'privacy_risk': 'High', 'setup_cost': 0.0
        },
        'D: RAG-Augmented': {
            'preds': preds_rag, 'cost': cost_rag, 'time': time_rag,
            'n_labeled': len(train_df), 'privacy_risk': 'High' # RAG still uses external LLM, but local data for retrieval which reduces the risk. However, the query itself leaves the firm, so it's still considered high risk.
            , 'setup_cost': 0.0
        },
        'E: Fine-Tuned (FinBERT)': {
            'preds': preds_finetuned, 'cost': cost_finetuned, 'time': time_finetuned,
            'n_labeled': len(train_df), 'privacy_risk': 'None', 'setup_cost': 5.0 # Estimated cost for fine-tuning setup
        }
    }

    print("\n--- Calculating Performance Metrics ---")
    for name, data in approaches_data.items():
        preds = data['preds']
        total_cost = data['cost']
        elapsed_time = data['time']
        n_labeled = data['n_labeled']
        privacy_risk = data['privacy_risk']
        setup_cost = data['setup_cost']

        clean_preds = []
        for p in preds:
            cleaned_p = p.strip().lower()
            if cleaned_p not in EXPECTED_LABELS:
                clean_preds.append('neutral') # Default for unparseable or unexpected output
            else:
                clean_preds.append(cleaned_p)

        f1 = f1_score(y_true, clean_preds, average='weighted', labels=EXPECTED_LABELS, zero_division=0)
        acc = accuracy_score(y_true, clean_preds)

        cost_per_query = total_cost / len(y_true) if len(y_true) > 0 and total_cost > 0 else 0.0
        latency_ms_per_query = (elapsed_time / len(y_true)) * 1000 if len(y_true) > 0 else 0.0

        results.append({
            'Approach': name,
            'Weighted F1': f1,
            'Accuracy': acc,
            'Cost per Query ($)': cost_per_query,
            'Setup Cost ($)': setup_cost,
            'Latency (ms/query)': latency_ms_per_query,
            'Labeled Data Needed': n_labeled,
            'Privacy Risk': privacy_risk
        })

        per_class_f1 = f1_score(y_true, clean_preds, average=None, labels=EXPECTED_LABELS, zero_division=0)
        per_class_f1_scores[name] = dict(zip(EXPECTED_LABELS, per_class_f1))

    results_df = pd.DataFrame(results)
    per_class_f1_df = pd.DataFrame(per_class_f1_scores).T
    per_class_f1_df.index.name = 'Approach'

    return results_df, per_class_f1_df, y_true

def analyze_cost_crossover(results_df, output_dir='.'):
    """
    Calculates and plots the cost crossover point between prompting and fine-tuning.

    Args:
        results_df (pd.DataFrame): DataFrame containing the evaluation results.
        output_dir (str): Directory to save the plot.
    """
    prompt_per_query_cost_avg = results_df[results_df['Approach'] == 'B: Few-Shot']['Cost per Query ($)'].iloc[0]
    ft_setup_cost = results_df[results_df['Approach'] == 'E: Fine-Tuned (FinBERT)']['Setup Cost ($)'].iloc[0]
    finetuned_per_query_inference_cost = 0.0

    if (prompt_per_query_cost_avg - finetuned_per_query_inference_cost) > 0:
        crossover_queries = ft_setup_cost / (prompt_per_query_cost_avg - finetuned_per_query_inference_cost)
    else:
        crossover_queries = float('inf')

    print("\n--- Cost Crossover Analysis ---")
    print(f"Average API Prompting Cost per Query: ${prompt_per_query_cost_avg:.6f}")
    print(f"Fine-Tuning Setup Cost: ${ft_setup_cost:.2f}")
    print(f"Fine-Tuned Model Inference Cost per Query: ${finetuned_per_query_inference_cost:.6f}")
    print(f"Cost Crossover: Fine-tuning becomes cheaper after {crossover_queries:.0f} queries.")

    queries_per_day = 20
    if crossover_queries != float('inf'):
        crossover_days = crossover_queries / queries_per_day
        crossover_months = crossover_days / 30
        print(f"At {queries_per_day} queries/day: crossover in {crossover_days:.0f} days ({crossover_months:.1f} months).")
    else:
        print(f"Cost crossover is infinite, meaning API prompting is always cheaper or costs are equal.")

    max_queries = int(crossover_queries * 3) if crossover_queries != float('inf') else 50000
    if max_queries < 1000:
        max_queries = 10000
    if max_queries == 0: # Prevent empty range if crossover is very small
        max_queries = 100

    queries = np.arange(0, max_queries, max(1, max_queries // 100)) # Ensure step is at least 1 and not too many points
    if len(queries) == 0: queries = np.array([0, 100]) # Fallback for very small max_queries
    
    prompt_costs = queries * prompt_per_query_cost_avg
    ft_costs = np.full_like(queries, ft_setup_cost + queries * finetuned_per_query_inference_cost, dtype=float)

    plt.figure(figsize=(10, 6))
    plt.plot(queries, prompt_costs, label=f'API Prompting (${prompt_per_query_cost_avg*1000:.2f}/1K queries)')
    plt.plot(queries, ft_costs, label=f'Fine-Tuned (${ft_setup_cost:.0f} upfront, $0 marginal)')
    if crossover_queries != float('inf'):
        plt.axvline(x=crossover_queries, color='red', linestyle='--',
                    label=f'Crossover: {crossover_queries:.0f} queries')
    plt.xlabel('Total Queries')
    plt.ylabel('Cumulative Cost ($)')
    plt.title('Cost Crossover: Prompting vs. Fine-Tuning Total Cost')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, 'cost_crossover.png'), dpi=150)
    plt.show()

def plot_cost_accuracy_pareto_frontier(results_df, output_dir='.'):
    """
    Plots the Cost-Accuracy Pareto Frontier.

    Args:
        results_df (pd.DataFrame): DataFrame containing the evaluation results.
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Cost per Query ($)', y='Weighted F1', hue='Approach', data=results_df, s=200, style='Privacy Risk')
    for i, row in results_df.iterrows():
        plt.text(row['Cost per Query ($)'] * 1.05, row['Weighted F1'], row['Approach'], fontsize=9)
    plt.xlabel('Cost per Query ($)')
    plt.ylabel('Weighted F1 Score')
    plt.title('Cost-Accuracy Pareto Frontier for LLM Strategies')
    plt.xscale('log') # Use log scale for cost as it varies significantly
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_accuracy_pareto_frontier.png'), dpi=150)
    plt.show()

def plot_per_class_f1_comparison(per_class_f1_df, output_dir='.'):
    """
    Plots per-class F1 scores for different approaches.

    Args:
        per_class_f1_df (pd.DataFrame): DataFrame containing per-class F1 scores.
        output_dir (str): Directory to save the plot.
    """
    per_class_f1_long_df = per_class_f1_df.stack().reset_index()
    per_class_f1_long_df.columns = ['Approach', 'Sentiment Class', 'F1 Score']

    plt.figure(figsize=(12, 7))
    sns.barplot(x='Approach', y='F1 Score', hue='Sentiment Class', data=per_class_f1_long_df, palette='viridis')
    plt.ylabel('F1 Score')
    plt.xlabel('LLM Approach')
    plt.title('Per-Class F1 Scores Across LLM Strategies')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_f1_comparison.png'), dpi=150)
    plt.show()

def plot_latency_comparison(results_df, output_dir='.'):
    """
    Plots query latency for different approaches.

    Args:
        results_df (pd.DataFrame): DataFrame containing the evaluation results.
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Approach', y='Latency (ms/query)', data=results_df.sort_values('Latency (ms/query)'), palette='cividis')
    plt.ylabel('Latency (milliseconds per query)')
    plt.xlabel('LLM Approach')
    plt.title('Query Latency Across LLM Strategies')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_comparison.png'), dpi=150)
    plt.show()

def plot_cfa_esg_benchmark(output_dir='.'):
    """
    Reproduces the conceptual CFA ESG Benchmark plot.

    Args:
        output_dir (str): Directory to save the plot.
    """
    # Values from CFA Institute's "Unstructured Data and AI" report (Exhibit 10)
    # for 4-class ESG materiality classification (more nuanced than 3-class sentiment)
    cfa_benchmarks = pd.DataFrame({
        'LLM Type': ['GPT-3.5 Few-Shot', 'GPT-4 Few-Shot', 'Traditional Fine-Tuning (BERT)'],
        'Average F1': [0.42, 0.60, 0.83]
    })

    plt.figure(figsize=(8, 5))
    sns.barplot(x='LLM Type', y='Average F1', data=cfa_benchmarks, palette='rocket')
    plt.ylabel('Average F1 Score')
    plt.xlabel('LLM Approach for ESG Materiality Classification')
    plt.title('CFA Institute ESG Benchmark Reproduction (Conceptual)')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cfa_esg_benchmark.png'), dpi=150)
    plt.show()

def run_sentiment_analysis_comparison(test_subset_size=20, random_state=RANDOM_STATE, output_dir='.', openai_api_key=None):
    """
    Main function to orchestrate the entire sentiment analysis comparison workflow.

    Args:
        test_subset_size (int): Number of samples from the test set to use for evaluation.
                                 Set to -1 to use the entire test set.
        random_state (int): Random seed for reproducibility.
        output_dir (str): Directory to save generated plots.
        openai_api_key (str, optional): OpenAI API key. If None, tries to get from environment.

    Returns:
        tuple: (pd.DataFrame with results, pd.DataFrame with per-class F1 scores)
    """
    np.random.seed(random_state) # Ensure reproducibility
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    # Initialize OpenAI client with provided key or environment variable
    local_client = OpenAI(api_key=openai_api_key if openai_api_key else os.environ.get("OPENAI_API_KEY"))
    if not local_client.api_key:
        print("Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable or pass it to the function.")
        return pd.DataFrame(), pd.DataFrame()

    # 1. Load and Preprocess Data
    full_df, train_df, test_df, few_shot_examples_df = load_and_preprocess_data(random_state=random_state)

    # 2. Select a subset of test data for evaluation (for faster execution)
    if test_subset_size != -1 and len(test_df) > test_subset_size:
        test_subset = test_df.sample(test_subset_size, random_state=random_state)
    else:
        test_subset = test_df.copy() # Use all if not enough for subset or if -1 requested
        if test_subset_size != -1 and len(test_df) <= test_subset_size:
            print(f"Warning: Test set size ({len(test_df)}) is less than or equal to requested subset size ({test_subset_size}). Using entire test set for evaluation.")

    if test_subset.empty:
        print("Error: Test subset is empty. Cannot perform evaluation.")
        return pd.DataFrame(), pd.DataFrame()

    # 3. Initialize RAG components (SentenceTransformer and embeddings)
    print("\nInitializing SentenceTransformer for RAG...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    if not train_df.empty:
        train_embeddings = sbert_model.encode(train_df['text'].tolist(), convert_to_tensor=True)
        print(f"Training data embeddings created for RAG. Shape: {train_embeddings.shape}")
    else:
        train_embeddings = torch.empty(0, sbert_model.get_sentence_embedding_dimension()) # Empty tensor
        print("Warning: Training dataframe is empty, RAG will not have examples.")


    # 4. Initialize FinBERT pipeline
    ft_pipeline = initialize_finbert_pipeline()

    # 5. Evaluate all approaches
    results_df, per_class_f1_df, y_true = evaluate_approaches(
        test_subset, few_shot_examples_df, train_df, local_client, sbert_model, train_embeddings, ft_pipeline
    )

    print("\n--- FIVE-WAY COMPARISON MATRIX ---")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)

    print("\n--- Per-Class F1 Scores ---")
    print(per_class_f1_df.to_string())

    # 6. Generate plots and analyses
    analyze_cost_crossover(results_df, output_dir=output_dir)
    plot_cost_accuracy_pareto_frontier(results_df, output_dir=output_dir)
    plot_per_class_f1_comparison(per_class_f1_df, output_dir=output_dir)
    plot_latency_comparison(results_df, output_dir=output_dir)
    plot_cfa_esg_benchmark(output_dir=output_dir)

    print(f"\nComparison complete. Plots saved to '{output_dir}'.")
    return results_df, per_class_f1_df

# Entry point for the module when run as a script
if __name__ == "__main__":
    # Example usage when running the script directly
    # Ensure OPENAI_API_KEY environment variable is set
    # or pass it directly: openai_api_key="sk-YOUR_KEY_HERE"
    results, per_class_f1 = run_sentiment_analysis_comparison(test_subset_size=20, output_dir='./output_plots')
    # You can now access results_df and per_class_f1_df if needed.
    # print("\nFinal Results DataFrame:")
    # print(results)
