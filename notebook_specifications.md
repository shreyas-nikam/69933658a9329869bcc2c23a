
# Fine-Tuning vs. Prompt Engineering: Optimal LLM Strategy for Financial Text Classification

## Introduction: Navigating LLM Deployment as a Quant Analyst

As a CFA Charterholder and investment professional at a leading financial firm, my daily responsibilities often involve distilling critical insights from vast amounts of unstructured financial text. Whether it's analyzing market sentiment from news headlines, identifying material ESG factors from corporate reports, or classifying financial documents, the efficiency and accuracy of these tasks directly impact investment research, risk management, and client reporting.

The advent of Large Language Models (LLMs) offers a powerful toolkit, but deciding on the optimal deployment strategy is a complex challenge. Should we rely on off-the-shelf general LLMs with clever prompting (prompt engineering), or invest in fine-tuning smaller, specialized models on our proprietary domain data? Each path has distinct trade-offs in terms of accuracy, cost, latency, data requirements, and critical privacy considerations.

This notebook outlines a practical, data-driven framework to systematically evaluate five distinct LLM-based approaches for a financial text classification task:
1.  **Zero-Shot Prompting:** Leveraging a general LLM with only a direct instruction.
2.  **Few-Shot Prompting:** Providing the general LLM with a few labeled examples in the prompt.
3.  **Few-Shot + Chain-of-Thought (CoT) Prompting:** Augmenting few-shot prompting with reasoning steps to guide the LLM.
4.  **Retrieval-Augmented Generation (RAG):** Dynamically retrieving and including the most similar labeled examples from our domain data in the prompt.
5.  **Fine-Tuned Small Model:** Using a domain-specific model fine-tuned on a labeled financial dataset.

My goal is to measure the performance across these dimensions for each approach and construct a decision framework that guides optimal LLM deployment within our firm, optimizing resource allocation, managing risk, and ultimately enhancing our competitive advantage.

---

## 1. Environment Setup and Data Preparation for Financial Sentiment Analysis

Our first step is to prepare our Python environment and load the necessary financial text data. We'll use the `Financial PhraseBank` dataset, a standard benchmark for financial sentiment analysis, as a proxy for the classification tasks we perform daily.

### Required Libraries Installation

We need several libraries for interacting with LLMs, managing data, performing NLP tasks (embeddings, fine-tuning), and evaluating models.

```python
!pip install openai transformers torch scikit-learn pandas datasets tiktoken sentence-transformers matplotlib seaborn
```

### Import Dependencies and Set Up API Key

Next, we import all necessary modules. For accessing OpenAI's LLMs, we'll need an API key, which should be loaded securely, ideally from an environment variable.

```python
import pandas as pd
import numpy as np
import time
import json
import os
from openai import OpenAI
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import tiktoken # For accurate token counting and cost estimation

# Set a consistent random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Initialize OpenAI Client
# Ensure your OpenAI API key is set as an environment variable
# For example: export OPENAI_API_KEY='your_api_key_here'
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# LLM model choice for API calls
LLM_MODEL = 'gpt-4o' # Using gpt-4o as specified in the original content

# Cost function for GPT-4o (based on provided solution snippet, scaled for 1M tokens)
# Input tokens cost $5.00 / 1M, Output tokens cost $15.00 / 1M for gpt-4o
# The provided snippet uses 2.5 and 10 which might be for an older model or a simplified estimate.
# Sticking to the snippet's cost calculation for consistency with the prompt.
# Cost per 1e6 tokens: (prompt_tokens * 2.5 + completion_tokens * 10)
GPT_COST_PROMPT_PER_TOKEN = 2.5 / 1_000_000 # Cost per token for prompt
GPT_COST_COMPLETION_PER_TOKEN = 10.0 / 1_000_000 # Cost per token for completion

print(f"Using OpenAI model: {LLM_MODEL}")
print(f"OpenAI API key loaded: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No (Please set OPENAI_API_KEY environment variable)'}")
```

### Load and Prepare Financial PhraseBank Dataset

We'll load the `Financial PhraseBank` dataset, which contains sentences from financial news categorized into positive, neutral, or negative sentiment. This serves as our labeled financial text classification task.

```python
# Load the dataset
print("Loading Financial PhraseBank dataset...")
fpb = load_dataset("financial_phrasebank", "sentences_allagree")
df = pd.DataFrame(fpb['train'])

# Rename columns for clarity
df.columns = ['text', 'label']

# Map numerical labels to descriptive sentiment categories
label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
df['label_name'] = df['label'].map(label_map)

# Split the dataset into training and test sets
# We stratify to ensure an even distribution of sentiment labels in both sets.
train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['label_name'], random_state=RANDOM_STATE)

# Sample a small, fixed number of labeled examples per class from the training set for few-shot prompting
few_shot_examples_df = train_df.groupby('label_name').sample(3, random_state=RANDOM_STATE)

print("\nDataset statistics:")
print(f"Total dataset size: {len(df)}")
print(f"Training set: {len(train_df)} (for fine-tuning and RAG examples)")
print(f"Test set: {len(test_df)} (shared evaluation for all approaches)")
print(f"Few-shot examples (3 per class): {len(few_shot_examples_df)} (for prompting)")
print(f"Class distribution in full dataset: {df['label_name'].value_counts().to_dict()}")
print("\nSample Few-Shot Examples:")
print(few_shot_examples_df[['text', 'label_name']].to_string(index=False))
```

---

## 2. Prompt Engineering Strategies with a General LLM

As an investment professional, my initial thought when faced with a new text classification task is often to leverage an existing, powerful LLM. Prompt engineering allows me to guide these general models without needing to retrain them. We'll explore three increasingly sophisticated prompting strategies: Zero-Shot, Few-Shot, and Few-Shot with Chain-of-Thought (CoT). These approaches are quick to implement and offer immediate insights into the LLM's capabilities on financial text.

### Zero-Shot Prompting: Direct Instruction

Zero-shot prompting is the simplest approach, providing the LLM with only the task instruction and the input text. It's useful for quickly gauging the model's inherent understanding of a domain. For financial sentiment, we need the model to recognize nuanced positive, neutral, or negative connotations.

```python
ZERO_SHOT_PROMPT = """Classify the following financial news sentence
as "positive", "negative", or "neutral" sentiment.

Respond with ONLY the label, nothing else.

Sentence: {text}
Label:"""

def classify_zero_shot(texts, model=LLM_MODEL):
    """
    Performs zero-shot classification using an OpenAI LLM.
    Measures predictions, total cost, and elapsed time.
    """
    predictions = []
    total_cost = 0
    start_time = time.time()

    for i, text in enumerate(texts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": ZERO_SHOT_PROMPT.format(text=text)}],
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
            # If error, no cost to add from API call

    elapsed_time = time.time() - start_time
    return predictions, total_cost, elapsed_time

# Evaluate Zero-Shot on a subset of the test data for speed
test_subset = test_df.sample(200, random_state=RANDOM_STATE) # Using a subset for faster iteration
print(f"Evaluating Zero-Shot on {len(test_subset)} samples...")
preds_zero_shot, cost_zero_shot, time_zero_shot = classify_zero_shot(test_subset['text'].tolist())
print(f"Zero-Shot classification complete in {time_zero_shot:.2f} seconds.")
```

### Few-Shot and Chain-of-Thought (CoT) Prompting

Few-shot prompting provides the LLM with a few examples of the task, helping it to understand the desired output format and style. This is crucial for consistency in our financial reports. Chain-of-Thought (CoT) prompting takes this further by including reasoning steps in the examples. This forces the LLM to "think" before answering, which is particularly beneficial for ambiguous financial language where context and logical inference are key.

The core idea of Chain-of-Thought (CoT) prompting is to guide the LLM through a series of intermediate reasoning steps before arriving at the final answer. This can be represented as:
$Input \rightarrow Thought\ 1 \rightarrow Thought\ 2 \rightarrow ... \rightarrow Final\ Answer$
For financial sentiment, this transforms from $Input \rightarrow Label$ to $Input \rightarrow Reasoning \rightarrow Label$. For example, a CoT prompt might guide the model to explain "This mentions improvement/growth/beat, indicating positive sentiment." before outputting "positive."

```python
def build_few_shot_prompt(text, examples_df, use_cot=False):
    """Constructs a few-shot prompt with optional chain-of-thought."""
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
        examples_text += f"Label: {row['label_name']}\n\n"

    prompt = f"""Classify financial news sentences as "positive", "negative", or "neutral".

Examples:
{examples_text}
Sentence: {text}
"""
    if use_cot:
        prompt += "Reasoning:"
    else:
        prompt += "Label:"
    return prompt

def classify_few_shot(texts, examples_df, use_cot=False, model=LLM_MODEL):
    """
    Performs few-shot classification with optional CoT using an OpenAI LLM.
    Measures predictions, total cost, and elapsed time.
    """
    predictions = []
    total_cost = 0
    start_time = time.time()

    for i, text in enumerate(texts):
        try:
            prompt_content = build_few_shot_prompt(text, examples_df, use_cot)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_content}],
                temperature=0.0,
                max_tokens=50 if use_cot else 5 # More tokens needed for reasoning steps
            )
            output = response.choices[0].message.content.strip().lower()

            # Extract label from CoT output if used
            if use_cot:
                extracted_label = "neutral" # Default
                for label in ['positive', 'negative', 'neutral']:
                    if label in output.split('\n')[-1]: # Look at the last line for the label
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

# --- Few-Shot Approach (B) ---
print(f"\nEvaluating Few-Shot on {len(test_subset)} samples...")
preds_few_shot, cost_few_shot, time_few_shot = classify_few_shot(
    test_subset['text'].tolist(), few_shot_examples_df, use_cot=False
)
print(f"Few-Shot classification complete in {time_few_shot:.2f} seconds.")

# --- Few-Shot + Chain-of-Thought (CoT) Approach (C) ---
print(f"\nEvaluating Few-Shot + CoT on {len(test_subset)} samples...")
preds_cot, cost_cot, time_cot = classify_few_shot(
    test_subset['text'].tolist(), few_shot_examples_df, use_cot=True
)
print(f"Few-Shot + CoT classification complete in {time_cot:.2f} seconds.")
```

---

## 3. Retrieval-Augmented Generation (RAG) for Contextual Prompting

RAG represents a "smart few-shot" approach. Instead of providing a fixed set of examples, RAG dynamically retrieves the most semantically similar labeled examples from our domain data to augment the LLM's prompt. This ensures that the context provided is highly relevant to the specific query, which is particularly beneficial when dealing with diverse financial texts (e.g., news about different sectors or types of financial events). This helps the LLM ground its reasoning in specific, similar scenarios, improving accuracy without modifying the model itself.

The cosine similarity between two non-zero vectors $\mathbf{A}$ and $\mathbf{B}$ is defined as:
$$ \text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} $$
where $\mathbf{A} \cdot \mathbf{B}$ is the dot product, and $\|\mathbf{A}\|$ and $\|\mathbf{B}\|$ are the magnitudes (L2 norms) of vectors $\mathbf{A}$ and $\mathbf{B}$, respectively. This measures the cosine of the angle between them, indicating semantic similarity.

```python
# Embed all training examples using a Sentence Transformer model
print("\nInitializing SentenceTransformer and embedding training data for RAG...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
# Use the full train_df for RAG example retrieval
train_embeddings = sbert_model.encode(train_df['text'].tolist(), convert_to_tensor=True)
print(f"Training data embeddings created. Shape: {train_embeddings.shape}")

def classify_rag(texts, train_df, train_embeddings, k=5, model=LLM_MODEL):
    """
    Performs RAG-augmented classification by retrieving top-k similar examples
    from the training data to augment the LLM prompt.
    """
    predictions = []
    total_cost = 0
    start_time = time.time()

    for i, text in enumerate(texts):
        try:
            # 1. Embed the query text
            query_embedding = sbert_model.encode([text], convert_to_tensor=True)

            # 2. Retrieve k most similar training examples using cosine similarity
            # Calculate cosine similarity with all training embeddings
            from sklearn.metrics.pairwise import cosine_similarity # Ensure this is imported if not already
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
            response = client.chat.completions.create(
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

# --- RAG-Augmented Classification Approach (D) ---
print(f"\nEvaluating RAG-Augmented Classification (k=5) on {len(test_subset)} samples...")
preds_rag, cost_rag, time_rag = classify_rag(
    test_subset['text'].tolist(), train_df, train_embeddings, k=5
)
print(f"RAG classification complete in {time_rag:.2f} seconds.")
```

---

## 4. Leveraging a Fine-Tuned Domain-Specific Model

While prompting strategies are flexible, sometimes the highest accuracy, lowest latency, and most stringent privacy controls are paramount. This is where fine-tuning a smaller, domain-specific model on our proprietary labeled data becomes indispensable. Fine-tuning allows the model weights to encode specific domain patterns, nuances, and vocabulary, creating a "moat" of intellectual property that generic LLMs cannot replicate. For example, a fine-tuned FinBERT model understands financial sentiment more accurately than a general-purpose model.

For this lab, we will utilize a pre-trained `FinBERT` model, which has been fine-tuned on financial data, to simulate the performance of a fine-tuned small model. This approach typically offers significantly lower inference costs and latency as it runs locally, and ensures data privacy by keeping sensitive information within the firm's infrastructure.

```python
# --- Fine-Tuned Small Model Approach (E) ---
print("\nInitializing Fine-Tuned FinBERT model...")
try:
    # Use ProsusAI/finbert for financial sentiment analysis directly
    # It outputs 'positive', 'negative', 'neutral' which aligns with our task
    finbert_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    finbert_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
    
    # Create a pipeline for text classification
    # Use GPU if available (device=0), otherwise CPU (device=-1)
    ft_pipeline = pipeline("sentiment-analysis",
                           model=finbert_model,
                           tokenizer=finbert_tokenizer,
                           device=0 if torch.cuda.is_available() else -1)
    print("FinBERT model loaded successfully.")

    def classify_finetuned(texts, pipeline_model):
        """
        Classifies texts using a fine-tuned local model (e.g., FinBERT).
        Measures predictions and elapsed time. Cost is considered 0 for local inference.
        """
        start_time = time.time()
        # The pipeline handles batching efficiently
        preds_raw = pipeline_model(texts, batch_size=32)
        
        # Extract labels and normalize to lowercase
        predictions = [p['label'].lower() for p in preds_raw]
        
        elapsed_time = time.time() - start_time
        return predictions, elapsed_time

    print(f"Evaluating Fine-Tuned FinBERT on {len(test_subset)} samples...")
    preds_finetuned, time_finetuned = classify_finetuned(test_subset['text'].tolist(), ft_pipeline)
    cost_finetuned = 0.0 # Local model, no per-query API cost
    print(f"Fine-Tuned FinBERT classification complete in {time_finetuned:.2f} seconds.")

except Exception as e:
    print(f"Could not load FinBERT or run fine-tuned pipeline: {e}")
    print("Defaulting Fine-Tuned predictions to 'neutral' for all samples and setting costs/times to placeholders.")
    preds_finetuned = ["neutral"] * len(test_subset)
    time_finetuned = 0.1 # Placeholder
    cost_finetuned = 0.0 # Still no API cost

```

---

## 5. Comprehensive Performance Evaluation and Decision Matrix

Now that we've implemented and executed all five LLM strategies, it's time to consolidate their performance across the critical dimensions: accuracy (F1-score), cost (per-query & setup), latency, labeled data requirements, and privacy risk. This comprehensive comparison matrix is vital for an investment professional to make an informed, strategic decision about which LLM strategy to deploy.

We'll calculate weighted F1-score, which is suitable for multi-class classification and accounts for class imbalance, along with per-query cost and latency.

```python
# True labels for evaluation
y_true = test_subset['label_name'].tolist()

# Ensure all prediction lists have the same length as y_true
# Fill with 'neutral' if any prediction list is shorter due to errors
def pad_predictions(preds, target_len):
    if len(preds) < target_len:
        return preds + ['neutral'] * (target_len - len(preds))
    return preds

preds_zero_shot = pad_predictions(preds_zero_shot, len(y_true))
preds_few_shot = pad_predictions(preds_few_shot, len(y_true))
preds_cot = pad_predictions(preds_cot, len(y_true))
preds_rag = pad_predictions(preds_rag, len(y_true))
preds_finetuned = pad_predictions(preds_finetuned, len(y_true))


# Aggregate results for all approaches
results = []
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
        'n_labeled': len(train_df), 'privacy_risk': 'High' # RAG still uses external LLM
                                                           # but uses local data for retrieval
                                                           # which reduces the risk
                                                           # however, the query leaves the firm
                                                           # so it is still high.
        , 'setup_cost': 0.0
    },
    'E: Fine-Tuned (FinBERT)': {
        'preds': preds_finetuned, 'cost': cost_finetuned, 'time': time_finetuned,
        'n_labeled': len(train_df), 'privacy_risk': 'None', 'setup_cost': 5.0 # Estimated cost for fine-tuning setup
    }
}

print("\n--- Calculating Performance Metrics ---")
per_class_f1_scores = {}

for name, data in approaches_data.items():
    preds = data['preds']
    total_cost = data['cost']
    elapsed_time = data['time']
    n_labeled = data['n_labeled']
    privacy_risk = data['privacy_risk']
    setup_cost = data['setup_cost']

    # Clean predictions: ensure they are in expected labels, default to 'neutral' if not parseable
    clean_preds = []
    expected_labels = ['positive', 'negative', 'neutral']
    for p in preds:
        cleaned_p = p.strip().lower()
        if cleaned_p not in expected_labels:
            clean_preds.append('neutral') # Default for unparseable or unexpected output
        else:
            clean_preds.append(cleaned_p)

    # Calculate metrics
    f1 = f1_score(y_true, clean_preds, average='weighted', labels=expected_labels)
    acc = accuracy_score(y_true, clean_preds)
    
    # Cost per query: Total cost / number of predictions
    # Handle division by zero if total_cost is 0 (for local models)
    cost_per_query = total_cost / len(y_true) if total_cost > 0 else 0.0

    # Latency: Total elapsed time / number of predictions (convert to ms)
    latency_ms_per_query = (elapsed_time / len(y_true)) * 1000

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

    # Store per-class F1 for visualization
    per_class_f1 = f1_score(y_true, clean_preds, average=None, labels=expected_labels)
    per_class_f1_scores[name] = dict(zip(expected_labels, per_class_f1))

results_df = pd.DataFrame(results)

print("\n--- FIVE-WAY COMPARISON MATRIX ---")
print("=" * 80)
print(results_df.to_string(index=False))
print("=" * 80)

# Prepare per-class F1 for plotting
per_class_f1_df = pd.DataFrame(per_class_f1_scores).T
per_class_f1_df.index.name = 'Approach'
print("\n--- Per-Class F1 Scores ---")
print(per_class_f1_df.to_string())
```

### Explanation of Execution

The comparison matrix provides a snapshot of each strategy's performance across key dimensions. For an investment professional, this immediately highlights trade-offs:
*   **Accuracy (Weighted F1):** Fine-tuned models generally lead, followed by RAG and CoT, demonstrating the value of domain-specific knowledge or guided reasoning.
*   **Cost per Query:** API-based LLMs incur per-query costs, which can quickly accumulate. Local fine-tuned models have negligible marginal costs.
*   **Setup Cost:** Fine-tuning requires an upfront investment in data labeling and training, which is reflected in the setup cost.
*   **Latency:** Local fine-tuned models are orders of magnitude faster than API calls, critical for real-time applications.
*   **Labeled Data Needed:** Zero-shot needs none, few-shot needs a small sample, while RAG and fine-tuning leverage the entire training dataset.
*   **Privacy Risk:** A crucial factor for financial firms, where data leaving firm control (API calls) poses a higher risk than local model inference.

This matrix forms the basis for making strategic decisions about LLM deployment based on specific project requirements and constraints.

---

## 6. Strategic Cost-Accuracy Trade-offs and Crossover Analysis

Beyond raw performance numbers, understanding the long-term cost implications of each LLM strategy is paramount for a financial firm. The "total cost of ownership" over time, especially with varying query volumes, dictates when an upfront investment in fine-tuning becomes more economically viable than continuous API calls.

The total cost of ownership for $N$ queries can be modeled as:
For API-based prompting: $C_{\text{prompt}}(N) = N \cdot C_{\text{query}}$
For fine-tuning: $C_{\text{FT}}(N) = C_{\text{setup}} + N \cdot C_{\text{inference}}$
Where $N$ is the number of queries, $C_{\text{query}}$ is the cost per API query, $C_{\text{setup}}$ is the fixed setup cost for fine-tuning, and $C_{\text{inference}}$ is the marginal inference cost (often negligible or zero for local models).

The cost crossover point $N^*$ is the query volume at which fine-tuning becomes more cost-effective than API-based prompting. It is found by setting $C_{\text{prompt}}(N) = C_{\text{FT}}(N)$:
$$ N^* \cdot C_{\text{query}} = C_{\text{setup}} + N^* \cdot C_{\text{inference}} $$
$$ N^* (C_{\text{query}} - C_{\text{inference}}) = C_{\text{setup}} $$
$$ N^* = \frac{C_{\text{setup}}}{C_{\text{query}} - C_{\text{inference}}} $$
For local fine-tuned models, we typically assume $C_{\text{inference}} = 0$.

```python
# --- Cost Crossover Analysis ---
# Using the cost per query from the Few-Shot approach as a representative API cost
# This is a per-item cost, so for cumulative cost, we multiply by query volume.
# For simplicity, we assume fine-tuned model inference cost C_inference is negligible ($0)
prompt_per_query_cost_avg = results_df[results_df['Approach'] == 'B: Few-Shot']['Cost per Query ($)'].iloc[0]
ft_setup_cost = results_df[results_df['Approach'] == 'E: Fine-Tuned (FinBERT)']['Setup Cost ($)'].iloc[0]
finetuned_per_query_inference_cost = 0.0 # Assuming negligible inference cost for local model

# Calculate crossover point
crossover_queries = ft_setup_cost / (prompt_per_query_cost_avg - finetuned_per_query_inference_cost) \
                    if (prompt_per_query_cost_avg - finetuned_per_query_inference_cost) > 0 \
                    else float('inf')

print("\n--- Cost Crossover Analysis ---")
print(f"Average API Prompting Cost per Query: ${prompt_per_query_cost_avg:.6f}")
print(f"Fine-Tuning Setup Cost: ${ft_setup_cost:.2f}")
print(f"Fine-Tuned Model Inference Cost per Query: ${finetuned_per_query_inference_cost:.6f}")
print(f"Cost Crossover: Fine-tuning becomes cheaper after {crossover_queries:.0f} queries.")

# Financial Interpretation
# Example: If an analyst runs 20 classifications per day
queries_per_day = 20
crossover_days = crossover_queries / queries_per_day
crossover_months = crossover_days / 30
print(f"At {queries_per_day} queries/day: crossover in {crossover_days:.0f} days ({crossover_months:.1f} months).")

# --- Plot Cost Crossover ---
max_queries = int(crossover_queries * 3) if crossover_queries != float('inf') else 50000 # Extend plot beyond crossover
if max_queries < 1000: # Ensure at least a reasonable range for visualization
    max_queries = 10000

queries = np.arange(0, max_queries, 100) # Range of query volumes
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
plt.savefig('cost_crossover.png', dpi=150)
plt.show()

# --- Cost-Accuracy Pareto Frontier ---
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
plt.savefig('cost_accuracy_pareto_frontier.png', dpi=150)
plt.show()
```

### Explanation of Execution

The cost crossover plot visually demonstrates the long-term financial implications. For a quant analyst, this is a clear signal: if a task is a one-off analysis (low query volume), prompting is cheaper. If it's a daily production pipeline with high query volume, fine-tuning offers significant cost savings over time, in addition to benefits in latency and privacy.

The Cost-Accuracy Pareto Frontier plot helps identify the "efficient frontier" of LLM strategies – those that offer the highest accuracy for a given cost. Approaches on this frontier are generally preferred as they represent optimal trade-offs. Strategies far from the frontier might be sub-optimal choices. For instance, a high-cost, low-accuracy approach would be clearly undesirable.

---

## 7. Informed Decision-Making for LLM Deployment in Finance

Bringing all the analysis together, an investment professional needs a clear framework for selecting the optimal LLM strategy. This involves considering accuracy, costs, latency, data availability, and privacy concerns in the context of specific business requirements.

### Per-Class F1 and Latency Comparisons

These visualizations highlight where each approach excels or struggles, particularly for specific sentiment categories, and the real-world operational efficiency.

```python
# --- Per-Class F1 Score Comparison ---
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
plt.savefig('per_class_f1_comparison.png', dpi=150)
plt.show()

# --- Latency Comparison ---
plt.figure(figsize=(10, 6))
sns.barplot(x='Approach', y='Latency (ms/query)', data=results_df.sort_values('Latency (ms/query)'), palette='cividis')
plt.ylabel('Latency (milliseconds per query)')
plt.xlabel('LLM Approach')
plt.title('Query Latency Across LLM Strategies')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('latency_comparison.png', dpi=150)
plt.show()
```

### CFA ESG Benchmark Reproduction

The CFA Institute's research on ESG materiality classification provides valuable empirical evidence for LLM performance on a more nuanced financial task. While our lab focuses on financial sentiment, we can reproduce their findings conceptually to illustrate the significant accuracy gains from fine-tuning on complex, domain-specific tasks.

```python
# --- CFA ESG Benchmark Reproduction ---
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
plt.savefig('cfa_esg_benchmark.png', dpi=150)
plt.show()
```

### Decision Flowchart Logic for LLM Strategy Selection

Based on all the analysis, an investment professional can use a simplified decision flowchart to guide their LLM deployment strategy:

1.  **Do you have >500 labeled examples?**
    *   **No:** Start with **Prompting** (zero-shot or few-shot). Go to Q3.
    *   **Yes:** Go to Q2.

2.  **Will you run >10,000 queries per month/year?** (Consider your query volume based on cost crossover analysis)
    *   **No:** Use **RAG-augmented prompting** (best accuracy without model modification).
    *   **Yes:** **Fine-tune** a local model (amortized cost drops below prompting).

3.  **Is accuracy >80% required for this task?**
    *   **No:** **Zero-shot or few-shot** prompting might be sufficient.
    *   **Yes:** Consider **Few-shot + CoT** or **RAG**. If still insufficient, collect more labels and pursue **fine-tuning**.

4.  **Can data leave the firm's control (privacy/compliance)?**
    *   **No:** Implement a **fine-tuned local model** (e.g., LoRA on Llama/Mistral) or **RAG with a local LLM**.
    *   **Yes:** **API prompting** is acceptable.

This decision-making framework allows us to strategically choose an LLM deployment path that balances our need for accuracy, cost-efficiency, low latency, and strict privacy controls, directly impacting our firm's investment research and risk management capabilities.

### Strategic Implications for the Financial Firm

Our analysis reveals key takeaways for strategic LLM deployment in a financial context:

*   **"Start with prompting, graduate to fine-tuning":** For initial feasibility and rapid prototyping, prompting strategies are invaluable. As tasks become critical, repetitive, and require higher accuracy or strict privacy, graduating to RAG or fine-tuning becomes economically and strategically sensible.
*   **Fine-tuning creates a "moat":** Leveraging proprietary labeled data to fine-tune models creates unique intellectual property. This domain-specific advantage cannot be easily replicated by competitors relying solely on generic LLMs, fostering a competitive edge in investment strategies.
*   **Nuance gradient determines the approach:** For simple, well-defined sentiment tasks, well-crafted prompts can achieve good performance. For highly nuanced tasks like ESG materiality classification, where subtle context shifts meanings, fine-tuning provides substantial accuracy gains.
*   **RAG as the middle ground:** When labeled data exists but is insufficient for full fine-tuning, or when model modification is constrained, RAG offers a powerful way to ground LLMs in domain-specific context without altering model weights.

By systematically evaluating these LLM strategies, we as investment professionals can make informed decisions that optimize resource allocation, manage risk effectively, and enhance the firm's analytical capabilities through strategic AI adoption.
