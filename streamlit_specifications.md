
# Streamlit Application Specification: Fine-Tuning vs. Prompt Engineering for Financial Text Classification

## 1. Application Overview

### Purpose of the Application

This Streamlit application serves as an interactive platform for CFA Charterholders and Investment Professionals to understand and compare various Large Language Model (LLM) deployment strategies for financial text classification. It provides a data-driven framework to evaluate five distinct approaches—Zero-Shot, Few-Shot, Few-Shot + Chain-of-Thought (CoT), Retrieval-Augmented Generation (RAG), and Fine-Tuned Small Model—across critical dimensions: Accuracy (F1-score), Cost, Latency, Data Requirements, and Privacy Risk. The goal is to equip financial professionals with the insights needed to make informed decisions on optimal LLM strategy for their specific use cases, balancing performance with practical constraints.

### High-Level Story Flow of the Application

The application guides the user, a Quant Analyst, through a structured workflow, simulating a real-world decision-making process for LLM deployment:

1.  **Introduction**: The user is introduced to the challenge of selecting an optimal LLM strategy and the five approaches to be evaluated for financial sentiment analysis.
2.  **Data Preparation**: The application facilitates loading and preparing the `Financial PhraseBank` dataset, demonstrating how the raw data is transformed and split for analysis.
3.  **Approach Execution (Step-by-Step)**: The user can individually run each of the five LLM classification approaches. For each approach, the app will execute the classification, calculate its performance metrics (accuracy, cost, latency), and store these results for later comparison.
    *   **Zero-Shot Prompting**: Direct instruction to `gpt-4o`.
    *   **Few-Shot & CoT Prompting**: `gpt-4o` with examples and guided reasoning.
    *   **RAG-Augmented Classification**: `gpt-4o` with dynamically retrieved context from `SentenceTransformer`.
    *   **Fine-Tuned Small Model**: Simulation using `FinBERT` for local inference.
4.  **Comprehensive Evaluation**: Once all approaches are executed, the user proceeds to a comprehensive comparison. This section consolidates all measured performance metrics into a five-way comparison matrix and visualizes key trade-offs through interactive plots (Cost-Accuracy Pareto Frontier, Cost Crossover Plot, Per-Class F1, Latency, and CFA ESG Benchmark reproduction).
5.  **Decision Framework & Strategic Implications**: Finally, the application presents a decision flowchart and strategic implications for financial firms, helping the user translate the quantitative analysis into actionable insights for LLM deployment decisions. This allows the learner to apply the concepts to their specific firm's needs.

## 2. Code Requirements

### Import Statement

The Streamlit application (`app.py`) will begin with the following import:

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Necessary for clearing plots if source.py plots do not return figures
import seaborn as sns # Necessary for clearing plots if source.py plots do not return figures
from source import * # Imports all functions, variables, and global execution from source.py
```

**Note on `source.py` Execution**: Due to the constraint "All application logic already exists in `source.py`" and "do not redefine, rewrite, stub, or duplicate them", it is assumed that importing `source.py` will execute all its top-level code blocks sequentially. This means that:
*   Global constants (`RANDOM_STATE`, `LLM_MODEL`, `GPT_COST_PROMPT_PER_TOKEN`, `GPT_COST_COMPLETION_PER_TOKEN`, `ZERO_SHOT_PROMPT`, `client`) are initialized.
*   The entire data loading, preparation, classification (for all five approaches), performance evaluation, and plotting logic will execute at application startup, generating all intermediate `preds_X`, `cost_X`, `time_X` variables, `results_df`, `per_class_f1_df`, `cfa_benchmarks`, and saving all required plot `.png` files to the local directory.
*   The `st.button`s in the Streamlit app will therefore primarily serve to `display` these pre-computed results, rather than triggering new computations. This design ensures strict adherence to the constraint against modifying or re-orchestrating `source.py`'s execution flow.

### `st.session_state` Usage

`st.session_state` will be used primarily for managing application navigation and displaying a more controlled, step-by-step "workflow" experience, even if computations are pre-executed.

*   **Initialization**:
    ```python
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Introduction"
    ```
*   **Update**:
    *   The `current_page` key will be updated based on the user's selection in the sidebar `st.selectbox`.
*   **Read Across Pages**:
    *   `st.session_state.current_page` will dictate which content block is rendered in the main area of the application.
    *   No other complex state management is needed for intermediate computation results, as all results (dataframes, predictions, plot images) are assumed to be globally available after `source.py` is imported and executed.

### UI Interactions and `source.py` Function Calls

The application will simulate a multi-page experience using a sidebar `st.selectbox`. Each "page" will display specific content and, where relevant, `st.button`s that, when clicked, will display the results (dataframes, metrics, images) that were pre-computed by the `source.py` script on startup.

#### Sidebar Navigation

```python
st.sidebar.title("Navigation")
page_selection = st.sidebar.selectbox("Go to", [
    "Introduction",
    "1. Data Preparation",
    "2. Zero-Shot Prompting",
    "3. Few-Shot & CoT Prompting",
    "4. RAG-Augmented Classification",
    "5. Fine-Tuned Small Model",
    "6. Comprehensive Evaluation",
    "7. Decision Framework & Strategic Implications"
])
st.session_state.current_page = page_selection
```

#### Page: "Introduction"

**Markdown:**
```python
st.markdown(f"# Fine-Tuning vs. Prompt Engineering: Optimal LLM Strategy for Financial Text Classification")
st.markdown(f"## Introduction: Navigating LLM Deployment as a Quant Analyst")
st.markdown(f"""
As a CFA Charterholder and investment professional at a leading financial firm, my daily responsibilities often involve distilling critical insights from vast amounts of unstructured financial text. Whether it's analyzing market sentiment from news headlines, identifying material ESG factors from corporate reports, or classifying financial documents, the efficiency and accuracy of these tasks directly impact investment research, risk management, and client reporting.

The advent of Large Language Models (LLMs) offers a powerful toolkit, but deciding on the optimal deployment strategy is a complex challenge. Should we rely on off-the-shelf general LLMs with clever prompting (prompt engineering), or invest in fine-tuning smaller, specialized models on our proprietary domain data? Each path has distinct trade-offs in terms of accuracy, cost, latency, data requirements, and critical privacy considerations.

This application outlines a practical, data-driven framework to systematically evaluate five distinct LLM-based approaches for a financial text classification task:
1.  **Zero-Shot Prompting:** Leveraging a general LLM with only a direct instruction.
2.  **Few-Shot Prompting:** Providing the general LLM with a few labeled examples in the prompt.
3.  **Few-Shot + Chain-of-Thought (CoT) Prompting:** Augmenting few-shot prompting with reasoning steps to guide the LLM.
4.  **Retrieval-Augmented Generation (RAG):** Dynamically retrieving and including the most similar labeled examples from our domain data in the prompt.
5.  **Fine-Tuned Small Model:** Using a domain-specific model fine-tuned on a labeled financial dataset.

My goal is to measure the performance across these dimensions for each approach and construct a decision framework that guides optimal LLM deployment within our firm, optimizing resource allocation, managing risk, and ultimately enhancing our competitive advantage.
""")
```

#### Page: "1. Data Preparation"

**Markdown:**
```python
st.markdown(f"## 1. Environment Setup and Data Preparation for Financial Sentiment Analysis")
st.markdown(f"Our first step is to prepare our Python environment and load the necessary financial text data. We'll use the `Financial PhraseBank` dataset, a standard benchmark for financial sentiment analysis, as a proxy for the classification tasks we perform daily.")

st.markdown(f"### Import Dependencies and Set Up API Key")
st.info("Dependencies are loaded and OpenAI API client initialized upon application startup.")

st.markdown(f"### Load and Prepare Financial PhraseBank Dataset")
st.markdown(f"We've loaded the `Financial PhraseBank` dataset, which contains sentences from financial news categorized into positive, neutral, or negative sentiment. This serves as our labeled financial text classification task.")

if st.button("Show Data Statistics"):
    st.markdown(f"**Dataset statistics:**")
    st.write(f"Total dataset size: {len(df)}")
    st.write(f"Training set: {len(train_df)} (for fine-tuning and RAG examples)")
    st.write(f"Test set: {len(test_df)} (shared evaluation for all approaches)")
    st.write(f"Few-shot examples (3 per class): {len(few_shot_examples_df)} (for prompting)")
    st.write(f"Class distribution in full dataset: {df['label_name'].value_counts().to_dict()}")
    st.markdown(f"**Sample Few-Shot Examples:**")
    st.dataframe(few_shot_examples_df[['text', 'label_name']].reset_index(drop=True))
```
**`source.py` Integration**:
*   `df`, `train_df`, `test_df`, `few_shot_examples_df` are global variables from `source.py` after import.

#### Page: "2. Zero-Shot Prompting"

**Markdown:**
```python
st.markdown(f"## 2. Prompt Engineering Strategies with a General LLM")
st.markdown(f"As an investment professional, my initial thought when faced with a new text classification task is often to leverage an existing, powerful LLM. Prompt engineering allows me to guide these general models without needing to retrain them. We'll explore three increasingly sophisticated prompting strategies: Zero-Shot, Few-Shot, and Few-Shot with Chain-of-Thought (CoT). These approaches are quick to implement and offer immediate insights into the LLM's capabilities on financial text.")

st.markdown(f"### Zero-Shot Prompting: Direct Instruction")
st.markdown(f"Zero-shot prompting is the simplest approach, providing the LLM with only the task instruction and the input text. It's useful for quickly gauging the model's inherent understanding of a domain. For financial sentiment, we need the model to recognize nuanced positive, neutral, or negative connotations.")

if st.button("Show Zero-Shot Classification Results"):
    st.markdown(f"Zero-Shot classification was completed upon app startup.")
    st.markdown(f"**Metrics for Zero-Shot:**")
    zero_shot_row = results_df[results_df['Approach'] == 'A: Zero-Shot'].iloc[0]
    st.write(f"Weighted F1: {zero_shot_row['Weighted F1']:.2f}")
    st.write(f"Cost per Query: ${zero_shot_row['Cost per Query ($)']:.6f}")
    st.write(f"Latency (ms/query): {zero_shot_row['Latency (ms/query)']:.2f} ms")
```
**`source.py` Integration**:
*   `results_df` is a global DataFrame from `source.py` after import.

#### Page: "3. Few-Shot & CoT Prompting"

**Markdown:**
```python
st.markdown(f"### Few-Shot and Chain-of-Thought (CoT) Prompting")
st.markdown(f"Few-shot prompting provides the LLM with a few examples of the task, helping it to understand the desired output format and style. This is crucial for consistency in our financial reports. Chain-of-Thought (CoT) prompting takes this further by including reasoning steps in the examples. This forces the LLM to 'think' before answering, which is particularly beneficial for ambiguous financial language where context and logical inference are key.")

st.markdown(r"The core idea of Chain-of-Thought (CoT) prompting is to guide the LLM through a series of intermediate reasoning steps before arriving at the final answer. This can be represented as:")
st.markdown(r"$$ Input \rightarrow Thought\ 1 \rightarrow Thought\ 2 \rightarrow ... \rightarrow Final\ Answer $$")
st.markdown(r"where $Input$ is the financial sentence, $Thought$ represents intermediate reasoning steps, and $Final\ Answer$ is the sentiment label.")
st.markdown(r"For financial sentiment, this transforms from $Input \rightarrow Label$ to $Input \rightarrow Reasoning \rightarrow Label$. For example, a CoT prompt might guide the model to explain 'This mentions improvement/growth/beat, indicating positive sentiment.' before outputting 'positive.'")

if st.button("Show Few-Shot & CoT Classification Results"):
    st.markdown(f"Few-Shot and CoT classifications were completed upon app startup.")
    st.markdown(f"**Metrics for Few-Shot:**")
    few_shot_row = results_df[results_df['Approach'] == 'B: Few-Shot'].iloc[0]
    st.write(f"Weighted F1: {few_shot_row['Weighted F1']:.2f}")
    st.write(f"Cost per Query: ${few_shot_row['Cost per Query ($)']:.6f}")
    st.write(f"Latency (ms/query): {few_shot_row['Latency (ms/query)']:.2f} ms")

    st.markdown(f"**Metrics for Few-Shot + CoT:**")
    cot_row = results_df[results_df['Approach'] == 'C: Few-Shot+CoT'].iloc[0]
    st.write(f"Weighted F1: {cot_row['Weighted F1']:.2f}")
    st.write(f"Cost per Query: ${cot_row['Cost per Query ($)']:.6f}")
    st.write(f"Latency (ms/query): {cot_row['Latency (ms/query)']:.2f} ms")
```
**`source.py` Integration**:
*   `results_df` is a global DataFrame from `source.py` after import.

#### Page: "4. RAG-Augmented Classification"

**Markdown:**
```python
st.markdown(f"## 3. Retrieval-Augmented Generation (RAG) for Contextual Prompting")
st.markdown(f"RAG represents a 'smart few-shot' approach. Instead of providing a fixed set of examples, RAG dynamically retrieves the most semantically similar labeled examples from our domain data to augment the LLM's prompt. This ensures that the context provided is highly relevant to the specific query, which is particularly beneficial when dealing with diverse financial texts (e.g., news about different sectors or types of financial events). This helps the LLM ground its reasoning in specific, similar scenarios, improving accuracy without modifying the model itself.")

st.markdown(r"The cosine similarity between two non-zero vectors $\mathbf{{A}}$ and $\mathbf{{B}}$ is defined as:")
st.markdown(r"$$ \text{{similarity}} = \cos(\theta) = \frac{{\mathbf{{A}} \cdot \mathbf{{B}}}}{{\|\mathbf{{A}}\| \|\mathbf{{B}}\|}} $$")
st.markdown(r"where $\mathbf{{A}} \cdot \mathbf{{B}}$ is the dot product, and $\|\mathbf{{A}}\| $ and $\|\mathbf{{B}}\| $ are the magnitudes (L2 norms) of vectors $\mathbf{{A}}$ and $\mathbf{{B}}$, respectively. This measures the cosine of the angle between them, indicating semantic similarity.")

if st.button("Show RAG-Augmented Classification Results"):
    st.markdown(f"RAG-Augmented Classification was completed upon app startup.")
    st.markdown(f"**Metrics for RAG-Augmented:**")
    rag_row = results_df[results_df['Approach'] == 'D: RAG-Augmented'].iloc[0]
    st.write(f"Weighted F1: {rag_row['Weighted F1']:.2f}")
    st.write(f"Cost per Query: ${rag_row['Cost per Query ($)']:.6f}")
    st.write(f"Latency (ms/query): {rag_row['Latency (ms/query)']:.2f} ms")
```
**`source.py` Integration**:
*   `results_df` is a global DataFrame from `source.py` after import.

#### Page: "5. Fine-Tuned Small Model"

**Markdown:**
```python
st.markdown(f"## 4. Leveraging a Fine-Tuned Domain-Specific Model")
st.markdown(f"While prompting strategies are flexible, sometimes the highest accuracy, lowest latency, and most stringent privacy controls are paramount. This is where fine-tuning a smaller, domain-specific model on our proprietary labeled data becomes indispensable. Fine-tuning allows the model weights to encode specific domain patterns, nuances, and vocabulary, creating a 'moat' of intellectual property that generic LLMs cannot replicate. For example, a fine-tuned FinBERT model understands financial sentiment more accurately than a general-purpose model.")
st.markdown(f"For this lab, we utilized a pre-trained `FinBERT` model, which has been fine-tuned on financial data, to simulate the performance of a fine-tuned small model. This approach typically offers significantly lower inference costs and latency as it runs locally, and ensures data privacy by keeping sensitive information within the firm's infrastructure.")

if st.button("Show Fine-Tuned Model Classification Results"):
    st.markdown(f"Fine-Tuned FinBERT classification was completed upon app startup.")
    st.markdown(f"**Metrics for Fine-Tuned (FinBERT):**")
    finetuned_row = results_df[results_df['Approach'] == 'E: Fine-Tuned (FinBERT)'].iloc[0]
    st.write(f"Weighted F1: {finetuned_row['Weighted F1']:.2f}")
    st.write(f"Cost per Query: ${finetuned_row['Cost per Query ($)']:.6f}")
    st.write(f"Latency (ms/query): {finetuned_row['Latency (ms/query)']:.2f} ms")
```
**`source.py` Integration**:
*   `results_df` is a global DataFrame from `source.py` after import.

#### Page: "6. Comprehensive Evaluation"

**Markdown:**
```python
st.markdown(f"## 5. Comprehensive Performance Evaluation and Decision Matrix")
st.markdown(f"Now that we've implemented and executed all five LLM strategies, it's time to consolidate their performance across the critical dimensions: accuracy (F1-score), cost (per-query & setup), latency, labeled data requirements, and privacy risk. This comprehensive comparison matrix is vital for an investment professional to make an informed, strategic decision about which LLM strategy to deploy.")
st.markdown(f"We've calculated weighted F1-score, which is suitable for multi-class classification and accounts for class imbalance, along with per-query cost and latency.")

if st.button("Show Comparison Matrix and Visualizations"):
    st.markdown(f"### FIVE-WAY COMPARISON MATRIX")
    st.dataframe(results_df)
    st.markdown(f"The comparison matrix provides a snapshot of each strategy's performance across key dimensions. For an investment professional, this immediately highlights trade-offs:")
    st.markdown(f"""
    *   **Accuracy (Weighted F1):** Fine-tuned models generally lead, followed by RAG and CoT, demonstrating the value of domain-specific knowledge or guided reasoning.
    *   **Cost per Query:** API-based LLMs incur per-query costs, which can quickly accumulate. Local fine-tuned models have negligible marginal costs.
    *   **Setup Cost:** Fine-tuning requires an upfront investment in data labeling and training, which is reflected in the setup cost.
    *   **Latency:** Local fine-tuned models are orders of magnitude faster than API calls, critical for real-time applications.
    *   **Labeled Data Needed:** Zero-shot needs none, few-shot needs a small sample, while RAG and fine-tuning leverage the entire training dataset.
    *   **Privacy Risk:** A crucial factor for financial firms, where data leaving firm control (API calls) poses a higher risk than local model inference.

    This matrix forms the basis for making strategic decisions about LLM deployment based on specific project requirements and constraints.
    """)

    st.markdown(f"## 6. Strategic Cost-Accuracy Trade-offs and Crossover Analysis")
    st.markdown(f"Beyond raw performance numbers, understanding the long-term cost implications of each LLM strategy is paramount for a financial firm. The 'total cost of ownership' over time, especially with varying query volumes, dictates when an upfront investment in fine-tuning becomes more economically viable than continuous API calls.")
    st.markdown(r"The total cost of ownership for $N$ queries can be modeled as:")
    st.markdown(r"For API-based prompting: $$ C_{{\text{{prompt}}}}(N) = N \cdot C_{{\text{{query}}}} $$")
    st.markdown(r"where $N$ is the number of queries, and $C_{{\text{{query}}}}$ is the cost per API query.")
    st.markdown(r"For fine-tuning: $$ C_{{\text{{FT}}}}(N) = C_{{\text{{setup}}}} + N \cdot C_{{\text{{inference}}}} $$")
    st.markdown(r"where $C_{{\text{{setup}}}}$ is the fixed setup cost for fine-tuning, and $C_{{\text{{inference}}}}$ is the marginal inference cost (often negligible or zero for local models).")
    st.markdown(r"The cost crossover point $N^*$ is the query volume at which fine-tuning becomes more cost-effective than API-based prompting. It is found by setting $C_{{\text{{prompt}}}}(N) = C_{{\text{{FT}}}}(N)$: ")
    st.markdown(r"$$ N^* \cdot C_{{\text{{query}}}} = C_{{\text{{setup}}}} + N^* \cdot C_{{\text{{inference}}}} $$")
    st.markdown(r"$$ N^* (C_{{\text{{query}}}} - C_{{\text{{inference}}}}) = C_{{\text{{setup}}}} $$")
    st.markdown(r"$$ N^* = \frac{{C_{{\text{{setup}}}}}}{{C_{{\text{{query}}}} - C_{{\text{{inference}}}}}} $$")
    st.markdown(r"For local fine-tuned models, we typically assume $C_{{\text{{inference}}}} = 0$.")

    # Assuming these variables are globally defined and calculated by source.py
    # This requires running the cost crossover analysis code block from source.py upon import
    # prompt_per_query_cost_avg, ft_setup_cost, finetuned_per_query_inference_cost, crossover_queries
    # are assumed to be globally available from source.py
    
    st.markdown(f"**Cost Crossover Analysis:**")
    st.write(f"Average API Prompting Cost per Query: ${prompt_per_query_cost_avg:.6f}")
    st.write(f"Fine-Tuning Setup Cost: ${ft_setup_cost:.2f}")
    st.write(f"Fine-Tuned Model Inference Cost per Query: ${finetuned_per_query_inference_cost:.6f}")
    st.write(f"Cost Crossover: Fine-tuning becomes cheaper after {crossover_queries:.0f} queries.")
    st.write(f"At 20 queries/day: crossover in {crossover_days:.0f} days ({crossover_months:.1f} months).") # Assuming crossover_days, crossover_months are also calculated in source.py

    st.markdown(f"### Cost Crossover Plot")
    st.image("cost_crossover.png", caption="Cost Crossover: Prompting vs. Fine-Tuning Total Cost")
    
    st.markdown(f"### Cost-Accuracy Pareto Frontier")
    st.image("cost_accuracy_pareto_frontier.png", caption="Cost-Accuracy Pareto Frontier for LLM Strategies")

    st.markdown(f"**Explanation of Execution:**")
    st.markdown(f"""
    The cost crossover plot visually demonstrates the long-term financial implications. For a quant analyst, this is a clear signal: if a task is a one-off analysis (low query volume), prompting is cheaper. If it's a daily production pipeline with high query volume, fine-tuning offers significant cost savings over time, in addition to benefits in latency and privacy.

    The Cost-Accuracy Pareto Frontier plot helps identify the "efficient frontier" of LLM strategies – those that offer the highest accuracy for a given cost. Approaches on this frontier are generally preferred as they represent optimal trade-offs. Strategies far from the frontier might be sub-optimal choices. For instance, a high-cost, low-accuracy approach would be clearly undesirable.
    """)

    st.markdown(f"### Per-Class F1 and Latency Comparisons")
    st.image("per_class_f1_comparison.png", caption="Per-Class F1 Scores Across LLM Strategies")
    st.image("latency_comparison.png", caption="Query Latency Across LLM Strategies")
    
    st.markdown(f"### CFA ESG Benchmark Reproduction")
    st.image("cfa_esg_benchmark.png", caption="CFA Institute ESG Benchmark Reproduction (Conceptual)")

    st.markdown(f"**Explanation of Execution:**")
    st.markdown(f"""
    These visualizations highlight where each approach excels or struggles, particularly for specific sentiment categories, and the real-world operational efficiency.
    The CFA Institute's research on ESG materiality classification provides valuable empirical evidence for LLM performance on a more nuanced financial task. While our lab focuses on financial sentiment, we reproduce their findings conceptually to illustrate the significant accuracy gains from fine-tuning on complex, domain-specific tasks.
    """)
```
**`source.py` Integration**:
*   `results_df`, `prompt_per_query_cost_avg`, `ft_setup_cost`, `finetuned_per_query_inference_cost`, `crossover_queries`, `crossover_days`, `crossover_months` are global variables from `source.py`.
*   The `.png` image files (`cost_crossover.png`, `cost_accuracy_pareto_frontier.png`, `per_class_f1_comparison.png`, `latency_comparison.png`, `cfa_esg_benchmark.png`) are generated by `source.py` and displayed by `st.image()`.

#### Page: "7. Decision Framework & Strategic Implications"

**Markdown:**
```python
st.markdown(f"## 7. Informed Decision-Making for LLM Deployment in Finance")
st.markdown(f"Bringing all the analysis together, an investment professional needs a clear framework for selecting the optimal LLM strategy. This involves considering accuracy, costs, latency, data availability, and privacy concerns in the context of specific business requirements.")

st.markdown(f"### Decision Flowchart Logic for LLM Strategy Selection")
st.markdown(f"""
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
""")

st.markdown(f"### Strategic Implications for the Financial Firm")
st.markdown(f"""
Our analysis reveals key takeaways for strategic LLM deployment in a financial context:

*   **"Start with prompting, graduate to fine-tuning":** For initial feasibility and rapid prototyping, prompting strategies are invaluable. As tasks become critical, repetitive, and require higher accuracy or strict privacy, graduating to RAG or fine-tuning becomes economically and strategically sensible.
*   **Fine-tuning creates a "moat":** Leveraging proprietary labeled data to fine-tune models creates unique intellectual property. This domain-specific advantage cannot be easily replicated by competitors relying solely on generic LLMs, fostering a competitive edge in investment strategies.
*   **Nuance gradient determines the approach:** For simple, well-defined sentiment tasks, well-crafted prompts can achieve good performance. For highly nuanced tasks like ESG materiality classification, where subtle context shifts meanings, fine-tuning provides substantial accuracy gains.
*   **RAG as the middle ground:** When labeled data exists but is insufficient for full fine-tuning, or when model modification is constrained, RAG offers a powerful way to ground LLMs in domain-specific context without altering model weights.

By systematically evaluating these LLM strategies, we as investment professionals can make informed decisions that optimize resource allocation, manage risk effectively, and enhance the firm's analytical capabilities through strategic AI adoption.
""")
```
**`source.py` Integration**:
*   No direct function calls on this page. It relies on the user having processed the previous evaluation pages.

