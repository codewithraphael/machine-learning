# Market Basket Analysis Using Association and Mining Based Rule

## Abstract

This work presents a market basket analysis conducted on a retail grocery transaction dataset. Using association and mining based rule learning, the study aims to discover frequent itemsets and generate interpretable rules that describe consumer purchasing behavior. The resulting insights can be used to improve product placement, recommendation systems, and marketing strategies.

## Introduction

Market basket analysis is a foundational technique in unsupervised learning and retail analytics. It seeks to identify associations among items that are purchased together in the same transaction. This analysis is particularly relevant for grocery datasets where item co-occurrence patterns can inform merchandising decisions and inventory optimization.

## Dataset

The dataset used in this project is a grocery transaction log stored in `data/grocery_dataset.txt`. Each record represents a single basket, with items separated by commas. Example transactions include:

- `whole milk, butter, yogurt, rice`
- `tropical fruit, yogurt, coffee`
- `other vegetables, whole milk, condensed milk, long life bakery product`

This format supports the application of standard transaction mining algorithms, including Apriori and FP-Growth.

## Methodology

The analysis follows a structured pipeline:

1. Data ingestion: Read and parse the transaction dataset from `data/grocery_dataset.txt`.
2. Data preprocessing: Clean item labels, remove whitespace inconsistencies, and tokenize each basket into discrete items.
3. Transaction encoding: Convert the list of transactions into a binary matrix or transaction set suitable for frequent itemset mining.
4. Frequent itemset generation: Apply Apriori or equivalent algorithms to identify itemsets that exceed a minimum support threshold.
5. Rule induction: Derive association rules from the frequent itemsets and assess them using support, confidence, and lift.
6. Validation and visualization: Evaluate the interestingness of rules and visualize results in the `plots/` directory.

## Project Structure

- `data/`
  - `grocery_dataset.txt` — primary transaction dataset
  - `groceries.csv` — alternate dataset representation
- `src/`
  - source code for data preparation, itemset mining, and rule generation
- `plots/`
  - visualizations such as support-confidence charts, lift plots, and frequent itemset diagrams

## Results

The analysis aims to identify the following:

- High-support itemsets that occur frequently across transactions
- High-confidence association rules that indicate strong purchase dependencies
- Rules with lift greater than 1, suggesting item relationships that exceed random chance
- Actionable observations for retail layout and cross-selling opportunities

The specific output of this project should include:

- A ranked list of frequent itemsets
- A set of association rules with corresponding support, confidence, and lift metrics
- Visual summaries of rule distributions and itemset frequencies

## Conclusions

Market basket analysis is an effective method for extracting purchase patterns from transactional data. By combining frequent itemset mining and rule evaluation, the project delivers interpretable insights that can be translated into business strategies. This study reinforces the value of association rule mining in retail analytics and provides a reproducible workflow for further experimentation.

## Future Work

Potential extensions include:

- Comparing Apriori and FP-Growth algorithms for runtime and rule quality
- Experimenting with alternative interestingness measures such as conviction or leverage
- Extending the dataset with temporal or customer segmentation metadata
- Deploying the resulting rules in a recommendation or promotion engine

## Requirements

The analysis can be executed with Python and the following libraries:

- `pandas`
- `mlxtend`
- `matplotlib`
- `seaborn`

Install dependencies with:

```bash
pip install pandas mlxtend matplotlib seaborn
```

Alternatively, if a `requirements.txt` file is provided:

```bash
pip install -r requirements.txt
```

## Notes

- `src/` is intended for implementation of the analysis pipeline and should contain preprocessing, frequent itemset mining, and rule generation scripts.
- `plots/` should contain the visualization artifacts used to support the research findings.
- `data/` contains the underlying grocery transaction dataset.
