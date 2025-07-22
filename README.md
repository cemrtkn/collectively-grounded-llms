# Collectively Grounded LLMs

## Summary

Generative AI systems like large language models (LLMs) are changing how we produce and share language. As these systems become more common, two main worries have emerged. One is the fear of global sameness, where centralized models erase cultural and linguistic differences. The other is the fear of extreme personalization, where AI adapts so much to each person that shared ways of speaking and understanding start to disappear.

This project explores a third, often overlooked possibility: using generative AI to support collective identity. Instead of making models that speak the same way to everyone—or that tailor their responses to each individual—we suggest designing them to reflect social groups: communities, movements, and publics. These groups have long shaped language and culture by offering shared values, perspectives, and ways of speaking.

We argue that collectives offer the right balance for AI design—not too general, like global models, and not too narrow, like personal ones. People are shaped by personal experiences, but it is through collectives that ideas grow, are tested, and refined through conversation and feedback.

Cultural evolution also depends on diverse and competing ways of thinking. This diversity often emerges through group-level processes: groups form shared identities, coordinate their behavior, and create new norms. Supporting a variety of collectives through AI could help keep culture dynamic, rather than making it more uniform.

While today’s LLMs can imitate social groups through prompting, these imitations are often based on outsider stereotypes rather than the group’s own language. Our project takes a different approach: we fine-tune models on real data from specific communities and train the system to recognize special tokens that represent different collective identities. This allows users to choose which group’s voice the chatbot should adopt.

By anchoring AI in real communities, we treat it not just as a neutral tool, but as a cultural medium—able to carry and express the values, styles, and ways of speaking of actual groups. This could lead to more context-aware and pluralistic AI systems that support, rather than erase, cultural diversity.

## Method

1. Create synthetic question - answer pairs form primary sources
2. Fine-Tune a large language model on these pairs while inserting the corresponding data source as meta data
3. Train a classifier on the synthetic data predicting the meta data (optional)
4. Use RL to either (optional)
    1. Fine-tune a vanilla instruct model with the classifier for different metadata
    2. Further fine-tune the instruct model from step 2 with the classifier for different metadata

### Results

- An interface that allows interacting the with the model
- Quantitative evaluation of model differences (e.g. answers to the world survey)

## Research Packages

### Literature research

Based on the abstract, read the paper with a set of questions. Do you want to learn about the method? Does the paper provide conceptional motivation? When reading, focus on these questions. You can skip parts that are irrelevant therefore. After reading, summarize in a few sentence why this paper is relevant for the project. This is not a summary of the paper, but very specific for the project. Add into a table: Title, Year, Authors, Journal, Citations, Relevance for project

To be read:

- The inspirational papers
- The potential relevant paper
- Further literature research

---

### Find Suitable Datasets

Given that we are building a proof of concept, the dataset can be selected based on personal interest.

Criteria

- Q&A pairs can be extracted from the documents by an LLM
- different, disjunct collective idendities (group, time, demographics) are present
- dataset allows the extraction of 5-10k Q&A per identity (often many Q&A pairs can created form a single document)

Some ideas:

https://chatgpt.com/share/687bbcf0-496c-800b-8a6b-a71054fcd80f

---

### Create Synthetic Q&A Pairs

Potentially usefull starting point on how to create synthetic data with an LLM and some primary sources:
https://github.com/meta-llama/synthetic-data-kit

---

### Fine-Tune an LLM on Q&A Pairs

We have a bunch of code examples that we can share. 

---

### Create Tool to allow interaction with the LLM

Suggestion: A gradio interface (https://www.gradio.app/) or an simple chat extension

---

### Quantitative Evaluation (optional)

tbd

---

### Refine LLM with RL (optional)

tbd

## References

### Inspiring References

- Generative AI enhances individual creativity but reduces the collective diversity of novel content (https://www.science.org/doi/full/10.1126/sciadv.adn5290)
- AI models collapse when trained on recursively generated data (https://www.nature.com/articles/s41586-024-07566-y)
- Towards Measuring the Representation of Subjective Global Opinions in Language Models (https://arxiv.org/abs/2306.16388)
- Maintaining Transient Diversity Is a General Principle for Improving Collective Problem Solving (https://pubmed.ncbi.nlm.nih.gov/37369100/)

### Further relevant reference

- Generative Agent Simulations of 1,000 People (https://arxiv.org/pdf/2411.10109)
{The dataset used here could be interesting.}
- Large language models that replace human participants can harmfully misportray and flatten identity groups (https://arxiv.org/abs/2402.01908)
- ComPO: Community Preferences for Language Model Personalization (https://arxiv.org/html/2410.16027)

### Automatic Literature Research

https://chatgpt.com/share/687bb60d-51fc-800b-8c7b-d171bbedf127

### On macOS:
1. Install pipx: `brew install pipx && pipx ensurepath`
2. Install Poetry: `pipx install poetry`
3. Install dependencies: `poetry install`
4. Activate virtual environment: `poetry env activate`

## Run Project

Instructions for running the project should be added here after forking.

## Running Tests

Execute tests using pytest:
```bash
poetry run pytest
```

## Contributing

Please read [contributing.md](contributing.md) for guidelines on how to
contribute to this codebase.

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE)
file.
