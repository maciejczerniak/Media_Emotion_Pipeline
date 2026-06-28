# Model Card

## 1. Model Overview

The developed model is a **fine-tuned [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base)** transformer.

This architecture was selected after thorough experimentation with traditional machine learning models, RNNs, and BERT. It achieved the **highest accuracy** while maintaining a **lightweight and efficient inference interface**, suitable for use on **non-specialized hardware**.

The model was built for a **client application** as an **emotion classifier** to replace their current, expensive large language model (LLM) solution. This approach enables:

* More **consistent and controlled predictions**
* Greater **independence from big-tech AI providers**
* Significant **cost reduction** in production environments

The development process was based on a few key assumptions:

* The **training data**, sourced from the client’s current pipeline (LLM-generated predictions), is **accurate and reliable enough** to be used as ground truth.
* The **intended use** of the model remains **within the communicated domain** and scope.

The main limitation of this work was the **time constraint** — the 8-week period was insufficient to deliver a fully production-ready system. Therefore, this model represents a **proof of concept (PoC)**.
Additionally, training was performed on **consumer-grade hardware (RTX 5080)**, which limited optimization efficiency. For a production-level solution, **specialized hardware** would be recommended.

## 2. Intended Use

### Intended Application
This model classifies emotions in video transcripts, focusing on the six core emotions defined by Paul Ekman - **happiness**, **sadness**, **anger**, **fear**, **surprise**, and **disgust** - plus a **neutral** category.
It is intended for use by the **Content Intelligence Agency** to analyze emotional tone in media content such as films, TV shows, and online series.
The resulting labels help media producers and analysts understand the emotional dynamics of their content and identify moments of high emotional engagement.

### Example Use Cases
- Analyzing a *MasterChef Poland* episode to detect when judges express anger or contestants show fear.
- Generating emotion timelines across a TV series to support editing and highlight creation.
- Supplying structured emotion labels as input for downstream media-analytics tools.

### Connection to Client Needs
Previously, the client relied on external large-language-model APIs for emotion detection, which were **expensive** and **opaque**.
This project delivers a **local, interpretable, and cost-efficient** alternative aligned with the client’s goal of building self-contained AI solutions for content intelligence.

### Limitations
- **Domain-specific data:** Training focused on translated TV show transcripts; results may not generalize to social-media or news data.
- **Translation noise:** Non-English source material may lose nuance during translation.
- **Class imbalance:** One emotion label represents ~44 % of the dataset, affecting performance on minority emotions.
- **Time constraints:** Eight-week project limited opportunities for extensive tuning.
- **Not for human evaluation:** The model must **not** be used for clinical, psychological, or individual-level emotion assessment.


## 3. Dataset Details

### Overview
- **Training + validation:** 30 409 translated sentences from multilingual TV show transcripts.
- **Test:** 792 manually translated sentences from *MasterChef Poland*.
- Data collected and cleaned by students to closely reflect the target domain of media analysis.

### Preprocessing & Labeling
1. Translation to English
2. Sentence tokenization and cleanup (removal of fillers and non-verbal cues)
3. Manual annotation for one of the six core emotions + neutral
4. Undersampling and stratified splitting to reduce imbalance

### Language and Cultural Considerations
English was chosen as the working language because high-quality NLP resources are readily available.
Polish-language data were translated due to limited local NLP support, though translation inconsistencies may blur emotional nuance.
TV dialogues include informal speech, slang, and interruptions that add complexity for emotion modeling.

### Representativeness
The dataset represents emotions typical of **scripted and reality entertainment** rather than spontaneous conversation.
It is **not linguistically or culturally comprehensive**, but optimized for the model’s media-analysis use case.

## 4. Error Anlysis

### Summary
After classifiyng each mistake to each class, we discovered that the largest amount of mistakes (and as such, the limitations of our model) are in short, ambigious sentences that are taken out of context, ambigious sentences even if context is taken into account, and in sentences with emotionally charged single words, which pull the model the wrong way.

### Full Report
[Link here](../experiments/error_analysis/Task%209%20Report%20(Error%20Analysis).pdf)

## 5. Explainable AI

### Summary
The XAI methods confirmed our theory of emotionally charged single words pulling the model the wrong way, and showed low confidence in the predictions of our model.

### Full Report
[Link here](../experiments/explainability/Task%2010%20Report%20(XAI).pdf)

## 6. Recommendations for Use

### Deployment Context
This model fits into a pipeline that processes audiovisual content:
**video → transcription → translation → emotion classification → structured output.**
It is suited for **media companies**, **researchers**, and **developers** performing narrative or affective analysis in video transcripts.

### Best Practices
- Verify transcription and translation quality before emotion tagging.
- Use batch processing for scalability and reproducibility.
- Apply temporal smoothing to create stable emotion trajectories.
- Manually review rare or uncertain classifications.

### Operational Risks
- Upstream transcription or translation errors can cascade into wrong emotion labels.
- Misclassification of subtle or mixed emotions is possible.
- Performance declines on out-of-domain or non-English data.

### Future Work
- Add **emotion-intensity tagging**.
- Extend to **fine-grained emotions** beyond the six core types.
- Incorporate **multilingual pretraining** to reduce dependence on translation.

### Stakeholder Summary
This model provides the **Content Intelligence Agency** with a transparent, maintainable, and cost-effective tool for emotion classification in media transcripts - enabling scalable emotion analytics without reliance on external LLM APIs.

## 7. Sustainability Considerations

### Energy Used for Training

This model is based on the pre-trained **RoBERTa-base** architecture, which significantly reduces the carbon footprint by reusing existing resources instead of training from scratch.

Fine-tuning was performed on an **RTX 5080 GPU** and took approximately **5 hours** in total. The estimated energy consumption was **2.6 kWh**, calculated as follows:

```
Energy [kWh] = 8 h × (360 W × 0.9) / 1000 = 2.592 kWh
```

> *0.9 represents the average GPU load during training.*

To put this into perspective, **2.6 kWh** is roughly equivalent to:

* Driving an electric car for **about 15 km**, or
* Brewing **around 100 cups of coffee**

All energy for this task was sourced entirely from **solar panels** at one of the team member’s homes, resulting in a **carbon footprint of 0 kg CO₂e**.

### Energy Use During Inference

Estimating daily or monthly energy usage is not possible due to the lack of access to company-wide statistics. Instead, the estimate below is provided **per 1,000 requests**.

Predicting **1,000 rows of data** takes approximately **20 seconds** and uses **90 % of GPU capacity (324 W)**, resulting in an energy consumption of **1.8 Wh per 1,000 requests**.

### Disclaimer

1. The calculations are based on an **RTX 5080 GPU**. Using a different configuration will change the results.
2. The estimates refer to **batch predictions**. Single-row predictions result in higher relative energy consumption.
