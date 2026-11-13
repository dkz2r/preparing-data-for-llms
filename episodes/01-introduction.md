---
title: "Introduction"
teaching: 10 # teaching time in minutes
exercises: 2 # exercise time in minutes
---

:::::::::::::::::::::::::::::::::::::: questions

- Why is processing data correctly important for language models?
- What will we cover in this workshop?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Introduce the importance of data processing for language models
- Outline the key steps involved in preparing text data for training

::::::::::::::::::::::::::::::::::::::::::::::::

## Introduction

Generative AI runs on data.
A large language model learns everything it knows from a vast body of text, so before any training can start that text has to be gathered and prepared, and preparing it is one of the hardest and least visible parts of building the model.
This material is a practical, self-paced introduction to how that preparation is done, written for practitioners who can already code but are not data specialists; plan on about an hour to read it, longer if you stop to work the exercises.
Specifically we treat data processing: the work of turning raw text into something a model can learn from.
It is one neighbor in a larger family that includes data management and data governance.
Most of that data is training data, but the same care applies to the smaller evaluation and fine-tuning sets a model also needs.

What follows comes from our experience training large language models at [Fraunhofer IAIS](https://www.iais.fraunhofer.de/), across several projects: [OpenGPT-X](https://opengpt-x.de/), [OpenEuroLLM](https://www.openeurollm.eu/), and [SOOFI](https://www.soofi.info/).
OpenGPT-X produced Teuken, a family of open models trained on all 24 official languages of the European Union.
The lessons are drawn from building real training data and generalized so they transfer to any large-scale text pipeline.

::: prereq

You should be comfortable **reading code**: the examples are short Python snippets, there to make an idea concrete, never for you to write or run.
A **general sense of what a language model is** (text in, next-token prediction out) is enough on the modeling side.
No prior experience with data pipelines, web crawling, or natural-language processing is assumed; building that understanding is the point of this material.

:::

## Learning Objectives

By the end of this material you will be able to:

- Explain why data processing matters for what a model can do.
- Define the basic concepts of data processing, starting with data quality and filtering (Part 0).
- Trace how raw text, whether curated or scraped from the open web, becomes clean, uniform training data (Part 1).
- Reason about why scale makes compute the binding constraint rather than logic alone (Part 2).
- Distinguish the two things a pipeline produces at once, the data and the traceable record of how it was made, and say why the record matters as much as the data (Parts 1–2).
- Explain why quality signals are proxies for judgments too expensive to run directly, and why good text is approximated rather than measured (Parts 3–4).

Each part opens with its own, more specific objectives.


## Why This Matters

A language model is a product of its training data.
It has no knowledge of the world beyond the text it was shown, so the size, the languages, and the quality of that text set the ceiling on what the model can do.
Get the data wrong and no amount of clever training recovers it.
This is also where transparency is thinnest: how a model’s data was processed is rarely reported in any detail, which is part of why the work is worth teaching.

None of that text arrives ready to use.
It has to be found at the scale of hundreds of terabytes, cleaned, labeled, scored for quality, and deduplicated.
You may never run a pipeline this size yourself, but the lesson still applies: the datasets you download and train on were shaped by exactly these steps, and knowing how they were made tells you what you can trust them for.

Two questions run underneath all of it and shape the rest of the material: where does the enormous amount of suitable text a model needs come from, and what does *suitable* even mean?

::::::::::::::::::::::::::::::::::::: discussion

Consider text you encounter when browsing the web day to day.
- How much of it do you think would be usable for training a model?
- How does it rate in terms of quality, relevance, and reliability?
- How would a model be affected if we trained it exclusively on text scraped from apps like social media platforms?
- What if we just used something like wikipedia?
- What criteria would you use to decide if a piece of text is suitable for training?

::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: keypoints

- Processing data correctly is an important step when it comes to ensuring the quality and reliability of the resulting language model.
- The data used for training LLMs are vast and diverse, and require well thought out processing to ensure that the final data is suitable for training.

::::::::::::::::::::::::::::::::::::::::::::::::
