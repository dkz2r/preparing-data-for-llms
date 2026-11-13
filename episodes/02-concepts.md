---
title: "Basic Concepts"
teaching: 10 # teaching time in minutes
exercises: 2 # exercise time in minutes
---

:::::::::::::::::::::::::::::::::::::: questions

- What is some basic terminology that we need to know before we start?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Become familiar with basic terminology related to data processing for language models

::::::::::::::::::::::::::::::::::::::::::::::::

## Where Data Processing Sits

Three terms sit close together and are easy to confuse. They name different jobs over the same data. Definitions vary between frameworks, but a useful working split is:

- **Data governance** sets the rules: who may use which data, for what, under which policies, and who is accountable. It is about decisions and oversight1.
- **Data management** carries those rules out across the data’s life: storing, cataloging, securing, and keeping it available and correct.
- **Data processing** is one step inside management: transforming raw data into a usable form. For us, that means turning messy raw text into clean training data.

Put simply, governance decides *what is allowed and required*, management *keeps the data*, and processing *transforms it*.
This material lives in the last of the three. It touches the other two only where they constrain the processing, for example when a record must stay traceable so a fault can be assigned later.

## Kinds of Data

Not all data plays the same role.
The bulk is **training data**, the text a model learns from and the subject of this material.
Smaller sets serve other purposes: **evaluation data** measures how well a trained model performs, and **fine-tuning data** adapts it to a particular task or style.
Evaluation and fine-tuning sets are far smaller in volume than the training corpus, though the same quality concerns apply to all three.

## Data Quality

There is no single number that captures whether text is “good.”
The closest standard definition comes from ISO/IEC 25012, which describes data quality as the degree to which the characteristics of data satisfy stated and implied needs under specified conditions ([International Organization for Standardization 2008](https://www.iso.org/standard/35736.html)).
An older and more intuitive phrasing calls quality fitness for use2: data is high quality when it serves the purpose you have for it.

That phrasing carries a warning for everything that follows.
Quality is defined against a use, and the use here is training a capable language model, so what counts as “good text” depends on a downstream outcome no filter can observe directly.
Later episodes ([The Quality Question](./episodes/04-quality-question.md) and [Cheap Proxies for Good Text](./episodes/05-cheap-proxies.md)) return to this.

## Filters and Signals

Two words appear throughout the pipeline and are worth separating now.

A **signal** measures something about a piece of text: the share of characters that are symbols, the average number of words per line, a language-detection confidence, a model’s quality score. A signal is a measurement.

A **filter** is the decision that acts on a signal. It sets a threshold and keeps or drops each document accordingly. A signal tells you a document is 60% symbols; a filter is the rule “drop anything above 30%.”

Keeping the two ideas distinct matters later, when we ask what a filter can and cannot know. A filter is only ever as good as the signal beneath it, and a signal measures the surface of the text; what the words actually mean stays out of its reach.

## Compute and Time

A web-scale pipeline runs on shared high-performance computing (HPC) clusters, where two different budgets govern what is possible. They are easy to conflate, so it is worth separating them now.

**Compute time** is the total machine work a run consumes, measured in **CPU hours** and **GPU hours**: one processor busy for one hour is one core-hour. It is what the cluster bills, and what a project’s **compute budget** is fixed in.

**Wall-clock time** is the elapsed calendar time from start to finish. It is set by the longest chain of steps that must run in order, and by how much of the work can run in parallel. The same ten thousand core-hours can finish in an afternoon on enough machines or stretch over months on few.

The two come apart constantly: a stage can be cheap in compute yet slow on the clock, or the reverse. A real project is handed both a fixed compute budget and a deadline, so the two have to be managed together. Part 2 follows a concrete run where this split decides the shape of the whole pipeline.

With the vocabulary in place, Part 1 follows raw text through the pipeline that turns it into training data.



::::::::::::::::::::::::::::::::::::: keypoints

- There are several different kinds of data used in the pipeline.
- There is no single definition of "good" data.
- We used signals and filters to assess and select data for training.
- Compute time and wall-clock time are important considerations in managing a web-scale data processing pipeline.

::::::::::::::::::::::::::::::::::::::::::::::::
