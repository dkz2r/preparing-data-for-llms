---
title: "Cheap proxies for good text"
teaching: 10 # teaching time in minutes
exercises: 2 # exercise time in minutes
---

:::::::::::::::::::::::::::::::::::::: questions

-

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

-

::::::::::::::::::::::::::::::::::::::::::::::::

## Cheap proxies for good text

[Part 3](#part-3) ended on a sharp question: can a pipeline keep the judgment of a language model while paying something close to the price of a surface heuristic? Answering it means facing a harder question the pipeline has so far avoided. Every step has sorted documents into good and bad without ever saying what *good* means: [Part 1](#part-1) caught malformed text and removed duplicates, [Part 3](#part-3) scored surface form, yet none of them defined the quality they were chasing. This part confronts that gap first, then shows the practical answer a real pipeline uses.

## Bad is easy to spot, good is not

Defining bad text is straightforward. Bad text has signs you can count: it is too short, it is mostly symbols, it repeats one line, it is a duplicate of something you already kept. Part 1 caught all of these by counting characters, because a malformed document reveals itself in its shape.

Good text has no such signs. There is no measurement you can run on a single document that returns “this is worth learning from”, because worth is *fitness for use* ([Part 0](#part-0)): it depends on the goal you are training for and on what the model still needs to learn, neither of which is written on the page. So nobody filtering data ever truly writes down what good is. They pick a *proxy* for it: a property they can measure on each document, chosen because it tends to track quality even though it is not quality itself. The proxy that FineWeb introduced is **educational value**: how much genuine, learnable information a document carries, scored on a small 0-to-5 scale by a classifier trained on LLM-written ratings ([Penedo et al. 2024](#ref-penedo2024fineweb)). It was chosen for three reasons, in rising order of importance: it is intuitive to people, it is cheap enough to score across the whole web, and, above all, models trained on the data it selects do better on the benchmarks. Educational value is a deliberate choice, open to debate, which is why even trained human annotators disagree on the borderline cases ([Ali et al. 2025](#ref-ali2025jql)).

This difference shapes the rest of the work. You can define bad up front and act on it cheaply. For good, you choose a proxy, apply it, and only find out whether the choice was sound much later, by training a model on the result and measuring how it does.

## Every quality signal is a cheap proxy

Seen this way, the tools in Parts 1 to 3 share one idea. A **surface heuristic** is a cheap proxy for “is this well-formed?”. An **LLM judge** is an expensive proxy for “would a careful reader call this good?”. Each replaces a judgment we cannot make directly with a measurement we can.

**Key takeaway**

**Every quality signal is a cheap proxy for an expensive judgment.** The recipe is always the same: take the most trustworthy judgment you can afford, then distill it into a cheaper one, and repeat, each step trading a little accuracy for a large cut in cost, until the signal is cheap enough to run at the scale you need.

That recipe stacks the judges in tiers, three of them. It helps to start at the most trustworthy, most expensive tier and work down, since each tier exists only to make the one below it affordable.

**The top tier is human judgment.** The closest thing to a true measure of whether a document is worth learning from is a person reading it, the way an editor decides what belongs in a book. It is also the costliest: a qualified annotator spends minutes per document and is paid by the hour, so scoring runs into euros per document[^1], against fractions of a cent for a model. Reading the roughly five billion surviving documents by hand is impossible, so people never judge the whole corpus. They score a small, curated **seed** instead, the same kind of high-trust curated data from [Part 1](#part-1).

**The middle tier is a large language model, calibrated against that seed.** A capable model can read a document and rate it much as a person would, and the seed is how you check that it does: you keep the model whose scores track the human ones. But a model big enough to judge well is still far too expensive to run on the whole web, around **1,5 million €** for a single pass ([Part 3](#part-3)). So it never sees every document either. It judges a larger *sample*, big enough to teach the next tier but far short of the full corpus.

**The bottom tier is a small model that runs everywhere.** The sample the large model judged becomes training data for a small, cheap model that reproduces its verdicts at a fraction of the cost, light enough to score every surviving document. Each tier, in short, trains or calibrates the one beneath it: people vouch for the large model, and the large model labels the data that teaches the small one.

This three-tier chain is what JQL (Judging Quality across Languages, pronounced *Jackal*) ([Ali et al. 2025](#ref-ali2025jql)) builds; it came after OpenGPT-X and sharpens the same idea, and the details belong to the paper. The shape is always the same: pay one large cost once, on a sample, so a cheap step can run at scale.

Plot the tiers by cost and coverage and they fall on a single downward line: the cheaper a judgment is per document, the more documents it can afford to run on. The costliest tier, human review, reaches only a curated seed; the cheapest, surface filters, runs across the whole web:

![A log-log scatter plot. The horizontal axis is the number of documents a judgment runs on, from ten thousand to a trillion; the vertical axis is cost per document in euros. Three judgment tiers fall on a downward line from top-left to bottom-right: human review of a curated seed at about 10 euro per document; the LLM judge at about 0,0003 euro per document over the roughly 5 billion survivors; and surface filters plus deduplication at about 0,00000007 euro per document across the whole web. An open marker at the bottom right is the raw web of 173 billion documents, ungraded.](fig/part4/quality-funnel.svg)

Figure 10: The judgment tiers on a cost-versus-coverage plot, both axes logarithmic. They fall on a downward line: human review costs about 10 € per document but reaches only a curated seed; the LLM judge costs about 0,0003 € and reaches the ~5 billion survivors; surface filters and deduplication cost about 0,00000007 € and run across the whole web. The raw web of 173 billion documents sits ungraded on the baseline.

Each tier down is cheaper than the one above, which invites one last question: could you take one more step and drop the model too? If some surface heuristic tracked the model’s verdicts closely enough, you could filter with plain character counts and skip the reading entirely. Nobody has found one that does, and [Part 3](#part-3) is the reason to doubt one exists: surface form does not determine value, so a rule that only counts characters has nothing to read meaning with. Whether a rich enough set of surface features could ever line up perfectly with content is genuinely open; what we know is that none does today. So in practice the chain bottoms out at a small model that reads what the words mean, and for now there is no cheaper tier beneath it.

The distilled model now puts a cheap value score on every surviving document, which turns the next step into a question: given a score for each document, where do you set the cutoff?

### Filtering on rank rather than score

A proxy that runs on every language raises a practical problem the moment you try to filter with it. A model returns a quality *score*, and you keep documents above some cutoff. But the score’s scale is arbitrary: one judge may rate most documents around 1, another around 4, while both rank documents in nearly the same order. An absolute cutoff like “keep everything scoring 3 or higher” then means something different for each judge and each language. The fix is to threshold on **rank**: keep the top share of documents, whatever their raw scores happen to be.

``` python
# Two quality judges score the same 10 documents from 0 to 5. They agree on the
# ranking but sit on different scales: "strict" clusters low, "generous" high.

strict   = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.1, 2.4, 2.8, 3.2]
generous = [2.1, 2.6, 3.0, 3.3, 3.6, 3.9, 4.1, 4.3, 4.6, 4.9]

def keep_absolute(scores, cutoff=3.0):
    return sum(s >= cutoff for s in scores)

def keep_top_fraction(scores, fraction=0.4):
    # keep the best-scoring `fraction` of documents, whatever the scale
    ranked = sorted(scores)
    cut = ranked[int(len(scores) * (1 - fraction))]
    return sum(s >= cut for s in scores)

for name, scores in [("strict", strict), ("generous", generous)]:
    print(f"{name:8}  absolute >=3.0 keeps {keep_absolute(scores)}/10"
          f"   top 40% keeps {keep_top_fraction(scores)}/10")
```

    strict    absolute >=3.0 keeps 1/10   top 40% keeps 4/10
    generous  absolute >=3.0 keeps 8/10   top 40% keeps 4/10

The absolute cutoff keeps almost nothing from the strict judge and almost everything from the generous one, even though both ranked the documents the same way. The rank-based cutoff keeps the same share from each, so the filter means the same thing no matter which judge or which language produced the scores. It also gives you one clear control: the fraction you keep sets the trade-off between corpus size and average quality directly.

**Your turn**

Tightening the rank cutoff is only one way to raise the value of what you keep, and it always costs you corpus size. Name two *other* levers that improve the average quality of the surviving documents without simply keeping a smaller top share. Then consider the case OpenGPT-X actually used: a document is kept only if *three* independent judges each rank it in their own top share. Why would requiring agreement be safer than trusting one judge’s ranking?

**One way to reason about it**

Tightening the cutoff just moves along the quality-versus-quantity trade-off: a smaller corpus, higher average quality, nothing new gained. The other levers change the trade-off itself. **Source mixing**, upsampling high-value sources and downsampling low-value ones, lifts the average without judging any single document harder. And **combining signals of different kinds**, an LLM judge that reads meaning together with deterministic surface metrics, catches what either alone would miss: the judge scores value, the counts catch the malformed cases it overlooks. Requiring three judges to agree is a variation on that same idea: it guards against any one of them being wrong, since a single proxy has blind spots and noise, and a document that three differently-trained judges all rank highly is a safer choice than one that only a single model rated highly.

## A wider family of proxies

Every proxy so far has done the same thing: score a whole document from its text, by its shape with a surface heuristic or by its meaning with an LLM judge, then keep it or drop it. Those are not the only instruments. Two more are worth knowing, because they show how broad the idea of a quality signal really is.

The first reads something other than the text at all. A document carries *implicit* signals from how the web treated it: how many pages link to it, where a search engine ranks it, how often people click or stay. These judge a page by its reception rather than its words, so they need neither a surface heuristic nor an LLM judge. OpenGPT-X did not use them, but they are a standing option.

The second does read the text, but asks a narrower question: how surprising is the text? To measure it, you can run a small language model over a document and measure its *perplexity*, how surprised the model is by the words: text that resembles the clean reference the model was trained on scores low, while gibberish, boilerplate, and spam score high.

Taken together, these signals are a reminder that quality can be guessed from almost anywhere, not only a literal reading of the words on the page.

## What “good” still leaves open

JQL gives a strong, cheap proxy for good text and runs it across every language. It still leaves the definition of good itself open. Each step in this material has approximated good with something it could measure: [Part 1](#part-1) with well-formedness, and with whether a document added anything new or merely repeated what was already kept, [Part 3](#part-3) with those same surface heuristics, which it showed cannot reach value, and Part 4 with a model’s value score. Underneath all of them, the final test of whether any of it worked is how the trained model does on a benchmark, which is itself a proxy for the quality we actually care about. The approximations stack, and at the bottom there is no measurement that reads “this text is good”.

This is the honest shape of the problem. Defining good text well enough to measure it, and across every language at once, is open research, and the choice of proxy decides what kind of model the data will make.

## Recap

Part 4 reframed the whole pipeline as a sequence of proxies. Bad text is definable and can be filtered cheaply; good text cannot be defined outright, so every quality signal is a cheap proxy for it, something measurable, and the only test of the choice comes downstream. Because a proxy good enough to read value is expensive to run, the move is to pay for it once on a sample and distill it into a cheap model that runs on the billions. Filtering on rank rather than raw score makes that proxy behave consistently across judges and languages. What remains open is the thing underneath: what makes text good at all.

[^1]: Illustrative: a qualified annotator costs on the order of tens of euros an hour and scores only a handful of documents in that time, so human scoring runs to euros per document, orders of magnitude above an LLM judge at about 0,0003 € per document. The machine prices behind that comparison, about 0,045 € per CPU-hour and about 3 € per GPU-hour with their sources, are in the cost footnotes of [Part 3](#part-3).

::::::::::::::::::::::::::::::::::::: challenge


:::::::::::::::::::::::: solution


:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: keypoints

-

::::::::::::::::::::::::::::::::::::::::::::::::
