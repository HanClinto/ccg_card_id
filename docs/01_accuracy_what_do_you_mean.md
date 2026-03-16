# Accuracy Tutorial — “How Accurate Is Your Scanner?”

When someone asks:

> “How accurate is your card scanner?”

…the correct response is:

> “Which kind of accuracy do you mean?”

So let's define our terms.

---

## 1) “Accuracy” is not one thing

In card scanning, people might crae about any number of things when they ask about identification accuracy:

1. **Can it identify the card name?**
2. **Can it identify the edition/printing?** (harder for subtle reprints)
3. **Can it identify the card language?**
4. **Can it identify card condition?**
5. **Can it detect counterfeit cards?**
6. **Can it distinguish physical instances?**
   - e.g., “Is this the same exact Beta Island copy I owned before?”

The last one is a bit extreme, but they each hve their uses, and all five are valid. 

Each one is also a different technicl problem.

---

## 2) Scope of *this* project

This project focuses on **#1 and #2**:

- **Card Name Identification**
- **Edition Identification**

Out of scope (for now):
- condition grading
- counterfeit detection
- physical-instance fingerprinting

Those are useful and interesting, but they require different datasets, labels, and model design.

---

## 3) Why this distinction matters

A scanner can be “very accurate” for card names and still be weak on editions.

Example:
- Predicting **“Sol Ring”** correctly is often easy.
- Predicting **which exact Sol Ring printing** is much harder.

If you are a player focusing on building a deck or a cube, then which particular edition you're holding might not matter much to you -- [a $0.67 Lightning Bolt](https://scryfall.com/card/clb/401/lightning-bolt) does the same 3 damage to the dome as [a $670 Judge promo](https://scryfall.com/card/jgp/1/lightning-bolt). 

However, if you're a seller, then edition can matter much more. If you ship the wrong edition:
- maybe the card value delta is small,
- but the **customer-service/time/reputation cost** can be much larger than the raw card price.

Do I want to spend my time dealing with you complaining about having received [a PLST reprint](https://scryfall.com/card/plst/C18-222/sol-ring) of Sol Ring instead of [the Commander 2018 Original](https://scryfall.com/card/c18/222/sol-ring) that you ordered? Absolutely not. 

There are more dramatic cases of misidentification we could point to, but even in the simple $1 Sol Ring case, the authors of this project believe that increasing edition accuracy is tangibly important because it increases customer order quality in every-day, real-world scenarios.

---

## 4) Reprint accuracy: the real torture test

The hardest scenarios are heavily reprinted cards with near-identical visuals.

### Why Sol Ring is used as the torture test

- Sol Ring is the [number-one card in Commander](https://scryfall.com/search?as=grid&order=edhrec&q=%28game%3Apaper%29+prefer%3Abest).
- The Mike Bierek Sol Ring artwork is the [number-one most-reprinted artwork in Magic](https://scryfall.com/search?q=illustrationid%3A146aaae4-93f4-409a-be32-010e86d137da+include%3Aextras+unique%3Aprints&unique=prints&as=grid&order=released).
  - As of this writing, this illustration appears in **43 prints**.

In other words: same/similar art, many editions, subtle differences.
That is exactly the kind of case where edition identification separates strong systems from weak ones.

---

## 5) Gold-standard benchmark logic in this repo

I have created a dataset of real-world images from images that I took of my personal collection of 20+ different editions of Sol Ring. We placed the cards against a plain white background and took video of the cards from various angles and lightings. We then extracted good keyframes from these videos and filtered it down to a set of 307 images that we feel should be able to be accurately identified, but are certainly not easy. We call this (unsurprisingly) the "solring" dataset, and it is a pathalogical worst-case scenario for scanning software. We have other datasets that we use for measuring other aspects of accuracy, but this is our gold standard for edition accuracy.

We use the `solring` dataset as a **gold-standard stress test** for edition ID:
- 20+ editions with very similar visual appearance
- tests whether a method can separate fine-grained print differences

Important note:
- We do **not** train specifically on Sol Ring alone.
- Sol Ring is a benchmark lens, not the entire learning target.

Why: many other cards are similarly reprinted.
So we evaluate with Sol Ring as a near-worst-case probe while training for broader generalization.

---

## 6) Top 5 heavily reprinted artworks (by English print count in local catalog)

These are useful “edition stress” candidates, with Scryfall links.

| Rank | Card | Artist | Prints |
|---|---|---|---|
| 1 | Sol Ring | Mike Bierek | [43 prints](https://scryfall.com/search?q=illustrationid%3A146aaae4-93f4-409a-be32-010e86d137da+include%3Aextras+unique%3Aprints&unique=prints&as=grid&order=released) |
| 2 | Terramorphic Expanse | Dan Murayama Scott | [39 prints](https://scryfall.com/search?q=illustrationid%3Ac6230ffb-dd2c-4a2c-aeaf-13ba42c89472+include%3Aextras+unique%3Aprints&unique=prints&as=grid&order=released) |
| 3 | Exotic Orchard | Steven Belledin | [35 prints](https://scryfall.com/search?q=illustrationid%3Af67845cb-fe3b-4f90-8de2-bdb69daca307+include%3Aextras+unique%3Aprints&unique=prints&as=grid&order=released) |
| 4 | Arcane Signet | Dan Murayama Scott | [32 prints](https://scryfall.com/search?q=illustrationid%3A06d7b990-021a-4b1f-9333-4fc82adf0ea4+include%3Aextras+unique%3Aprints&unique=prints&as=grid&order=released) |
| 5 | Swiftfoot Boots | Svetlin Velinov | [31 prints](https://scryfall.com/search?q=illustrationid%3A798a59e5-d388-4110-9437-9616bdfd7c14+include%3Aextras+unique%3Aprints&unique=prints&as=grid&order=released) |

---

## 7) Practical definition of scanner quality (for this project)

When we report “accuracy,” we should explicitly report both:

1. **Dataset Identification**
   - Report which dataset this used for test imagery
2. **Name Identification Accuracy**
   - Correct card name / artwork-level match
3. **Edition Identification Accuracy**
   - Correct printing-level match

And always state:
- split used for the dataset (if any)
- whether test is easy/typical/hard

---

## 8) Quick communication template

If someone asks “How accurate is it?”, answer like this:

> “For card-name identification, we get X on dataset Y. For exact edition identification, we get Z on dataset Y, and W on our Sol Ring stress test (near-worst-case reprint confusion).”

That is honest, precise, and useful.

---

## 9) Related docs

- Project intro: `docs/00_welcome.md`
- Vectorization concepts: `04_vectorize/readme.md`
- Edge deployment tutorial: `docs/04_vectorize_part2_edge_deployment.md`
- Model survey: `docs/embedding-model-survey.md`
