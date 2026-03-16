# Accuracy Tutorial — “How Accurate Is Your Scanner?”

When someone asks:

> “How accurate is your card scanner?”

…the correct response is:

> “What kind of accuracy do you mean?”

This is the zero-to-hero version of that idea, and it’s foundational for this project.

---

## 1) “Accuracy” is not one thing

In card scanning, people usually care about five different questions:

1. **Can it identify the card name?**
2. **Can it identify the edition/printing?** (harder for subtle reprints)
3. **Can it identify card condition?**
4. **Can it detect counterfeit cards?**
5. **Can it distinguish physical instances?**
   - e.g., “Is this the same exact Beta Island copy I owned before?”

All five are valid. They are also different technical problems.

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

If a seller ships the wrong edition:
- maybe the card value delta is small,
- but the **customer-service/time/reputation cost** can be much larger than the raw card price.

So edition accuracy is not academic. It’s operational.

---

## 4) Reprint accuracy: the real torture test

The hardest scenarios are heavily reprinted cards with near-identical visuals.

### Why Sol Ring is used as the torture test

- Sol Ring is one of the most played cards in Commander.
- The Mike Bierek Sol Ring artwork is one of the most reprinted artworks in Magic.
- As of this writing, this illustration appears in **43 prints**.

Scryfall reference:
- https://scryfall.com/search?q=illustrationid%3A146aaae4-93f4-409a-be32-010e86d137da+include%3Aextras+unique%3Aprints&unique=prints&as=grid&order=released

In other words: same/similar art, many editions, subtle differences.
That is exactly the kind of case where edition identification separates strong systems from weak ones.

---

## 5) Gold-standard benchmark logic in this repo

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

| Rank | Card (sample) | Prints | Scryfall link |
|---|---|---:|---|
| 1 | Sol Ring | 43 | https://scryfall.com/search?q=illustrationid%3A146aaae4-93f4-409a-be32-010e86d137da+include%3Aextras+unique%3Aprints&unique=prints&as=grid&order=released |
| 2 | Terramorphic Expanse | 39 | https://scryfall.com/search?q=illustrationid%3Ac6230ffb-dd2c-4a2c-aeaf-13ba42c89472+include%3Aextras+unique%3Aprints&unique=prints&as=grid&order=released |
| 3 | Exotic Orchard | 35 | https://scryfall.com/search?q=illustrationid%3Af67845cb-fe3b-4f90-8de2-bdb69daca307+include%3Aextras+unique%3Aprints&unique=prints&as=grid&order=released |
| 4 | Arcane Signet | 32 | https://scryfall.com/search?q=illustrationid%3A06d7b990-021a-4b1f-9333-4fc82adf0ea4+include%3Aextras+unique%3Aprints&unique=prints&as=grid&order=released |
| 5 | Swiftfoot Boots | 31 | https://scryfall.com/search?q=illustrationid%3A798a59e5-d388-4110-9437-9616bdfd7c14+include%3Aextras+unique%3Aprints&unique=prints&as=grid&order=released |

---

## 7) Practical definition of scanner quality (for this project)

When we report “accuracy,” we should explicitly report both:

1. **Name Identification Accuracy**
   - Correct card name / artwork-level match
2. **Edition Identification Accuracy**
   - Correct printing-level match

And always state:
- dataset/split used
- whether test is easy/typical/hard
- whether Sol Ring-style stress cases are included

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
