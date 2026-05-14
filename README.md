![Uploading Gemini_Generated_Image_bqryevbqryevbqry.png…]()



# Fine-Tuning my own philisopher friend STOA — My Learning Journal

I wanted to learn how to finetune and what exactly happens. The easiest way for me is to do it practically.

First problem when I decided I would go ahead and finetune a model was to decide what to actually finetune. Because personally the more I thought about it, the more I realised you really need to have some valid use case to justify finetuning, apart from the cost scenario which is the only one that is somewhat universally valid. So after spending some time with my AI buddy Kiro and lot of back and forth, I decided to finetune a model to make it a stoic philosopher.

## Why this?

- After checking a lot of datasets I realised that without good data quality, finetuning was basically a very useless activity.
- For my use case, I had top 3 books available without any copyright issues — all on Project Gutenberg, free to download, total ~1.4 MB of raw text (~200,000 words).
- Another thing — the top books by these authors were written in very different formats:
  1. **Marcus Aurelius — Meditations**, written like a letter to himself. So great for self-reflection questions.
  2. **Epictetus — Discourses + Enchiridion**, written in instruction format. So great when someone is asking for guidance.
  3. **Seneca — Morals / Benefits / Anger / Clemency**, written as letters/conversations to a friend. So very useful when the user is asking for friendly advice.

If our model can blend all three, it can adjust its tone based on what the user needs: stern with someone making excuses (Epictetus), gentle with someone grieving (Seneca), and reflective with someone journaling (Marcus).

So then we started checking the dataset. Downloaded it from Gutenberg, realised lot of useless information was there. So first step was going to be cleaning the scripts.

## Picking a model

But before that, I needed to choose a model to finetune. This led me to a rabbit hole of different types of models and what it exactly means when I download an open-source model.

After exploring and AI buddy recommendation, I decided to go with **Qwen2.5-1.5B-Instruct** primarily because I had 16 GB RAM on my Mac and it would fit. Thought process was: we can go smaller if it still does not fit.

### How to check whether a model fits on your machine

When we say a "1.5B model", we mean it has 1.5 billion **parameters** — the weights the model learned during pre-training. The simple rule-of-thumb for memory is:

- **bfloat16** (most common modern format): every parameter takes **2 bytes**. So 1.5B params × 2 = ~3 GB just for weights.
- **float32** (older default): 4 bytes per param. So same model = ~6 GB for weights.
- **int8 / int4 quantization**: 1 byte / 0.5 bytes per param. Lets you squeeze bigger models in, with some quality loss.

On top of weights you need overhead for the KV cache, activations, and (during training) gradients + optimizer states. For inference, budget ~1.5x the weight size. For training, budget ~6-8x (more on this later).

Here's the scale to keep in mind:

| Model | Parameters | Weight memory (bf16) | Can run on... |
|-------|-----------|---------------------|---------------|
| Qwen2.5-0.5B | 500 M | 1 GB | Anything |
| **Qwen2.5-1.5B** | **1.5 B** | **3 GB** | **My MacBook (16 GB RAM)** |
| Qwen2.5-7B | 7 B | 14 GB | Mid-range GPU |
| Qwen2.5-72B | 72 B | 144 GB | Multi-GPU cluster |

I picked the **-Instruct** variant (not the base) because it already understands chat format. We're going to shape the *style* of responses, not teach the model what a conversation is.

## Establishing the baseline

Now that we had decided the model the next part was to first check the responses **without** actually training or finetuning the model. See what it responds so we have something to compare with later.

I picked **8 evaluation prompts** that I'd reuse at every stage (before SFT, after SFT, after RFT) — same prompts, same generation settings, apples-to-apples comparison.

### Observations (the "before" picture)

The untuned model has a consistent, generic "assistant" voice with clear patterns:

- **Listy** — every answer becomes a numbered list of tips.
- **Over-cushioned** — opens with "It's understandable to feel this way..." almost every time.
- **Escape-hatchy** — routinely defers to therapists, HR, or friends rather than engaging.
- **Vague & universal** — "Everyone has their own unique journey."
- **Risk-averse** — refused the fear-of-death question with "As an AI language model, I cannot provide..." 🚩

None of the answers invoke Stoic frameworks (dichotomy of control, virtue, memento mori, amor fati, impermanence). No short, declarative prose. No reframing. This is the gap we want to close.

### Actual examples from the baseline

> **Q:** "How do I deal with the fear of death?"
> **A:** *"As an AI language model, I cannot provide medical or psychological advice..."*

> **Q:** "I'm anxious about a presentation tomorrow. What should I do?"
> **A:** *"It's completely understandable to feel anxious about an upcoming presentation. Here are some steps you can take to help manage your anxiety:* 1. Practice deep breathing... 2. Visualize success... 3. Get enough sleep..."

> **Q:** "My coworker took credit for my work. How do I react?"
> **A:** *"It's understandable to feel upset... here are some steps:* 1. Stay calm... 2. Document your contributions... 3. Talk to HR..."

You can see the pattern. Every answer is a list of tips. Every opener is some flavour of *"It's understandable..."*. The model is helpful and safe and also very, very bland.

## Understanding the data we have

The next step was to actually understand the data.

What I had: three plain-text files from Project Gutenberg.

| File | Source | Size | Translator |
|------|--------|------|-----------|
| `marcus_meditations.txt` | Gutenberg 2680 | 415 KB | George Long (1862) |
| `epictetus_discourses.txt` | Gutenberg 10661 | 343 KB | George Long |
| `seneca_morals.txt` | Gutenberg 56075 | 621 KB | Sir Roger L'Estrange (1882) |

A small course-correction worth mentioning: my first guesses at Gutenberg IDs were wrong twice. Gutenberg 40082 turned out to be *The Thirty Years' War*, not Seneca. Lesson learned: always `curl` the first few lines before assuming an ID is right.

The most interesting part for me was — even though the data was there, the English was **not modern**. Imagine your model responding in *"thou shalt"* and *"unto thee"*. Useless for an actual user.

### Cleaning and chunking the corpus

So first part was cleaning and chunking the corpus.

**Why cleaning?** The raw Gutenberg files are wrapped in headers, footers, prefaces, translator notes, table-of-contents pages, biographical intros — none of which are actual Stoic philosophy. Train on that and the model will start producing *"This edition first published in..."* mid-sentence.

**Why chunking?** Two reasons:
1. The model has a context length limit (32k tokens for Qwen2.5, but realistically we want training examples way smaller — a few hundred tokens each). We can't feed it an entire book at once.
2. We want the model to learn from *coherent units of thought*. A paragraph is usually one idea — short enough to embed cleanly, long enough to carry an argument. Sentences are too fragmented; whole chapters are too sprawling.

What we needed was **structured format**, and we decided on paragraphs so we could train on them. When we finetune the model, we feed it the training set as **JSONL** — one JSON object per line, each one a self-contained training example (prompt + response). That format is the standard input HuggingFace's trainer expects.

The cleaning logic, in pseudo-code:

```
for each raw text file:
    strip everything outside "*** START ***" / "*** END ***" markers
    skip past biographical intro until first real-content marker
    normalize whitespace, collapse line wraps within a paragraph
    split on blank lines  →  list of paragraphs
    drop paragraphs under 30 words   (kills ToC lines, headings)
    drop ALL-CAPS lines              (kills section headers)
    write each paragraph as one JSON line: {id, author, work, text, word_count}
```

### Result

| Author | Passages | Words | Avg words/passage |
|--------|---------|-------|-------------------|
| Marcus Aurelius | 436 | 63,372 | 145 |
| Epictetus | 276 | 61,412 | 222 |
| Seneca | 326 | 78,805 | 241 |
| **Total** | **1,038** | **203,589** | — |

Each philosopher's natural style shows in average passage length: Marcus's short journal entries, Epictetus's lecture-sized paragraphs, Seneca's longer rhetorical letters.

### Exploring the corpus

The next step after we chunked the data was to start understanding and exploring the corpus we had. What I wanted to check was whether our sample data made sense, so we started checking random passages — does each author come through, are enough varied themes covered, is the prose what we think it is.

Here we also confirmed the **thou/unto problem is real**. `thou` (801 occurrences) and `unto` (433) are both in the top-30 most common words of the corpus. I did not want the model to respond in Shakespearean English. Confirmed plan: ground future training data in this corpus for **ideas**, but generate the actual training prose in modern English using Claude.

### Tagging passages with themes

Next thing we also did was tag the data — labeling each passage with the Stoic themes it touches (death, control, virtue, anger, fear_anxiety, friendship, etc.). 15 themes total, regex-based, multi-label.

Why bother tagging? Because when we generate training questions later (e.g. "How do I deal with anger?"), we need to retrieve a *grounding passage on anger* to feed the answer-generator. Tagging makes that retrieval one cheap dictionary lookup instead of an embedding-and-search pipeline.

**88% of passages got at least one tag, average 2.86 themes each.** A useful sanity check: my first regex for "control" only caught 81 passages — surprisingly low for the *most iconic* Stoic concept. After improving patterns to include Victorian phrasings (*"in our own power"*, *"free will"*, *"what is thine"*), it jumped to 194. Lesson: regex tagging is leaky; treat coverage as a lower bound, not ground truth.

## Designing a system prompt and question bank for SFT

Another thing we needed was to design a system prompt and a question bank for SFT.

### Why a system prompt is needed

A **system prompt** is the model's "identity card" — a block of text injected at the top of every conversation that tells it who it is and how to behave. Every SFT training example carries it, every inference run includes it. If we train on one version of the system prompt and evaluate with a different version, the model will drift. So I put it in one file and use it everywhere.

The Stoa system prompt defines:
- The character (warm but unflinching, in the tradition of Marcus/Epictetus/Seneca).
- The voice (short declarative sentences, simple language, no numbered lists, no deferrals to therapists).
- One safety carve-out: if the user signals self-harm or acute crisis, Stoa drops the philosopher voice and points them to a crisis line.

### How SFT works (mental model)

**Supervised Fine-Tuning** is the simplest kind of fine-tuning. You give the model thousands of `(input, ideal_output)` pairs and train it to predict the ideal output given the input.

**What is loss?**

Loss is just a number that says *"how wrong was the model on this example?"*. Higher loss = more wrong.

A simple example. Suppose the input is `"The sky is"` and the ideal next word is `"blue"`. The model spits out a probability for every possible next word in its vocabulary:

```
"blue"   → 0.10
"green"  → 0.40   ← model thinks green is most likely
"falling"→ 0.05
... thousands more ...
```

The model gave the correct word `"blue"` only 10% probability. Loss for this example is `-log(0.10) ≈ 2.3`. If the model had given `"blue"` 99% probability, loss would be `-log(0.99) ≈ 0.01` — tiny, because the model was nearly right.

So during training, we just want to push that probability up for the right answer. Repeat over thousands of examples, the model gets less wrong.

**How do the weights change?**

Every weight in the model is just a number. The loss is a function of all those numbers. **Autograd** (PyTorch's automatic differentiation engine) computes, for every single weight, how much that weight contributed to the loss — that's the **gradient**.

If a weight had a positive gradient, it means *increasing* it would *increase* the loss. So we go the other way: we *decrease* it slightly. The "slightly" is controlled by the learning rate (more on that in a sec).

In one optimizer step:

```
for each weight w in the model:
    w_new = w_old  -  learning_rate * gradient_of_w
```

That's it. Repeat over many examples and the weights drift toward values that produce lower loss = better predictions.

For our SFT, we don't compute loss on every token — only on the **assistant's tokens**. The model isn't graded on predicting the user's question or the system prompt, only on producing the right Stoa-style reply. This is called *completion-only loss*.

### Generating the questions

So to do SFT we need pairs: **(user question, ideal Stoa-style answer)**. We need lots of them, of varied topics, in the voice we want the model to learn.

We used **AWS Bedrock** to call Claude Sonnet 4.5 and generate sample questions. I tried to keep the questions realistic — like real Reddit posts, not sanitized "training data". This took deliberate prompt engineering: explicitly asking for specific contexts, prior efforts mentioned, contractions, mixed registers.

15 categories tilted toward practical emotional topics (anxiety, anger, grief, relationships) because that's what real users actually bring. Total: **530 questions**.

Sample outputs (these read like real people, not bot prompts):

> *"My neighbor's dog barks at 6am every single day and they don't care. I've asked nicely five times. What now?"*
>
> *"Is it normal that I'm angrier at my husband for dying and leaving me alone than I am sad?"*

### Generating the answers

We also generated answers for them.

**Why generate answers at all? Why not just use Claude live?** This is the question I had to think about most clearly.

The whole point of fine-tuning is to bake a specific voice and behavior **into the weights of a smaller model** that we own and can run cheaply. If we just called Claude every time a user asked a question:
- We'd pay per-call forever.
- We'd be at the mercy of API uptime, rate limits, and model deprecations.
- We couldn't run it offline, on-device, or in a closed environment.
- And we wouldn't *learn anything* about how fine-tuning works — which is the whole point of this exercise.

So we use Claude **once** to generate ~500 high-quality training pairs. After that, our 1.5B model has internalized the voice. We never need to call Claude at inference time.

### What's a "grounding passage"?

Each answer was generated with a **grounding passage**. This is a key idea, so let me unpack it.

A grounding passage is just a real chunk of text from the original corpus (one of those ~200-word paragraphs from Marcus / Epictetus / Seneca) that we hand to Claude *along with* the user question. The prompt to Claude becomes:

> *Here is a question from a real user: "I'm anxious about a presentation tomorrow."*
> *Here is a relevant passage from Epictetus: "Some things are in our power, and some are not. Of things in our power are opinion, pursuit, desire, aversion..."*
> *Now write a 200-400 word Stoa-voiced answer that draws on this passage's reasoning.*

Why bother? Without grounding, Claude generates **plausible-sounding wellness blog prose** — vaguely Stoic vibes but mostly generic life-coach content. With grounding, the answer is anchored in **actual Stoic argument** from a real philosopher. The reasoning has more concrete bite. The model we eventually train will learn to produce *that kind* of reasoning, not the wellness-blog kind.

We pick the grounding passage by matching the question's themes (e.g. an anxiety question gets a passage tagged `fear_anxiety` or `control`). We also use the question's ID as a random seed so the same question always gets the same passage — reproducible.

The answer-generation pseudo-code:

```
for each question:
    themes = question.corpus_themes
    passage = corpus.sample(themes=themes, max_words=400, seed=question.id)
    prompt = ANSWER_PROMPT.format(question=question.text, passage=passage.text)
    answer = claude.call(system=STOA_SYSTEM_PROMPT, user=prompt, temperature=0.8)
    write {question, answer, grounding=passage.id} to examples.jsonl
```

After this, an automated quality audit confirmed: zero truncations, zero too-short answers, zero "consult a therapist" phrases, zero numbered-list answers. Lines that emerged:

> *"You're bleeding before you're cut."*
> *"That's not catastrophe. That's Tuesday."*
> *"Virtue doesn't wait for courage to arrive."*

This is the quality ceiling of our future fine-tuned model. Whatever voice the 1.5B Qwen learns, this is the data it's learning from.

Then a deterministic 90/10 split:
- `train.jsonl` — 477 examples
- `val.jsonl` — 53 examples

## Formatting for the trainer

So all of this was done to actually create our data into a question and answer format ready for the trainer. The trainer expects each example as a chat-style record with system / user / assistant roles:

```
{
  "messages": [
    {"role": "system",    "content": STOA_SYSTEM_PROMPT},
    {"role": "user",      "content": "I'm anxious about my presentation tomorrow."},
    {"role": "assistant", "content": "You're bleeding before you're cut. ..."}
  ]
}
```

I rendered each record through Qwen's chat template and measured token lengths. Every example fit in ~700-800 tokens, max 795. Set training max-length to 900 — comfortable margin, no truncation.

## SFT training with LoRA

Then we decided to train SFT with **LoRA**, since full fine-tuning was not fitting on my local Mac.

### LoRA, with a baby example

Imagine the model contains a single weight matrix `W` that is 1000 × 1000 — that's 1,000,000 numbers. Full fine-tuning means updating all 1,000,000 of them every step. For a real model with hundreds of these matrices stacked in layers, that's billions of numbers to update + gradients + optimizer state. Memory blows up.

The **LoRA insight**: when you adapt a pre-trained model to a new task, the *change* you need to make to `W` isn't random — it has low intrinsic rank. You can approximate `ΔW` (the change) as the product of two skinny matrices:

```
ΔW  ≈  B  ×  A

where:
  W is 1000 × 1000      (1,000,000 numbers, FROZEN)
  A is    8 × 1000      (    8,000 numbers, trainable)
  B is 1000 × 8         (    8,000 numbers, trainable)

The 8 here is the LoRA "rank" r.
```

So instead of training 1,000,000 numbers, we train 16,000 — about **1.6%** of the original. At inference time the model uses `W + B×A`. The original weights stay frozen; only `A` and `B` are learned.

A toy concrete walk-through: if `W` is `[[1, 2], [3, 4]]` (a 2×2 matrix, frozen), and we set rank `r=1`, then:
- `A` is 1×2, say `[[0.1, 0.2]]`
- `B` is 2×1, say `[[0.5], [0.5]]`
- `B × A` = `[[0.05, 0.10], [0.05, 0.10]]` — the "delta" we add to `W`.
- Effective weight: `W + B×A` = `[[1.05, 2.10], [3.05, 4.10]]`.

Train just 4 numbers (the contents of A and B) instead of 4 (well, in this toy case it's the same — but scale this up to 1000×1000 and you train 16k numbers instead of 1M).

In our actual model, we attach a LoRA adapter to the four attention projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`) in **all 28 layers**, with rank `r=16`. Total trainable: about **4.4 million parameters** out of 1.54 billion. **0.28% of the model.**

The result: a 17 MB adapter file that you ship alongside the base model. Anyone with the base Qwen2.5-1.5B and our adapter gets the Stoa-tuned model.

### Running training

I used a **GPU** for training. MPS on the M1 was way too slow (one optimizer step took ~5.5 minutes — at that rate, 3 epochs would take ~16 hours). On the GPU, the entire training run took under 5 minutes.

### What's actually happening during training

Let me walk through one training step with a tiny imaginary example.

Suppose our batch has one example:

```
input:  "I'm anxious about tomorrow."
target: "Anxiety is a wound made before the cut."
```

Step-by-step:

1. **Forward pass** — feed the input through the model. The model produces a probability distribution over the vocabulary at every position. At the first target position, ideally `"Anxiety"` gets high probability. Maybe it gives:
   ```
   "Anxiety"  → 0.02
   "It's"     → 0.30
   "I"        → 0.15
   ... etc
   ```
   The model is currently bad at this — it gave the right word only 2% probability.

2. **Compute loss** — for each target token, take `-log(probability the model gave it)` and sum. Right now the loss is high (because each correct token got low probability).

3. **Backward pass** — autograd propagates gradients backward through the network. For every trainable LoRA parameter, we now know: "if I nudge this number up by a tiny bit, the loss would go up by *this much*". That's the gradient.

4. **Optimizer step** — for each LoRA parameter, take a small step in the direction that *reduces* loss:
   ```
   param_new = param_old - learning_rate * gradient
   ```
   Note: the base model weights (the 1.54 billion frozen ones) get **zero updates**. Only the ~4.4M LoRA parameters move.

5. **Repeat** with the next batch of training examples. Each step nudges the LoRA weights a little. Over many steps the model gets better at predicting Stoa-style answers.

### What's an epoch and what's a learning rate?

**An epoch is one full pass over the training data.** If you have 477 training examples and you've shown the model each one once, that's one epoch. Standard SFT does 2-5 epochs — too few and the model hasn't fully absorbed the dataset; too many and it starts memorizing exact answers (overfitting). I used 3 epochs.

**The learning rate is the size of each step.** Given a gradient, how big a nudge do we take? Too small → training takes forever, model barely moves. Too big → updates are too aggressive, weights overshoot, training becomes unstable, loss spikes.

Typical values:
- **Full fine-tuning:** `1e-5` to `5e-5` (very small — you're updating billions of weights, even a tiny nudge per weight adds up).
- **LoRA fine-tuning:** `1e-4` to `5e-4` (10-30× higher — you're updating only a few million weights, each one needs to move more to actually shift the model's behavior).
- **RFT on top of SFT:** `5e-6` (much smaller — you're polishing, not relearning).

I used `2e-4` for SFT. I also used a **cosine schedule with 5% warmup**: the learning rate ramps up linearly for the first 5% of steps (so the brand-new LoRA adapters don't slam the frozen model with huge noisy gradients on step 1), then smoothly decays toward zero over the rest of training.

### Training results

| Stage | Train loss | Val loss |
|-------|-----------|----------|
| Step 5 (start) | 2.44 | — |
| Epoch 1 end | 2.11 | 2.09 |
| Epoch 2 end | 2.00 | 2.05 |
| Epoch 3 end | 1.97 | **2.05** |

Loss decreased smoothly. Val loss plateaued around 2.05 — we extracted most of the signal the dataset contained. No overfitting (val loss never rose).

### Evaluating the SFT model

I ran the **same 8 baseline prompts** through the SFT-tuned model. Same generation settings as the baseline. Direct apples-to-apples comparison.

| Metric | Baseline | SFT | Δ |
|--------|---------|-----|---|
| Total words | 1,218 | 2,526 | **+1,308** |
| Stoic-term matches | 4 | 5 | +1 |
| Refusals / "consult a therapist" / "sorry to hear" | 8 | **0** | **-8** |
| List/bullet lines | 13 | **0** | **-13** |

The transformation is dramatic. Some examples:

- **Fear of death:** baseline refused with *"As an AI language model, I cannot provide..."*. SFT answers: *"The fear of death is a wound that cuts deep because you think it's real. It's not real. Death is only absence."*
- **Anxious about presentation:** baseline gave a 7-step numbered list. SFT gave prose: *"You're asking the wrong question."*
- **Coworker took credit:** baseline said *"Talk to HR..."*. SFT engages directly.

Stoic frameworks appear naturally now — dichotomy of control, impermanence, memento mori. Memorable lines emerge. The "generic RLHF'd assistant" persona is thoroughly overwritten.

**Issues that surfaced** (and informed the next step):

- **Gender assumption leak.** SFT assumed the coworker in one prompt was male.
- **One possibly-fabricated quote.** SFT said *"Marcus said, 'The universe has no regard for virtue or vice.'"* — Marcus didn't literally say this. It's a paraphrase that reads as a direct quote.
- **Occasional sycophancy / scripted reframes.**
- **Warmth varies between answers.**

These are exactly the kind of issues SFT can't fix easily — you'd need thousands more hand-curated examples — but they're easy to **score** with a reward function. Which leads us to RFT.

## RFT: polishing with a reward function

Next was to use RFT after doing SFT.

### Why was this required?

SFT gets you a model that imitates the training data. But what if you want behaviors that are hard to *write down* but easy to *score*? Things like:
- "Don't fabricate quotes."
- "Don't make gender assumptions."
- "Be appropriately concise — not too long, not too short."
- "Use real Stoic frameworks rather than generic wellness phrases."

You can't easily add 500 SFT examples that demonstrate "don't fabricate quotes" — every example would have to be a near-fabrication that the answer correctly avoids. But you *can* write a function that takes any answer and scores it on: does it cite philosophers? if yes, do those quotes actually appear in the corpus? That's a reward function.

**Reinforcement Fine-Tuning** uses such a reward function to nudge the model: generate multiple candidate answers per prompt, score each, reinforce the high-scoring ones, suppress the low-scoring ones.

### How RFT works (the GRPO flavor we used)

The specific algorithm is **GRPO — Group Relative Policy Optimization**. For each training step:

```
for each prompt in the batch:
    # 1. Generate K different answers (sampling, temperature high so they vary)
    answers = [model.sample(prompt) for _ in range(K)]   # K=4

    # 2. Score each answer with our reward function
    rewards = [reward_fn(prompt, a) for a in answers]    # e.g. [0.62, 0.81, 0.74, 0.55]

    # 3. Compute relative advantage of each answer vs. the group's mean
    mean = average(rewards)
    advantages = [r - mean for r in rewards]             # [-0.06, +0.13, +0.06, -0.13]

    # 4. Update model weights:
    #    increase probability of high-advantage answers
    #    decrease probability of low-advantage answers
    update_policy(advantages)

    # 5. Add a small KL-divergence penalty against the SFT model
    #    so the policy doesn't drift too far from the voice we already have.
```

The clever bit is the *relative* in "relative advantage" — instead of needing a separate value-function model (like classic PPO), GRPO uses the group's own mean as the baseline. Cheaper, simpler, very effective for stylistic fine-tuning.

### Designing the reward function

Six sub-rewards, each in [0,1], combined with weights:

| Signal | Weight | What it measures |
|--------|--------|------------------|
| `stoic_voice` | 0.22 | How many Stoic concepts appear (saturates at 4 hits) |
| `length` | 0.18 | Rewards 300-600 word answers |
| `no_lists` | 0.18 | Penalizes numbered/bulleted lists |
| `no_deferrals` | 0.15 | Penalizes "consult a therapist" / "As an AI" |
| `citation_ok` | 0.15 | If the answer puts something in quotes claiming to cite a philosopher, verifies against the actual corpus via 5-gram overlap. No citations = full score. |
| `addresses_user` | 0.12 | Direct second-person opener |

All cheap regex + n-gram lookups. No LLM-as-judge calls during training. **Sanity check before training:** running this reward on the SFT outputs gave Baseline = 0.48, SFT = 0.84. Strong separation, exactly the signal RFT needs.

### Expanding the prompt pool

Before training, I expanded the prompt pool from 530 → **730 prompts** by generating 200 new ones across 7 RFT-only categories (very_short, philosophical, ambiguous, mixed_emotion, small_stakes, positive, meta). These deliberately stress edges that SFT didn't cover — minimum-context prompts, conflicting-emotion prompts, prompts where the user is doing fine.

### Training

200 GRPO steps, K=4 generations per prompt, temperature 0.9, KL coefficient β=0.04, learning rate 5e-6 (much lower than SFT — we're polishing, not relearning).

Training reward climbed from ~0.78 to ~0.82 over the run. KL stayed at ~0.001 — model staying close to SFT, not collapsing. Completion lengths stayed in the 388-423 token sweet spot.

### Evaluating RFT — and an honest finding

I ran the **same 8 prompts** again through the RFT-tuned model:

| Metric | Baseline | SFT | RFT |
|--------|---------|-----|-----|
| stoic_voice | 0.00 | 0.375 | 0.188 |
| no_lists | 0.75 | 1.00 | 1.00 |
| no_deferrals | 0.78 | 1.00 | 1.00 |
| citation_ok | 1.00 | 0.875 | 0.75 |
| length | 0.11 | 0.98 | 0.94 |
| addresses_user | 0.50 | 1.00 | 1.00 |
| **total** | **0.480** | **0.841** | **0.774** |

Honest finding: **on the 8 eval prompts, RFT scored slightly lower than SFT on our own reward function.** This is interesting and worth digging into.

Why? Three things:
1. **8 prompts is a tiny sample.** During training on the full 720-prompt pool, reward was climbing. The 8 eval prompts just don't reflect the training distribution.
2. **`stoic_voice` is lexical, not semantic.** It rewards explicit keywords like "virtue", "Marcus", "dichotomy of control". The RFT model produced more *distilled* Stoic reasoning — *"None of these things actually happens to people who live well"* — which is **more authentic** Stoicism but scores lower on regex.
3. **RFT became more cautious with quotes.** Good behavior, occasionally penalized by our strict citation checker.

Qualitatively, the RFT outputs are arguably **better** than SFT — tighter prose, less name-dropping, more direct engagement with the user's specific worry. But our reward function couldn't fully capture that.

This is the textbook **specification gaming** lesson: a reward function is a *hypothesis* about what "good" means. Ours captured the surface features of Stoic voice but missed the deeper reasoning quality. The model dutifully optimized exactly what we asked for. RFT works — our spec was incomplete.

### Takeaway

SFT transformed the voice (0.48 → 0.84 — massive). RFT on top polished the outputs in ways the eye likes but our regex couldn't fully measure. Both adapters are useful artifacts. Which one to ship depends on what you're optimizing for.

## What I actually have now

Three artifacts, each ~17 MB:
- `models/stoa-sft-v1/` — backup of SFT adapter
- `models/stoa-sft/` — current SFT adapter
- `models/stoa-rft/` — RFT adapter trained on top of merged-SFT base

To run inference, you load:
1. Base Qwen2.5-1.5B-Instruct (~3 GB).
2. The SFT adapter (or merge it in for RFT).
3. The RFT adapter on top of merged-SFT, if using RFT.

Total memory at inference: ~4-5 GB on bf16. Runs fine on the M1.

## What I'd do differently

- **Build a semantic `stoic_voice` reward** (embed the answer, compare to a reference set of Stoa-voiced text). That likely flips the RFT-vs-SFT result. Skipped because it costs an LLM/embed call per sample per training step.
- **Run RFT longer.** 200 steps was enough to see the mechanism; for production you'd want 1k+.
- **Diversify SFT openers.** ~28% of SFT answers open with `"You're..."` — that's the Stoic voice we asked for, but it's a monoculture. RFT could explicitly diversify openings.
- **Try a 0.5B model** to see whether the same approach works at half the size — would be useful for on-device deployment.
