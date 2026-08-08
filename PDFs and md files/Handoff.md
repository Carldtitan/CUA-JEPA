JEPA for Computer Use — Codex Handoff

1. What I am trying to build

I want to test whether JEPA-style predictive representation learning can improve computer-use agents.

The immediate hypothesis is:

Current computer-use agents are typically built by taking a strong pretrained VLM, then post-training it on GUI demonstrations and sometimes RL. I want to test whether the same VLM becomes a better computer-use model if it first learns software dynamics from large amounts of state-action-next-state interaction data using a JEPA-style objective.

The commercial thesis is not simply “JEPA for computer use.” The hoped-for advantage is:

better performance on unseen software,

less task-specific SFT data,

less RL / demonstration data,

better sample efficiency,

potentially lower training cost for adapting to enterprise software.

The first experiment should be cheap and falsifiable.

2. Why computer use is interesting for JEPA

Computer environments naturally generate:

state_t
+
action_t
+
state_t+1

Example:

Gmail inbox
+
click Compose
+
Compose window opens

This is useful because the environment automatically gives the supervision signal: the resulting next state.

Unlike robotics:

computers are cheap to reset,

virtual desktops can run in parallel,

failures are inexpensive,

interaction data can be generated automatically,

GUI state can sometimes include structured information such as DOM / accessibility trees in addition to pixels.

This maps well onto Yann LeCun's world-model framing, where learning depends heavily on diverse (state, action, resulting state) sequences.

3. What JEPA actually is

JEPA = Joint Embedding Predictive Architecture.

The core idea is:

x -> encoder -> s_x
y -> encoder -> s_y

s_x (+ optional latent z)
        |
        v
     predictor
        |
        v
predicted s_y

The model predicts the representation of y, not y itself.

Generic form:

s_x = Enc_x(x)
s_y = Enc_y(y)
predicted_s_y = Pred(s_x, z)

The training objective makes predicted s_y compatible / close to the true s_y.

Important details from LeCun's 2022 paper:

The two encoders do not have to be identical.

They do not have to share weights.

x and y may even be different modalities.

JEPA is non-generative: it predicts in representation space instead of generating every pixel/token.

An optional latent variable z can represent information about the future that cannot be inferred from x.

Collapse prevention is essential.

LeCun emphasizes non-contrastive regularization such as VICReg-like variance/covariance constraints.

JEPA is a component of a larger autonomous-agent architecture, not the entire agent.

4. JEPA vs LLM vs Transformer

These are not equivalent categories.

Transformer

A neural-network architecture.

It can be used inside:

LLMs,

VLMs,

JEPA models,

many other systems.

LLM

A large model focused primarily on language/token sequences.

Examples:

GPT-family models

Claude-family models

Llama-family models

Qwen language models

Most modern LLMs use Transformer architectures.

JEPA

A family of predictive architectures / training approaches where the model learns to predict representations rather than raw outputs.

Examples:

I-JEPA

V-JEPA

V-JEPA 2

hierarchical JEPA variants

A JEPA can itself use Transformer blocks.

World model

A broader functional concept.

A world model is something that represents or predicts how an environment behaves.

Possible world models include:

generative next-frame models,

latent dynamics models,

JEPA-style predictive models,

classical state-transition models.

JEPA is one approach to building a world model.

5. How computer-use agents are normally built today

Typical pipeline:

general pretrained VLM
        |
        v
GUI / grounding SFT
        |
        v
trajectory SFT
        |
        v
optional RL / preference optimization
        |
        v
computer-use model
        |
        v
agent harness
        |
        v
actual browser / desktop

At runtime:

user goal
   |
   v
screenshot + history
   |
   v
VLM
   |
   v
action: click / type / scroll
   |
   v
executor
   |
   v
computer
   |
   v
new screenshot
   |
   +------ repeat

The central policy objective is approximately:

(goal, current_state, history) -> next_action

Example:

Goal: Download invoice
Current screen: Salesforce customer page
Target action: click "Invoices"

This teaches:

What should I do?

It does not explicitly require learning:

What state will this action cause?

A sufficiently strong policy model can still learn some dynamics implicitly, but that is not the explicit pretraining target.

6. The proposed JEPA computer-use experiment

The core research question:

Does explicitly learning GUI state transitions in latent space make a pretrained VLM easier to turn into a strong computer-use model?

Baseline branch

same open VLM checkpoint
        |
        v
computer-use SFT
        |
        v
evaluate

JEPA branch

same open VLM checkpoint
        |
        v
JEPA-style GUI transition training
        |
        v
same computer-use SFT
        |
        v
evaluate

Everything after the JEPA stage should be kept as identical as possible.

The important comparison is:

baseline SFT model
vs
JEPA-pretrained + same SFT model

7. The cheapest JEPA architecture to test first

For a small-budget experiment, do not start with:

hierarchical JEPA,

latent-variable z,

live planning,

a separate large world model,

training from scratch,

RL.

Use a very small action-conditioned JEPA pretraining setup.

Data

x = (screen_t, action_t)
y = screen_t+1

Architecture

screen_t
   |
   v
VLM visual / multimodal backbone
   |
   v
z_t
   |
   +---- action embedding
              |
              v
        small predictor
              |
              v
      predicted z_t+1


screen_t+1
   |
   v
target / shared encoder
   |
   v
actual z_t+1

Training makes:

predicted z_t+1 ~= actual z_t+1

Use:

one small pretrained VLM,

LoRA / PEFT rather than full-parameter training,

a tiny action encoder,

a small 2-layer MLP or small Transformer predictor,

VICReg-style regularization or another valid anti-collapse mechanism.

For the first experiment, no latent z.

After JEPA training:

discard / ignore predictor
        |
        v
keep adapted VLM backbone
        |
        v
computer-use SFT

At inference, the JEPA predictor does not need to run.

8. Why JEPA pretraining could help

The strongest possible advantage is sample efficiency.

Normal computer-use training cares about examples like:

instruction
+
current screenshot
+
correct action

These examples can be expensive because they need:

successful trajectories,

curated tasks,

action labels,

reward design,

RL environments,

human or synthetic demonstrations.

JEPA-style transition training can learn from nearly every interaction:

before
+
action
+
after

Even if the action was not useful.

Examples:

click correct button -> state changes
click wrong button -> wrong state
click dead area -> no change
scroll -> viewport changes
type text -> field changes
close modal -> modal disappears

The environment produces the supervision automatically.

Potential benefits to test:

Less SFT data required

Better generalization to unseen applications

Higher task success at the same SFT budget

Potentially cheaper adaptation to enterprise software

Possibly lower compute than generative future-state prediction, because JEPA predicts latent representations rather than every pixel/token

Only 1–3 should be treated as primary claims initially. The compute advantage must be measured, not assumed.

9. Three main JEPA computer-use architectures we discussed

Architecture 1 — JEPA pretraining -> normal policy

raw GUI transitions
        |
        v
JEPA predictive pretraining
        |
        v
better representation
        |
        v
SFT / RL
        |
        v
normal CUA policy

Best first experiment.

Pros:

cheapest runtime,

simplest,

easy A/B comparison,

JEPA predictor can be removed at inference.

Architecture 2 — Joint JEPA + policy

One model learns both:

What should I do?
AND
What state will that action cause?

Conceptually:

screen + goal
      |
      +----> action head
      |
      +----> JEPA next-state predictor

Loss:

total_loss =
action_loss
+
lambda * JEPA_loss

Pros:

action learning and dynamics learning may reinforce each other.

Cons:

harder to train,

harder to debug,

more expensive than architecture 1.

Architecture 3 — Separate VLM policy + JEPA world model

Runtime planning system:

goal + current state
        |
        v
VLM policy proposes actions
     /    |    \
    A     B     C
    |     |     |
    v     v     v
JEPA world model predicts future latent states
     \    |    /
        scorer
          |
          v
     choose action
          |
          v
       execute

This explicitly asks:

What happens if I do A, B, or C?

Pros:

possible planning / safety advantage,

useful for irreversible or expensive actions.

Cons:

highest inference cost,

more moving parts,

requires scoring / cost model,

not suitable for the $100 first experiment.

10. LeCun's Mode 1 vs Mode 2 framing

This is useful for thinking about computer-use agents.

Mode 1

Reactive policy:

state -> action

No explicit planning through the world model.

This resembles today's computer-use VLM loop.

Mode 2

Planning with a world model:

propose action sequence
        |
        v
predict future states
        |
        v
evaluate cost
        |
        v
search for better sequence
        |
        v
execute first action

LeCun also proposes that expensive Mode-2 reasoning can be used to train a fast Mode-1 policy.

Possible future computer-use architecture:

JEPA world model + planner
        |
        v
generate strong trajectories
        |
        v
distill into fast VLM policy

This is interesting later, but not the first experiment.

11. Hierarchical JEPA idea for software

Not for the first experiment, but potentially important later.

Possible hierarchy:

low level:
click -> dropdown opens

middle level:
fill form -> application becomes ready

high level:
complete onboarding -> account activated

The idea is that:

low-level representations retain detailed short-term information,

high-level representations discard details and support longer-horizon prediction.

This maps naturally to multi-step computer workflows.

12. Data to use for the first experiment

Primary candidate:

OpenCUA / AgentNet

Use a small subset.

Create two views from the same trajectories.

JEPA view

screen_t
action_t
screen_t+1

SFT view

instruction
screen_t
history
target_action_t

Suggested $100-scale starting point:

5k–10k GUI transitions for JEPA

1k–2k SFT action examples

300–500 held-out eval examples

Prefer holding out entire applications if possible.

Example:

train:
Chrome
LibreOffice
Files
Settings

test:
Thunderbird
GIMP
some unseen enterprise-like app

The objective is to see whether JEPA helps transfer to unseen UI dynamics.

13. Training environment

For the first experiment, training can be entirely offline.

You do not need a live browser or desktop during training.

AgentNet files
   |
   v
PyTorch Dataset
   |
   v
GPU

Possible later interactive evaluation:

OSWorld

DesktopEnv

browser / desktop VM environments

But with the current budget, avoid expensive live RL.

14. Model choice

Use a small open VLM.

Current candidate:

Qwen3-VL-2B-Instruct

Reasons:

already has vision + language,

already supports visual grounding,

already has computer-use capabilities,

much cheaper than larger models,

suitable for LoRA.

Do not train a foundation model from scratch yet.

The purpose of the first experiment is to isolate:

Does JEPA-style transition pretraining add value?

Training from scratch would confound the experiment and massively increase cost.

15. Training stack

Recommended stack:

Python
PyTorch
Hugging Face Transformers
PEFT / LoRA
TRL or standard Trainer for SFT
Weights & Biases or simple local logging

For JEPA:

Qwen3-VL backbone
+
LoRA
+
small action encoder
+
small predictor
+
VICReg-style anti-collapse loss

No distributed training unless needed.

No RL initially.

16. Evaluation plan

There should be three experiments.

Experiment 0 — zero-shot original VLM

Qwen3-VL-2B-Instruct
        |
        v
no training
        |
        v
held-out computer-use eval

Record:

action accuracy,

coordinate accuracy,

invalid-action rate,

per-application performance.

Experiment 1 — normal baseline

original Qwen
      |
      v
SFT
      |
      v
evaluate

Call this score B.

Experiment 2 — JEPA branch

original Qwen
      |
      v
JEPA transition pretraining
      |
      v
same SFT
      |
      v
evaluate

Call this score C.

Main result:

C vs B

The strongest evaluation is not just final score.

Compare learning curves.

Example:

SFT examples

Baseline

JEPA-pretrained

250

?

?

500

?

?

1,000

?

?

2,000

?

?

The best possible result would be something like:

JEPA-pretrained model reaches baseline performance using 4x less task-specific data.

That is more commercially meaningful than a tiny absolute benchmark gain.

17. $100 budget strategy

The $100 goal is not to prove a frontier result.

The goal is to determine whether there is a signal.

Suggested budget philosophy:

$10–15  zero-shot baseline + debugging
$20–25  baseline LoRA SFT
$30–40  JEPA transition training
$20–25  same post-JEPA SFT
remainder evaluation / reruns

These are spending caps, not guaranteed exact prices.

If 2B is still too expensive:

reduce model size,

reduce image resolution,

freeze more layers,

use LoRA,

use fewer transitions,

run only one or two epochs,

focus on a small held-out application set.

18. What NOT to build right now

Do not start with:

a new foundation model from scratch,

billions of transitions,

RL,

hierarchical JEPA,

latent-variable planning,

a separate JEPA simulator at runtime,

multiple large models,

full OSWorld benchmark sweeps,

large-scale enterprise data collection.

The first goal is simply:

Does JEPA-style transition pretraining produce a measurable benefit over normal VLM -> SFT training?

19. Business thesis if the experiment works

Possible company positioning:

Build computer-use foundation models that learn how software behaves before they learn specific tasks.

Potential customer value:

adapt to proprietary enterprise software with fewer demonstrations,

lower integration / training cost,

better performance on unseen applications,

more reusable foundation model across many software environments.

Possible future product:

customer gives:
proprietary software
+
small number of demonstrations

our model:
already understands generic software dynamics
        |
        v
requires far less task-specific training
        |
        v
working enterprise agent

The customer does not care about JEPA.

They care that the model works with less custom data and less tuning.

20. Important caution

Do not overclaim.

What is already plausible / supported:

learning forward GUI dynamics can improve downstream computer-use models,

state-action-next-state data is cheap to generate,

JEPA is designed to learn predictable abstract representations,

LeCun explicitly argues for learning world models from observation and active agency.

What is NOT yet proven:

JEPA is better than generative GUI transition prediction,

JEPA will definitely improve computer use,

JEPA will reduce compute,

JEPA will solve long-horizon planning,

every CUA company needs an explicit JEPA world model.

The experiment exists precisely to test these claims.

21. EF context

I have an Entrepreneur First final-round path after winning an EF hackathon.

For EF, the pitch should not sound like:

"I need funding to research JEPA for years."

Better:

"I want to build a new kind of computer-use foundation model. Current systems rely heavily on post-training general VLMs with GUI demonstrations and RL. Software gives us enormous amounts of cheap state-action-next-state supervision. I want to test whether JEPA-style predictive pretraining lets a model learn software dynamics first, so it can adapt to unseen enterprise applications with much less task-specific data."

EF is primarily evaluating the founder, not just the idea.

The strongest founder-level framing is:

clear technical taste,

willingness to run fast experiments,

willingness to kill the idea if evidence is weak,

focus on measurable economic advantage rather than architectural novelty for its own sake.

22. Immediate Codex task

Build the smallest possible reproducible A/B experiment.

Goal

Compare:

Qwen VLM -> SFT

against:

same Qwen VLM -> JEPA transition training -> same SFT

Requirements

Use the same starting checkpoint.

Use the same SFT dataset.

Use the same held-out test set.

Keep action format identical.

Keep training budget as similar as possible.

Log losses and evaluation metrics.

Add a learning-curve experiment if budget permits.

Use LoRA / PEFT.

Do not add RL.

Do not build runtime planning.

Minimal repository shape

jepa-cua/
├── data/
│   ├── prepare_agentnet.py
│   ├── transitions.py
│   └── sft_dataset.py
├── models/
│   ├── action_encoder.py
│   ├── jepa_predictor.py
│   └── qwen_jepa_wrapper.py
├── train/
│   ├── train_jepa.py
│   └── train_sft.py
├── eval/
│   ├── eval_actions.py
│   └── compare_runs.py
├── configs/
│   ├── baseline.yaml
│   └── jepa.yaml
└── README.md

First milestone

Get a tiny run working on approximately:

100–500 transitions

Verify:

tensor shapes,

action encoding,

target encoding,

JEPA loss decreases,

representation does not collapse,

checkpoint saves and reloads.

Then scale to the actual small experiment.

23. Core sentence to remember

I am not trying to replace VLMs with JEPA. I am testing whether a VLM that first learns software dynamics through JEPA-style predictive representation learning becomes a more data-efficient and general computer-use model.