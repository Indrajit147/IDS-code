# System and threat model (draft of the paper's Section III)

Draft for the IEEE IoT-J outline's "System and threat model" section, including the privacy analysis that decision D5 asks for. Every byte count and rate below is measured in the project's runs, and its source is given; everything about what an adversary could infer is **argued, not measured**, because no attack was run. The paper must keep that distinction.

## A. System model

**Parties.**
- **Edge gateways.** K = 15 gateways, one per IoMT site: a ward, a lab or a home-care hub. Each sees its own network flows.
  - The sites are non-IID. Each class's share comes from a Dirichlet(α = 0.5) draw, and device captures belong to one site only.
  - Each gateway runs the edge ensemble (XGBoost + LightGBM with a calibrator) on every flow.
  - Each holds its own labelled flows for training, in six windows t0–t5.
- **One cloud server.** It holds:
  - D_pub, 90,429 public flows, used as inputs only. Its labels are never used in training. The simulation reads them for one diagnostic table and to decide which window each row enters.
  - the distillation aggregator and the calibrator g_t;
  - the gate thresholds τc and τd;
  - the LLM verifier.

**Data flow per window** (DA-FedDistill, E3; E4 adds drift-triggered updates):
1. **Local update.** Each gateway updates its edge model incrementally on the window's labelled flows plus a class-balanced rehearsal buffer. Raw flows and labels stay on the gateway.
2. **Upload.** Each gateway sends its soft predictions on the D_pub rows of the classes introduced so far, plus drift features (Section C).
3. **Server step.** The server aggregates the soft predictions into pseudo-labels on D_pub. It fits g_t, re-tunes τc and τd for the escalation budget, and sends g_t, the thresholds and each client's drift score back down.
4. **Inference.** Each gateway labels flows locally. A flow whose confidence or booster disagreement crosses the gate, or that arrives while ADWIN's drift hold is active, is *escalated*: its evidence goes to the cloud verifier.

## B. Threat model

- **Server: honest-but-curious.** It runs the protocol as specified, but may try to infer anything about a site's traffic, labels or devices from what it receives. Disclosure to the server is the subject of Section C.
- **Clients: honest.** Poisoning, Byzantine updates and free-riding are **out of scope and not evaluated**. The aggregation is a plain class-balanced mean with no robust rule, so one site sending crafted soft labels could shift the pseudo-labels for everyone. The paper must state this as a limitation.
- **Network: protected.** Links are assumed to be protected by transport encryption (TLS). Eavesdroppers are out of scope.
- **The LLM: self-hosted.** It runs on infrastructure the operator controls. In the experiments, Gemma 2 2B and Mistral 7B run with open weights on a Kaggle GPU, and no third-party API sees any data. A hosted API would add a third party that sees every escalation payload (Section C.3).
- **Data sensitivity.** Flow records are network metadata: packet counts, rates, flag counts and protocol indicators. They contain no clinical data. They can still reveal which devices a site runs, when, and what attacks it faces. Whether this metadata counts as protected health information depends on the jurisdiction and deployment, and **this draft makes no compliance claim**.

## C. What leaves a gateway, and what it could reveal

| channel | method | content | size (full pool, measured) | what the server could infer (argued) |
|---|---|---|---|---|
| soft labels | E3, E4 | each booster's 6-class probabilities on every D_pub row of the classes introduced so far | 4.34 MB per client per round at t5 as float32; 338.6 MB up over the run; 16.6 MB as 8-bit + zlib (M7) | the client model's behaviour on public inputs, hence the site's label mix (a site missing a class gives it low probability) and a distilled copy of its model |
| drift features | E3, E4 | booster disagreement and entropy (on D_pub, so already derivable from the soft labels) and the share of the client's features over the PSI alert | a few scalars per round | whether the site's own traffic shifted this window |
| downlink | E3, E4 | g_t, τc, τd, the client's drift score | 392 bytes per client per round (M7); the drift score adds 4 bytes (M9) | nothing, as it flows to the client |
| escalation payload | E5 | the edge's label and its 6 probabilities; each booster's label; disagreement; the top-5 TreeSHAP features with their signed attributions **and the flow's raw values**; class medians of those 5 features on D_pub; the client's recent predicted class shares; the drift flag and score | about 1,020 bytes per escalated flow; 75–84 escalations per 1,000 flows on the full stream at the gate's budget (M11) | the escalated flow's key feature values, and the site's recent attack mix |
| model weights | E2 (baseline) | the full MLP each round | 57 kB each way per client per round (M8) | the usual gradient and weight leakage of FedAvg |
| gradient histograms | E7 (baseline) | a gradient and a Hessian histogram per new tree node | 760–3,200 times E3's uplink (M13, an upper bound) | per-bin gradient sums over the site's own rows, more direct than soft labels on public data |

**What never leaves a gateway in E3/E4:** raw flows, labels, the rehearsal buffer and the model parameters. Feature selection (E4's top-K by mutual information) is computed and kept locally (M9).

### C.1 Soft labels on public data

E3 and E4 share less than weight-based FL in kind: they send outputs on *public* inputs, not parameters or gradients computed on private rows. This is the property the design relies on, and it is the reason the paper can compare it favourably with E2 and E7.

It still discloses something:
- **The model.** The server receives enough input–output pairs to distil a copy of each site's model on D_pub's support. Distillation does exactly that by design, so model confidentiality from the server is not offered.
- **The label mix.** A site that has never seen a class, or sees it rarely, assigns it low probability, so its label mix is inferable from its soft labels. In this setup the non-IID split makes site label mixes strongly different (Dirichlet α = 0.5).
- **Membership.** Inference about individual private training flows is plausible in principle, through the model's behaviour near public points. It is weaker than with gradients, but it was **not measured**.

### C.2 Drift features

The PSI share is one number per client per window, computed on the client's own flows. It reveals whether that site's traffic distribution moved, which the server could also partly read from changes in the soft labels.

**Open, and must be settled in the text:** in the simulation, ADWIN's drift score is computed from each client's per-row disagreement and entropy, and only the 4-byte score is charged, on the downlink. A deployment must either run ADWIN at the gateway and upload only the score, or upload the per-row signals; the second is far larger and discloses the inference stream's uncertainty profile. The paper should state the first as the deployed design and describe the simulation as its equivalent. This is my recommendation, not a measured result.

### C.3 Escalation payload (the largest per-flow disclosure)

Each escalated flow sends its top-5 features with their actual values, plus the site's recent class shares. That is real traffic data from the site, not a statistic over public rows, so it is the channel to discuss most carefully.

Three properties keep its scale bounded:
- **Gating.** Only escalated flows are sent: 7.5–8.4% of the stream (75–84 per 1,000 flows, M11), against all of them for E6, which is always-on.
- **Feature subset.** Only 5 of 43 features are sent.
- **Self-hosting.** The verifier runs on the operator's infrastructure.

Mitigations the paper can name, **none tested here**: send SHAP ranks and signs without raw values, coarsen the values to the class-median scale, drop the class shares, or run the verifier on-premises at each site.

Given M11's results, the paper should also say plainly that this disclosure currently buys nothing. The verifier didn't improve detection: boundary macro-F1 was 0.6726 against the edge's 0.6738 over three seeds, with an identical false-positive rate. Gemma's narratives also named no operator action (H4 (c), automatic items).

## D. What the system does not provide

These must be stated as limitations:
- No differential privacy on soft labels or escalations.
- No secure aggregation: the server sees each client's soft labels individually.
- No encryption beyond transport.
- No robustness to poisoned soft labels.
- No defence against a malicious server.

Adding DP to the soft labels would cost accuracy that wasn't measured. The 8-bit quantisation measured in M7 (−0.0002 on the full pool) is the only perturbation of the uplink tested.

## Sources in this repository

- M7 (`reports/m7/full/`): E3 communication.
- M8 (`reports/m8/full/`): FedAvg bytes.
- M9 (`reports/m9/full/`): drift score bytes, local top-K.
- M10 (`reports/m10/full/`): the SHAP payload without centroids, about 175 bytes.
- M11 (`reports/m11/full/VERIFY.md`): payload 1,020 bytes, calls per 1k flows, verifier results.
- M13 (`reports/m13/full/`): E7 uplink bound.
- M14 (`reports/m14/full/`): three-seed verifier results.
- README D2 for D_pub, and D5 for this section's scope.
