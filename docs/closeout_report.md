# Cost-Efficient and Accurate LLM Inference at the Edge

**Final Report for AI2C Seed Grant**  
**Period of Performance:** October 1, 2025–September 30, 2026  
**Principal Investigator:** Gauri Joshi, Professor of Electrical and Computer Engineering, Carnegie Mellon University

## 1. Research Objectives and Overview

State-of-the-art language models with frontier reasoning capabilities often possess trillions of parameters, making them resource-intensive and infeasible for deployment in edge applications.

Many smaller and specialized models can often match or outperform frontier language model capabilities, demonstrating the promise of efficiency and low resource requirements at the edge.

A multitude of Army applications demand resource-constrained language model inference, including but not limited to:

- Forward-deployed decision-making systems
- Power-constrained equipment
- Autonomous devices

In this project, we examined language model inference efficiency through:

- Query routing
- Model capability representation
- Model fusion

These approaches aim to better harness capabilities across language models, improve inference efficiency, and augment existing systems with generative capabilities.

Below, we present our research directions and provide documentation of our findings and resulting publications at premier machine learning venues.

## 2. Investigation Outcomes and Findings

### Query Routing

Given a diverse pool of language models with varying capabilities and specializations, models can be hosted at geographically distributed computational nodes across the edge-cloud spectrum. A crucial aspect of harnessing diverse language model capabilities involves mapping query requirements to language model abilities.

Language query routing canonically learns representations that enable such mapping for improved inference efficiency. We examine four directions for improving query routers:

1. Improving robustness to time-varying query distributions
2. Generalizing router training to federated and private-data settings
3. Scaling routers to thousands of language models by capturing model behavior through embeddings
4. Performing routing in latency-critical applications

#### 2.1 Robust Query Routing

**Resources:** Paper Link \| arXiv \| Project Page

In *“ProxRouter: Proximity-Weighted LLM Query Routing for Improved Robustness to Outliers”* (AISTATS 2026, [1]), we identify poor generalization of nonparametric language query routers to out-of-distribution queries presented during test time.

This issue is further exacerbated by limited diversity in training data, which is expensive to collect and difficult to keep current with ever-evolving use cases. We propose **ProxRouter**, which uses an exponentially tilted aggregation approach to balance bias and variance in performance and cost estimates for nonparametric routers, improving their robustness to outliers.

ProxRouter demonstrates a statistically grounded and efficient approach for improving router robustness to evolving query distributions and applications across Army infrastructure.

#### 2.2 Distributed Router Training

**Resources:** arXiv \| Code Repository \| Project Page

*“Federate the Router: Learning Language Model Routers with Sparse and Decentralized Evaluations”* (NeurIPS 2026, [2]) challenges the assumption that centrally available training data is required and proposes a decentralized training paradigm for language query routers.

Because security-constrained and multi-level access protocols for sensitive Army data seldom permit centralized data collection and router training, our approach presents a privacy-preserving and performant method to build routers using existing but distributed evaluation data.

Our framework supports both parametric and nonparametric routers under heterogeneous query distributions and non-uniform language model coverage.

#### 2.3 Scalable Model Embeddings

**Resources:** arXiv \| Code Repository \| Project Page

In *“LOCUS: Low-Dimensional Model Embeddings for Efficient Model Exploration, Comparison, and Selection”* (Preprint 2026, [3]), we address the challenge of scaling language query routers to thousands of language models.

Typically, query routers are trained once for a given pool of language models using diverse query-model evaluation data. To generalize query routing to a dynamic model pool—which may arise in systems with active updates and local node failures—we introduce model embeddings that capture model behavior through low-dimensional vectors.

These embeddings capture semantic information about model capabilities. A common decoder uses the encoded information in model embeddings to estimate the probability of response correctness on unseen queries.

Moreover, the embedding-space geometry reflects model similarity, enabling downstream applications including:

- Model comparison
- Model portfolio selection
- Resilient fallback-model selection

#### 2.4 Latency-Critical Inference and Query Routing

**Resources:** arXiv \| Code Repository \| Project Page

Language query routers select appropriate models based on generative performance and cost, but many Army applications require short response duration and latency-critical inference.

Estimating generation latency for routing decisions requires consideration of present workload information and serving-framework scheduling policies at language model deployments, such as:

- vLLM [4]
- SGLang [5]

In *“Mind the Delay: Latency-aware Query Routing for Heterogeneous LLM Pools”* (Preprint 2026, [6]), we introduce a **Batch Evolution Aware (BEA)** latency estimator. BEA provides fine-grained latency estimates for incoming queries with negligible computational overhead.

The latency estimates produced by BEA are then used in our latency-aware router, **BEAR**, which jointly optimizes accuracy, cost, and latency when selecting the best language model instance to generate a response to a query.

### Model Fusion

In *“Navigating the Accuracy-Size Trade-Off with Flexible Model Merging”* (ICLR 2026, [7]), we propose **FLEXMERGE**, a novel data-free model-merging framework that flexibly generates merged models of varying sizes. The framework spans the spectrum from a single merged model to retaining all individual fine-tuned models.

FLEXMERGE treats fine-tuned models as collections of sequential blocks and progressively merges them using any existing data-free merging method, halting at a desired size.

We systematically explore the accuracy-size trade-off exhibited by different merging algorithms in combination with FLEXMERGE.

Model merging can be performed before hosting models on deployed hardware. This minimizes the memory footprint of language models while retaining performance, which is critical for operational applications with limited resources across Army infrastructure.

**Resources:** Paper Link \| arXiv \| Code Repository

## 3. Future Work: Efficient Orchestration of Multi-Agent Systems

Modern applications increasingly combine language models with retrieval, tools, verification, iterative refinement, and other forms of test-time computation.

These workflows generate dependent and temporally related inference calls:

- Some branches execute concurrently.
- Others pause while awaiting tools.
- Later stages may reuse or extend context produced earlier.

Their end-to-end performance therefore depends not only on the language models used, but also on:

- How computation is scheduled and batched across serving instances
- How intermediate state and context memory are managed throughout execution

We envision a systems-aware orchestration layer that jointly coordinates these decisions efficiently by selectively using:

- Stronger models
- Additional context or reasoning
- Verification
- Redundant computation where it provides the highest performance improvement

This broadens multi-model inference from selecting an appropriate model for a single query to jointly coordinating models, inference strategies, and serving resources across an entire task. The goal is to improve end-to-end efficiency and reliability.