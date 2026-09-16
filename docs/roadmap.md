# Roadmap

## Near term: reproducibility and validity

- Add the matched off-state ablation.
- Pin a tested dependency lock file and document hardware/runtime.
- Add small redistributable fixture images for deterministic smoke tests.
- Document reference hardware and runtime measurements.

## Research expansion

- Evaluate stratified ImageNet subsets and additional architectures.
- Compare removal policies: black, blur, inpainting, and generative replacement.
- Measure explanation stability across seeds.
- Compare the online agent with an exhaustive offline oracle.
- Evaluate source inspection and visual grounding through ablations.
- Extend beyond the implemented LIME, sparse LIME (Lasso), and Kernel SHAP providers.

## Agent-controlled configuration

The current system loads method parameters from validated profiles. The planned
configuration layer will expose bounded parameter schemas to the agent, require
a pre-execution rationale, validate safe ranges, and record every configuration
attempt. Evaluation should compare:

1. fixed global parameters;
2. human-selected profiles;
3. agent-selected methods with fixed parameters;
4. agent-selected methods and bounded parameter configurations.

Implementation status: planned.

## Engineering maturity

- Add typed XAI provider interfaces and isolate OpenAI-specific transport.
- Extend the offline control-loop fixtures to more model and failure scenarios.
- Introduce structured logging and resumable experiment manifests.
- Add runtime cost and deployment documentation.
