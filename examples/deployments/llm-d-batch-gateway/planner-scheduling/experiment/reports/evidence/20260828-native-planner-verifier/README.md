<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Planner verifier fixture

This is a minimized, sanitized projection of canonical run
`20260828T213549Z-planner-native-1e3ff8`. It contains only fields and samples
consumed by `workloads/verify_native_planner_e2e.py`; host paths, namespace,
pod identities, registry details, unrelated metrics, and unrelated log lines
were removed.

The evidence is historical. It uses the August 2026 `VllmDecodeWorker`
component and `qwen3-0-6b-batch-vllmdecodeworker` adapter names, before the
September rebase adopted the short `worker` topology.

From the experiment root, replay the assertions with:

```bash
python3 workloads/verify_native_planner_e2e.py \
  --run-dir reports/evidence/20260828-native-planner-verifier/run \
  --evidence-dir reports/evidence/20260828-native-planner-verifier/evidence \
  --worker-component VllmDecodeWorker \
  --adapter-name qwen3-0-6b-batch-vllmdecodeworker \
  --expected-lease-duration-seconds 60
```
