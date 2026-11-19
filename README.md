# Sisyphus Never Stops

**Task-Agnostic Object Reset Toward Unattended Real-Robot Manipulation**

Anonymous project page for a CVPR 2027 submission.

Page: https://anonymous1232112.github.io/Sisyphus/

## Overview

Every real-robot manipulation trial leaves objects out of place, and a person
has to put them back before the next trial can begin. Sisyphus is a
task-agnostic object reset framework for physical robots that removes this
manual step.

- A multimodal large language model (MLLM) decides which object to move and
  where. It describes each object by visual attributes (colour, shape,
  position) instead of its name.
- Outis-VLA, a semantics-decoupled vision-language-action policy, executes
  these attribute instructions. Pick-and-place is learned once from block
  demonstrations and transfers to unseen everyday objects.
- A training-free initial-noise search and a hierarchical closed-loop reset
  reduce and recover from failed attempts.

We also build Sisyphus-Bench, a real-robot benchmark for object reset, and
propose Tries-to-Reset (TTR), the mean number of attempts per object.

## Contents

```
index.html        project page (single static file, no build step)
assets/img/       figures and rollout frames from the paper
assets/videos/    demo videos
```

## Viewing locally

```
python3 -m http.server 8000
```

Then open http://localhost:8000 in a browser.

## Anonymity

This repository is anonymized for double-blind review. It contains no author
or affiliation information.
