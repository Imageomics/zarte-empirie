# zarte-empirie: Browser-Native BioCLIP Classification — Fresh-Start Plan

## Context

`zarte-empirie` (Goethe's *delicate empiricism*) is a new static-hosted web app at `/c/Users/thompson.4509/projects/zarte-empirie`, entirely distinct from `pybioclip`. The repository currently contains `LICENSE`, a README pointing to this plan, and this plan itself (on the `initial-plans` branch).

Tagline: *"Tools for attending to biological specimens and the traits that set them apart, interactively, through the lens of machine learning. A way of looking that extends what a researcher can perceive."*

**What this plan covers:** the v0.1 MVP — a client-compute, browser-native BioCLIP-family classification tool. The seed tool; more institute tools (Finer-CAM interpretability, TreeOfLife subset mode, patch-masking) follow in later phases. This document focuses on the MVP and its prerequisites.

**Key fact:** Nobody has put BioCLIP in the browser yet. The ONNX export pipeline is the critical path and highest-risk item — **estimated 2–4 weeks of careful engineering**. It's treated as its own blocking milestone (Phase 0) with numerical verification before any frontend code is written.

**Constraints confirmed with user:**
- Framework: Vite + React + TypeScript + Tailwind. Permanent. Any long-form docs/philosophy site is a *separate* deployment (possibly Astro), not a migration.
- Model: **Selector from day 1** supporting BioCLIP v1 (`hf-hub:imageomics/bioclip`) and BioCLIP v2 (`hf-hub:imageomics/bioclip-2`). Extensible **model registry** so additional models slot in cleanly in later phases.
- Branding: **Imageomics-branded**. ONNX artifacts hosted under `imageomics/` HF org; institute affiliations cited prominently.
- Timeline: open-ended / research pace. Build each piece to keep. Numerical-parity tests are day-one work.
- Hosting: **Cloudflare Pages primary**; GitHub Pages viable as fallback with specific limitations noted.

## Architectural Thesis: Boring at Every Layer Where AI Writes Code

This stack is optimized for a specific situation: a small research team whose contributors will not have deep frontend expertise and will rely heavily on AI coding tools (Claude Code, Codex, etc.) for implementation.

Given that, each layer of the stack is chosen to be the most mainstream, best-documented, highest-training-data option available. The goal is a stack so unsurprising that AI tools have seen thousands of versions of it and generate reliable code with minimal friction. Novelty at any layer multiplies AI error rates and erodes the "AI does the heavy lifting" advantage.

**This thesis — "boring at every layer where AI writes code" — supersedes preferences for technical elegance, framework aesthetics, or bundle-size optimization.** Any non-boring choice carries a burden of justification: the gain must clearly outweigh the AI-friendliness cost, *or* the tool must be well-supported enough that AI handles it without friction.

## Stack Decisions

| Concern | Choice | Rationale |
|---|---|---|
| Language | **TypeScript** | De facto standard; every major library ships TS types. Strong AI support (occasional type-invention mistakes, easy to correct). |
| Framework | **React** | Most mainstream UI framework; dramatically more AI training data than Svelte/Solid/Vue/Angular. Ecosystem advantage for ML library integration. |
| Build tool | **Vite** | Modern default for new React+TS projects. Fast dev server, sensible defaults, minimal config. AI handles Vite config well. |
| Styling | **Tailwind CSS** 3.4.x + `@tailwindcss/typography` | Utility-first. AI generates consistent Tailwind classes across files; plain CSS and CSS-in-JS get inconsistent AI results. Added to the stack specifically for AI-friendliness. |
| State | React `useState` / `useContext` / `useReducer` first; **Zustand** as escape valve if prop-threading gets painful | No store until genuine pain. Avoid Redux (legacy overhead) and obscure alternatives (poor AI support). |
| Routing | None for v0.1; **react-router-dom** v6 when a second tool arrives | Conventional and well-supported. |
| Fonts | `@fontsource/eb-garamond` (display) + `@fontsource-variable/inter` (UI) | Self-hosted; no external CDN. Tone per the visual-design brief. |
| Icons | `lucide-react` | Line-art, widely used with React. |
| ML runtime | `onnxruntime-web` ≥1.21 | Only mature browser ML runtime supporting arbitrary ViT architectures. Chosen for ecosystem maturity, not because ONNX is technically superior to alternatives like MLIR/ExecuTorch — those aren't browser-ready in 2026. |
| Execution provider | WebGPU primary, WASM-SIMD fallback | WebGPU avoids cross-origin isolation header requirements. WASM fallback covers older/limited browsers. |
| Tokenizer | `@xenova/transformers` **tokenizer submodule only** (CLIP BPE) — not the inference runtime | Battle-tested; verify byte-for-byte vs Python reference on day 1. See Common Pitfalls below. |
| Model hosting | HF Hub under `imageomics/bioclip-onnx-web` (or agreed alternative), per-model subfolders (`v1/`, `v2/`, future additions) | Correct CORS, CDN, version-pinned via commit SHA. Single repo simplifies permissioning. |
| Model registry | First-class `src/config.ts` registry: `{id, name, hfPath, commitSha, logitScaleExp, architecture, patchSize, gridSize, description}` per model | Selector reads from this array; adding a model is one config edit plus export artifacts. Per-model patch-specifics captured so patch-aware tools (future phases) can adapt. |
| Package manager | **bun** | Npm-command-compatible; significantly faster installs and script execution. Well-supported enough in 2026 that AI friction is low. Deliberate exception to "boring at every layer" — the speed gain is real and compatibility is high. |
| Testing | `vitest` | Vite-native, minimal config, strong AI support. **Day-one parity tests (tokenizer, preprocessing, classification) are non-negotiable**; component/UI tests added incrementally. |
| App hosting | **Cloudflare Pages** primary; GitHub Pages viable fallback | See notes below. |

### App hosting: Cloudflare Pages vs GitHub Pages

Both are strong free baselines for a client-compute static site. **Cloudflare Pages is chosen as primary** because it avoids two specific GitHub Pages limitations that will bite this project:

- **Custom HTTP headers.** GitHub Pages doesn't let you set response headers. WebAssembly multi-threading (used in the CPU-fallback path for ONNX Runtime Web) requires specific cross-origin isolation headers (COOP/COEP). If users without WebGPU fall back to CPU and you want that fallback to be fast (multi-threaded), GitHub Pages blocks you.
- **SPA routing.** When a user visits `/tools/masking` directly, GitHub Pages returns a 404 because there's no file at that path. React Router needs `index.html` served for any unknown path. Workarounds exist (a `404.html` hack, or hash-based URLs like `/#/tools/masking`), but they're ugly.

GitHub Pages remains a viable emergency option — the WebGPU common path works there, and v0.1 is a single-page app with no client-side routing yet. Cloudflare Pages is chosen up-front so we don't hit either limitation later.

## Model Weights and Caching

- Weights are fetched from Hugging Face Hub on first use, not bundled with the app.
- The app manages caching explicitly via the browser's **Cache API**, keyed on the pinned model commit SHA so that model updates bust the cache cleanly.
- IndexedDB and OPFS are alternatives; Cache API was chosen for simpler semantics and well-understood eviction behavior.
- Transformers.js is used **only** for its tokenizer submodule (CLIP tokenization from JS). The ML runtime is ONNX Runtime Web, which does **not** auto-cache weights — caching is our responsibility.

## Common Pitfalls (negative assertions)

This section captures mistakes that are cheap to avoid if flagged and expensive to debug if not. Read before making architectural decisions or writing code.

### Transformers.js vs ONNX Runtime Web

**Do not assume Transformers.js behaviors apply to our stack.** We use Transformers.js *only* for its tokenizer. The ML runtime is ONNX Runtime Web, which has different caching, different loading patterns, and different APIs. When searching for how to do something, check ONNX Runtime Web docs specifically, not general Transformers.js tutorials. A plausible-sounding Transformers.js idiom applied to our code will silently do the wrong thing (e.g., assume auto-caching when nothing is caching). AI tools in particular will reach for Transformers.js idioms because they're more common in training data — resist.

### General disambiguation principle

Any time two similar-looking technologies could be confused — Transformers.js vs ONNX Runtime Web, Next.js vs Vite, Astro vs Next.js — the plan should state which one is used for what and what is explicitly **not** used. This prevents the "plausible but wrong" failure mode in general, not just for the caching question.

Current disambiguations in this project:

- **Transformers.js (tokenizer only) vs ONNX Runtime Web (inference).** See Model Weights and Caching above.
- **Vite (build tool) vs Next.js (NOT used).** No SSR, no file-based routing, no Next-specific APIs. Vite-flavored React patterns only.
- **React Router (future, when a second tool arrives) vs Next.js App Router (NOT used).** When routing is added, it's `react-router-dom` v6.
- **Astro (possible future *separate* docs site) vs Vite+React (the main app).** The app itself is never ported to Astro. If long-form docs ever spin up, they're a separate deployment.

## Critical Risks and Mitigations

Each of these breaks the project silently if unwatched. Addressed on day 1, not after debugging a wrong result.

1. **`torch.compile` blocks export.** `pybioclip/src/bioclip/predict.py:203` wraps the model in `torch.compile`. The export script must bypass that wrapper.
2. **QuickGELU vs plain GELU.** Base `ViT-B-16.json` has no `act_layer` override → plain GELU. Confirm per-model before export; a wrong activation is a silent ~10% accuracy regression.
3. **`logit_scale` is not in either exported encoder.** Extract `model.logit_scale.exp().item()` at export time, ship in per-model `config.json`.
4. **Tokenizer parity.** Day-1 byte-for-byte verification of JS tokenizer output vs `open_clip.get_tokenizer("ViT-B-16")` reference.
5. **Preprocessing parity.** Day-1 numerical verification (max-abs-diff < 0.02) on a fixed test image.
6. **Dynamic batch axis at export.** Text encoder called with `batch = 80 × N_classes`. Export with `dynamic_axes={'input_ids': {0: 'batch'}}`.
7. **MultiheadAttention / argmax quirks.** Target opset 17. Fall back to CPU-side slicing if the text encoder's EOT-token pooling trips WebGPU op coverage.
8. **First-load time.** v1 ≈ 75 MB int8, v2 ≈ 125 MB int8. Per-model Cache API entries mean users only pay for the model they actually pick. UX must aggressively communicate first-use download per model.
9. **TreeOfLife-200M is not a browser blob.** Multi-GB even quantized. TreeOfLife deferred to Phase 4 via curated-subset approach.
10. **HF org write access.** Confirm Imageomics admins before committing to `imageomics/bioclip-onnx-web`. Fallback: personal HF account, retargetable.
11. **Model architecture differences.** BioCLIP v1 and v2 may have different vision backbones / patch-specifics (e.g. 14×14 vs 16×16 patch/grid structure). Exact architecture is verified during Phase 0. The model registry captures `patchSize` and `gridSize` per model; any patch-aware tooling (future phases) must read from the registry, never hardcode.

## Phase 0 — ONNX Export Pipeline (BLOCKING; ~2–4 weeks)

Lives in `zarte-empirie/tools/export-bioclip-onnx/` with its own Python environment. Exported artifacts land on HF Hub; only the script and `MODEL_CARD.md` stay in the repo.

**Parameterized over model**: the export script takes a model-string argument and produces a full artifact set per model. Runs cleanly for both `hf-hub:imageomics/bioclip` (v1) and `hf-hub:imageomics/bioclip-2` (v2); reusable for future models.

### Script responsibilities (per model)

1. Load via `open_clip.create_model_from_pretrained(model_str, return_transform=True)`. **Do not** wrap in `torch.compile`.
2. **Inspect the architecture config** — read `patch_size`, `image_size`, `embed_dim`, and GELU variant. Record in `config.json`. Fail loudly if QuickGELU is detected.
3. Wrap vision and text encoders in thin `nn.Module` shells exposing `encode_image` and `encode_text`.
4. Extract `model.logit_scale.exp().item()` → per-model `config.json` (also record normalization mean/std and image size for reference).
5. Export each encoder with `torch.onnx.export(..., opset_version=17, dynamic_axes={...})`.
6. Run `onnxruntime.transformers.optimizer.optimize_model(model_type="clip")` on both encoders.
7. Produce int8 dynamic-quantized variants via `quantize_dynamic`. Keep FP32 originals as quality reference.
8. **Numerical-verification harness** (blocking gate per model): load ONNX, run fixed image + fixed prompt through both ONNX and PyTorch. Top-1 must match; FP32 probabilities within 1e-4, int8 within 1e-2.
9. Upload per model: `vision.onnx`, `vision-int8.onnx`, `text.onnx`, `text-int8.onnx`, `config.json` into its own subfolder (`v1/`, `v2/`) of the HF repo. Single top-level `MODEL_CARD.md` credits the BioCLIP papers, preserves MIT license, documents export details and per-model verification.

### Deliverables

- `imageomics/bioclip-onnx-web` HF repo populated with `v1/` and `v2/` subfolders.
- Export script at `zarte-empirie/tools/export-bioclip-onnx/export.py` with README.md.
- Verification report per model.

**Do not begin Phase 1 until Phase 0 verification passes for at least one of the two models.** If only one model has verified by the time Phase 1 begins, the UI selector hides the unverified option behind a feature flag.

## Phase 1 — v0.1 MVP: CustomLabels Classification in the Browser

### In scope

- Single-page app: **select model (BioCLIP v1 / v2)**, drag-drop image, enter 2–20 newline-separated class labels, click Classify, see top-k results with calibrated scores.
- Model selector with per-model size/speed copy.
- Model loading: streaming fetch from HF Hub with progress bar (`Content-Length` + cumulative reader bytes).
- Per-model Cache API entries (keyed on version-pinned HF URL with commit SHA) survive independently; switching back to a previously-used model is instant.
- WebGPU EP primary, WASM-SIMD fallback, user-visible EP indicator.
- Tokenize classes × 80 ImageNet templates → text-encode → L2-normalize → mean per class → re-L2-normalize → `[embed_dim, N]`.
- Image preprocess → image-encode → L2-normalize → `[embed_dim]`.
- Active model's `logitScaleExp * img @ txt` → softmax → top-k.
- Memoize text features keyed on `(modelId, sortedClassList)`.
- Natural-history visual design (see brief).

### Out of scope (explicitly deferred)

- Finer-CAM interpretability (Phase 3)
- TreeOfLife mode (Phase 4)
- Patch-masking tool (Phase 5)
- Side-by-side comparison UI (Phase 2)
- Custom model-string input / "advanced mode" (Phase 2)
- Multi-image batch
- Mobile-optimized UI (works on iPad; doesn't promise mobile-first)
- Analytics / telemetry
- User accounts or history

### Repository structure

```
zarte-empirie/
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── components/
│   │   ├── ImageDropzone.tsx
│   │   ├── LabelEditor.tsx
│   │   ├── ModelSelector.tsx         # Reads src/config.ts registry
│   │   ├── ModelStatus.tsx           # Download progress + EP indicator
│   │   ├── Results.tsx
│   │   └── Layout.tsx
│   ├── lib/
│   │   ├── bioclip/
│   │   │   ├── classifier.ts
│   │   │   └── templates.ts          # 80 ImageNet templates
│   │   ├── model/
│   │   │   └── loader.ts             # HF Hub fetch + Cache API + ORT session
│   │   ├── tokenizer/
│   │   │   └── clip.ts               # CLIP BPE wrapper over @xenova/transformers
│   │   └── preprocess/
│   │       └── image.ts              # Blob → Float32 ort.Tensor
│   ├── styles/
│   │   └── globals.css
│   └── config.ts                     # Model registry + default model id
├── public/                           # Static assets only (NO model weights)
├── tools/
│   └── export-bioclip-onnx/
│       ├── export.py
│       ├── verify.py
│       ├── pyproject.toml
│       └── README.md
├── tests/
│   └── parity/
│       ├── tokenizer.test.ts
│       ├── preprocess.test.ts
│       └── classification.test.ts
├── index.html
├── package.json
├── bun.lockb
├── vite.config.ts
├── tailwind.config.ts
├── tsconfig.json
├── README.md
└── LICENSE
```

### Ordered task sequence

1. Scaffold: `bun create vite . --template react-ts`. Install Tailwind, `@fontsource/eb-garamond`, `@fontsource-variable/inter`, `lucide-react`, `vitest`, `@testing-library/react`, `@xenova/transformers`, `onnxruntime-web`.
2. Tailwind config: warm parchment background (`#f5efe3`), ink body (`#2b2622`), muted earth-tone accents. Typography plugin configured. Layout helpers for centered column (~980px max).
3. Author `src/config.ts` model registry with entries for v1 and v2 (HF URLs with pinned commit SHAs, extracted `logitScaleExp`, per-model `patchSize` and `gridSize` from Phase 0 `config.json`, human-readable `name` and `description`, download-size estimate). Expose a default model id.
4. Implement `src/lib/tokenizer/clip.ts` + `tests/parity/tokenizer.test.ts` against a Python-generated reference JSON (produced once by a throwaway script that runs `open_clip.get_tokenizer("ViT-B-16")(fixtures)` and saves to `tests/parity/fixtures/tokens.json`). Test passes = tokenizer trustable. Same tokenizer across v1 and v2.
5. Implement `src/lib/preprocess/image.ts` + `tests/parity/preprocess.test.ts` against a Python-generated reference tensor for a fixed test image. Max-abs-diff threshold 0.02. Same preprocessing across v1 and v2.
6. Implement `src/lib/model/loader.ts` — takes a registry entry, fetches vision+text ONNX with streaming progress, Cache API read-through (keyed on full URL including commit SHA), creates `InferenceSession` with `executionProviders: ['webgpu', 'wasm']`, warmup dummy inference on first load per model. Returns a `LoadedModel` holding both sessions + registry metadata.
7. Implement `src/lib/bioclip/templates.ts` — port the 80 OpenAI ImageNet templates verbatim from `pybioclip/src/bioclip/predict.py:32–113`.
8. Implement `src/lib/bioclip/classifier.ts` orchestrating the flow per active `LoadedModel`. Memoize text features keyed on `(modelId, sortedClassList)`.
9. Build the UI: four panels (model selector + dropzone + label editor + results), state machine `idle → downloading-model → model-ready → classifying → results`, separate state per active model load. Results as ranked list with horizontal probability bars (reads as a natural-history table, not a pie). Model switching preserves image and label state; only predictions recompute.
10. Classification parity tests: one per registered model. Fixed image + class list, top-1 must match pybioclip reference within defined thresholds.
11. Deploy to Cloudflare Pages.

### Visual-design brief

- Tone: **natural-history journal**. Phenomenological, specimen-dominant, chrome-recedes. General natural-world vibe — warm and organic, not tied to any specific kingdom or domain.
- Palette: warm parchment (`#f5efe3`), deep ink (`#2b2622`), muted earth accents (a sage or walnut, whichever reads most neutral in context).
- Typography: EB Garamond for tagline, page heading, display text; Inter for form controls and small UI labels. Italic Garamond for epigraphs.
- Layout: one centered column, ~980px max. Generous vertical rhythm (≈1.6 line-height body, large section padding). No shadowed cards. Thin rules between sections.
- Imagery: the uploaded specimen occupies center stage at reasonable resolution, not cropped to a thumbnail.
- Icons: `lucide-react`, 1.5px stroke, matched to Garamond's fine line weight.
- Dark mode: **not** in v0.1. Light mode only.

## Reference: pybioclip

Implementation details not explicitly captured in this plan should be derived from pybioclip as the source of truth. Key files:

- `C:/Users/thompson.4509/projects/pybioclip/src/bioclip/predict.py`
  - Lines 32–113: 80 OpenAI ImageNet templates (verbatim port to TS)
  - Lines 157–166: image preprocessing (reproduce numerically)
  - Line 242: `logit_scale` usage
  - Lines 334–404: `CustomLabelsClassifier` orchestration
  - Lines 481+: `TreeOfLifeClassifier` (reference for Phase 4)
- `C:/Users/thompson.4509/projects/pybioclip/src/bioclip/_constants.py`
  - Lines 8–10: `BIOCLIP_V1_MODEL_STR` / `BIOCLIP_V2_MODEL_STR`
- `C:/Users/thompson.4509/projects/pybioclip/.venv312/Lib/site-packages/open_clip/tokenizer.py:133–170`
  - Canonical `SimpleTokenizer` — JS output must match byte-for-byte.
- `C:/Users/thompson.4509/projects/pybioclip/.venv312/Lib/site-packages/open_clip/model_configs/ViT-B-16.json`
  - Architecture config reference (embed_dim=512, context_length=77, vocab_size=49408, plain GELU).
- `C:/Users/thompson.4509/projects/pybioclip/src/bioclip/patch_gui.py:17–162` (Phase 5 reference)
  - Mask/crop logic for the later patch-masking tool port.

When the plan is silent on a specific detail, **pybioclip is the starting point**. Anything that diverges from pybioclip should be captured here (or in a successor plan) as a deliberate decision.

## Later Phases (summary)

- **Phase 2 (v0.2): Polish + PWA.** Graceful error states (WebGPU unsupported, download failed, invalid image, OOM). Side-by-side model comparison on the same specimen. Custom model-string input (advanced mode — iterating toward pybioclip's full model surface). iPad-friendly responsive polish. Share-a-prediction via URL-encoded hash (no image upload). **PWA:** service worker + manifest for offline use and "install as app" experience. Additive to the same Vite+React app; no architectural change.

- **Phase 3 (v0.3): Finer-CAM interpretability.** Gradient-based class-activation mapping. ONNX Runtime Web does not provide native autograd, so the path is to **export the gradient computation as part of the ONNX graph** using `torch.func` symbolic differentiation. This is meaningful additional engineering on top of the base Phase 0 conversion. **Architectural option worth considering:** structure this as a reusable **interpretability-to-ONNX export toolkit** — a library contribution to the community — rather than one-off project code. Decision deferred until Phase 3 kickoff.

- **Phase 4 (v0.4): TreeOfLife subset mode.** Curated subset artifacts (e.g., domain-specific species bundles — "North American fishes," "European butterflies") precomputed and hosted on HF Hub. Ports `create_taxa_filter_from_csv` logic from pybioclip.

- **Phase 5 (v0.5): Patch-masking tool.** At `/tools/masking` (or similar). Ports `pybioclip/src/bioclip/patch_gui.py:17–162` to JS/Canvas. **Per-model patch-specifics:** grid and patch size read from the model registry, not hardcoded. Reuses the active model's image-encoder session — no additional model download.

- **Future, if warranted:** separate Astro deployment for long-form docs (philosophy, methodology, Goethean-method essays, contributor guides). Lives at e.g. `docs.zarte-empirie.org`. Entirely independent from the SPA — not a migration of it.

## What This Plan Explicitly Rejects

Capturing rejections explicitly prevents future churn:

- **Over-engineering:** no Next.js (SSR not needed), no Nx/Turborepo monorepo tooling (premature), no Redux (legacy overhead), no GraphQL, no CSS-in-JS (inconsistent AI support).
- **Novelty for its own sake:** no Svelte, Solid, or Qwik despite their technical merits. Training-data availability matters more than elegance given the AI-implementation thesis.
- **Dual deployment:** no Python backend for ML inference. Platform is committed to browser-native inference. Tools that cannot run in the browser are out of scope for now.
- **Day-one tests skipped:** parity tests (tokenizer, preprocessing, classification) are day-one mandatory. Only component/UI tests are deferrable.

## Open Technical Questions (not yet decided)

- **BioCLIP ONNX conversion timing.** Estimated 2–4 weeks of careful engineering. Treat as prerequisite, not background task.
- **Finer-CAM packaging.** Whether to structure the autograd-in-ONNX work as a reusable community library (interpretability-to-ONNX export toolkit) or one-off project code. Decision at Phase 3 kickoff.
- **BioCLIP v1 vs v2 vision backbones.** Exact architecture (patch size, grid size) confirmed during Phase 0. Different backbones → real per-model `patchSize`/`gridSize` in the registry. Identical backbones → simpler patch-aware tooling later.

## Verification Strategy

End-to-end manual test on v0.1:

1. `bun dev`.
2. Open in Chrome with WebGPU enabled.
3. Select BioCLIP v2 in the selector; wait for download and ready state.
4. Drag-drop a known test image (e.g., pybioclip's `tests/images/mycat.jpg` or equivalent).
5. Enter three candidate classes (e.g., "Felis catus", "Canis lupus familiaris", "Panthera leo").
6. Run `bioclip predict --cls "Felis catus,Canis lupus familiaris,Panthera leo" --model-str hf-hub:imageomics/bioclip-2 tests/images/mycat.jpg` against pybioclip.
7. Compare: top-1 must match; probabilities within 1e-2.
8. Switch selector to BioCLIP v1 — expect download first time, instant on subsequent switches.
9. Run equivalent pybioclip command with `--model-str hf-hub:imageomics/bioclip`; verify parity for v1.
10. Switch back to v2 — comes from Cache API, zero network activity.
11. Reload page — both cached models remain cached; selector defaults to the configured default.
12. DevTools Performance: second classification with same class list should be image-encoder-bound (< 100ms on WebGPU on a modern laptop).

Automated: parity tests in `tests/parity/` run on every commit via `bun test` (vitest under the hood). Classification parity parameterized over all registered models.

## Open Dependencies (non-blocking but worth confirming early)

- Write access to `imageomics/` HF org for `bioclip-onnx-web`. Fallback: personal HF account, retargetable.
- Imageomics institute style/brand constraints to honor. If none, proceed with the natural-history brief above.
- Both BioCLIP v1 and v2 training configs (GELU variant, exact architecture, patch/grid specifics) — verify during Phase 0 before export. Fail loudly in the export script on QuickGELU detection.
- Cloudflare Pages account and target production domain.
