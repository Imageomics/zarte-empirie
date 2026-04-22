# zarte-empirie: Browser-Native BioCLIP Classification — Fresh-Start Plan

## Context

`zarte-empirie` (Goethe's *delicate empiricism*) will be a new static-hosted web app at `/c/Users/thompson.4509/projects/zarte-empirie`, entirely distinct from `pybioclip`. The repository today contains only `LICENSE` + `.git`.

Tagline: *"Tools for attending to biological specimens and the traits that set them apart, interactively, through the lens of machine learning. A way of looking that extends what a researcher can perceive."*

**What this plan covers:** the v0.1 MVP — a client-compute, browser-native BioCLIP-family classification tool. The seed tool; more institute tools (patch-masking interpretability, etc.) follow in later phases but this plan focuses only on MVP and its prerequisites.

**Key fact:** Nobody has put BioCLIP in the browser yet. The ONNX export pipeline is the critical path and highest-risk item. It's treated as its own blocking milestone (Phase 0) with numerical verification before any frontend code is written.

**Constraints confirmed with user:**
- Framework: Vite + React + TypeScript + Tailwind + pnpm. Permanent. If a long-form docs/philosophy site is later desired, it ships as a *separate* Astro deployment — not a migration.
- Model: **Selector from day 1** supporting BioCLIP v1 (`hf-hub:imageomics/bioclip`) and BioCLIP v2 (`hf-hub:imageomics/bioclip-2`). Designed as an extensible **model registry** so additional models (other BioCLIP variants, custom open_clip model strings, eventually the full pybioclip model surface) slot in cleanly in later phases.
- Branding: **Imageomics-branded**. ONNX artifacts hosted under `imageomics/` HF org; institute affiliations cited prominently; design respects any institute style/brand constraints.
- Timeline: open-ended / research pace. Build each piece to keep — no rough-draft scope cuts. Numerical-parity tests are day-one work.

## Architecture Decisions

| Concern | Decision | Rationale |
|---|---|---|
| Project type | Vite+React SPA served as static files | Client-compute MVP is a deeply interactive single page; Astro's static-first value proposition doesn't apply here. |
| ML runtime | `onnxruntime-web` ≥1.21 | Transformers.js doesn't support open_clip (BioCLIP's base); ORT Web is the only realistic path. |
| Execution provider | WebGPU primary, WASM-SIMD fallback | WebGPU avoids COEP/COOP complications. WASM fallback covers older browsers. |
| Model hosting | HF Hub under `imageomics/bioclip-onnx-web` (or similar), one repo with per-model subfolders (`v1/`, `v2/`, future additions) | Correct CORS headers, CDN, version-pinned via commit SHA. Single repo simplifies permissioning and versioning. Do NOT commit ONNX weights to git. |
| Model registry | First-class `src/config.ts` registry: array of `{id, name, hfPath, commitSha, logitScaleExp, description}` entries | Selector reads from this array; adding a new model is one config edit plus the export artifacts. |
| Model cache | `Cache API` keyed on version-pinned HF URL | Survives reloads; browser handles eviction. IndexedDB is overkill. |
| Tokenizer | `@xenova/transformers` tokenizer submodule (CLIP BPE) | Battle-tested; verify byte-for-byte vs Python reference on day 1. |
| State management | React `useState`/`useReducer` only | No store until genuine prop-drilling pain. |
| Routing | None for v0.1 | Add `react-router-dom` when second tool arrives. |
| Styling | Tailwind 3.4.x + `@tailwindcss/typography` | Proven; v4 still causing pain. |
| Fonts | `@fontsource/eb-garamond` (display) + `@fontsource-variable/inter` (UI) | Goethean/botanical tone; self-hosted so no external CDN. |
| Icons | `lucide-react` | Line-art, matches botanical aesthetic. |
| Testing | `vitest` + `@testing-library/react` | Numerical-parity tests against pybioclip live here. |

## Critical Risks and Mitigations

These are the things that break the project silently if not watched for; each is fixed on day 1, not after debugging a wrong result.

1. **`torch.compile` blocks export.** `pybioclip/src/bioclip/predict.py:203` wraps the model in `torch.compile`. The export script must bypass that wrapper — load the raw model, do not call `torch.compile`.
2. **QuickGELU vs plain GELU.** Verified: base `ViT-B-16.json` (at `.venv312/Lib/site-packages/open_clip/model_configs/ViT-B-16.json`) has no `act_layer` override → plain GELU. Both BioCLIP v1 and v2 should train with this config; confirm per-model before export by inspecting each checkpoint's preprocess config or architecture name. A wrong activation is a silent ~10% accuracy regression.
3. **`logit_scale` is not in either exported encoder.** It's a top-level `nn.Parameter` on the CLIP model. Extract `model.logit_scale.exp().item()` at export time, save to a `config.json` alongside ONNX files. JS multiplies on the CPU (scalar, free).
4. **Tokenizer parity.** CLIP's BPE + the `ftfy` `basic_clean` step differ subtly across implementations. On day 1, tokenize a fixed sentence in both Python (`open_clip.get_tokenizer("ViT-B-16")`) and JS (`@xenova/transformers` CLIP tokenizer). Assert byte-for-byte identical output for at least the 80 ImageNet templates. If `ftfy` cases diverge, add a thin JS cleanup shim.
5. **Preprocessing parity.** `predict.py:157–166` uses `ToTensor → Resize((224,224), antialias=True) → Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))`. In JS: `createImageBitmap(blob, { resizeWidth: 224, resizeHeight: 224, resizeQuality: "high" })` → OffscreenCanvas → getImageData → normalize → transpose HWC→CHW. Verify numerical parity on a fixed test image: dump the preprocessed tensor from both pipelines, assert max-abs-diff < 0.02.
6. **Dynamic batch axis at export.** The text encoder is called with `batch = 80 × N_classes` (80 prompt templates per class). Export with `dynamic_axes={'input_ids': {0: 'batch'}}` or expect a hard size ceiling.
7. **MultiheadAttention / argmax quirks.** Export target opset 17. If the text encoder's `text.argmax(dim=-1)` for EOT-token pooling trips ORT's WebGPU op coverage, fall back to CPU-side slicing post-inference.
8. **First-load time.** BioCLIP v1 is ~300MB unquantized / ~75MB int8; v2 is ~500MB unquantized / ~125MB int8. At 30 Mbps home connection, v2 quantized ≈ 35s first load. With a selector, users pay the download cost only for the model they actually choose, and subsequent loads of the same model come from Cache API instantly. UX must *aggressively* communicate: "First use of [model name] downloads ~N MB. Subsequent uses of this model are instant." Speculative prewarm on idle for the default-selected model unless `navigator.connection.saveData` is set. Never prewarm both models.
9. **TreeOfLife-200M is not browser-feasible as a blob.** `imageomics/TreeOfLife-200M` holds ~200M species embeddings — multi-GB even quantized. TreeOfLife support is explicitly deferred to Phase 3 via a curated-subset approach (ship precomputed subset embeddings per domain, e.g., "North American birds").
10. **HF org write access.** Hosting under `imageomics/bioclip-onnx-web` assumes write access to the org. Confirm with Imageomics admins before committing to the repo path — fallback is a personal HF account, retargetable later.

## Phase 0 — ONNX Export Pipeline (BLOCKING; out of band)

Lives in `zarte-empirie/tools/export-bioclip-onnx/` with its own Python environment (not mixed with the frontend). The exported artifacts land on HF Hub; only the script and MODEL_CARD.md files stay in the repo.

**Parameterized over model**: the export script takes a model-string argument and produces a full artifact set per model. It must run cleanly for both `hf-hub:imageomics/bioclip` (v1) and `hf-hub:imageomics/bioclip-2` (v2); the same script is reused for future models added to the registry.

### Script responsibilities (per model)

1. Load via `open_clip.create_model_from_pretrained(model_str, return_transform=True)`. **Do not** wrap in `torch.compile`.
2. Confirm the architecture's GELU variant by inspecting the model's config (fail loudly if QuickGELU is detected, since it would require an alternate export path).
3. Wrap vision and text encoders in thin `nn.Module` shells exposing `forward(x) = model.encode_image(x)` and `forward(t) = model.encode_text(t)` respectively.
4. Extract `model.logit_scale.exp().item()` → per-model `config.json` (also include image-size, normalization mean/std for reference).
5. Export each encoder with `torch.onnx.export(..., opset_version=17, dynamic_axes={...})`.
6. Run `onnxruntime.transformers.optimizer.optimize_model(model_type="clip")` on both encoders.
7. Produce int8 dynamic-quantized variants via `onnxruntime.quantization.quantize_dynamic`. Keep FP32 originals as the quality reference.
8. **Numerical-verification harness:** load ONNX via `onnxruntime.InferenceSession`, run the same fixed test image and fixed prompt through both ONNX and PyTorch. Top-1 class must match; FP32 probabilities within 1e-4, int8 within 1e-2. **Blocking gate per model** — if v2 passes but v1 fails, publish v2 and hold v1 back until it verifies.
9. Upload per model: `vision.onnx`, `vision-int8.onnx`, `text.onnx`, `text-int8.onnx`, `config.json` into its own subfolder (`v1/`, `v2/`) of the HF repo. A single top-level MODEL_CARD.md covers all models, credits the BioCLIP papers, preserves MIT license, documents the export (opset, quantization method, script URL, per-model verification results).

### Deliverables

- `imageomics/bioclip-onnx-web` (or agreed alternative) HF repo populated with `v1/` and `v2/` subfolders.
- Export script checked into `zarte-empirie/tools/export-bioclip-onnx/export.py` with a README.md documenting how to re-run for a new model.
- Verification report per model showing top-1 agreement and probability deltas.

**Do not begin Phase 1 until Phase 0 deliverables exist and verification passes for at least one of the two models.** If only one model has verified by the time Phase 1 begins, the UI selector can hide the unverified option behind a feature flag.

## Phase 1 — v0.1 MVP: CustomLabels Classification in the Browser

### In scope

- Single-page app: **select model (BioCLIP v1 / v2)**, drag-drop image, enter 2–20 newline-separated class labels, click Classify, see top-k results with calibrated scores.
- Model selector with size/speed tradeoff copy ("v1 — ~75 MB, faster first-load; v2 — ~125 MB, institute default"). Switching models triggers download-and-swap of the active `InferenceSession`; previous session is disposed.
- Model loading: fetch ONNX weights from HF Hub with progress bar (reads `Content-Length` and cumulative bytes from `Response.body.getReader()`).
- Cache API caching keyed on version-pinned HF URL — **entries per model** survive independently; switching back to a previously-used model is instant.
- WebGPU EP primary, WASM-SIMD fallback, user-visible indicator of which is active.
- Tokenize classes × 80 ImageNet templates → text-encode → L2-normalize → mean per class → re-L2-normalize → `[512, N]`.
- Image preprocess → image-encode → L2-normalize → `[512]`.
- Multiply by the active model's `logit_scale_exp`, compute `img @ txt`, softmax, top-k display.
- Memoize text features keyed on **(modelId, sortedClassList)** — switching models invalidates the text-feature cache but not the image (since the encoder is model-specific, all features must be recomputed per model; the cache key reflects that).
- Botanical/Goethean visual design.

### Out of scope (explicitly deferred)

- TreeOfLife mode
- Patch-masking tool
- Side-by-side comparison UI (see both models' predictions on the same image simultaneously — that's Phase 2 polish)
- Custom model-string input (advanced mode for arbitrary open_clip models) — Phase 2+
- Multi-image batch
- Mobile-optimized UI (works on iPad; doesn't promise mobile-first)
- Analytics / telemetry
- User accounts or history

### Repository structure

```
zarte-empirie/
├── src/
│   ├── main.tsx                      # React entry
│   ├── App.tsx                       # Single-page root
│   ├── components/
│   │   ├── ImageDropzone.tsx
│   │   ├── LabelEditor.tsx
│   │   ├── ModelSelector.tsx         # Reads src/config.ts registry; switches active model
│   │   ├── ModelStatus.tsx           # Download progress + EP indicator for the active model
│   │   ├── Results.tsx
│   │   └── Layout.tsx
│   ├── lib/
│   │   ├── bioclip/
│   │   │   ├── classifier.ts         # Orchestrates image+text encoding, scoring
│   │   │   └── templates.ts          # 80 ImageNet templates (port of predict.py:32-113)
│   │   ├── model/
│   │   │   └── loader.ts             # HF-Hub fetch + Cache API + InferenceSession
│   │   ├── tokenizer/
│   │   │   └── clip.ts               # CLIP BPE wrapper over @xenova/transformers
│   │   └── preprocess/
│   │       └── image.ts              # Blob → [1,3,224,224] Float32 ort.Tensor
│   ├── styles/
│   │   └── globals.css               # Tailwind + font imports + CSS vars for palette
│   └── config.ts                     # Model registry: array of {id, name, hfPath, commitSha, logitScaleExp, sizeMb, description} entries; default model id; extensible for future additions
├── public/                           # Static assets only (NO model weights)
├── tools/
│   └── export-bioclip-onnx/
│       ├── export.py
│       ├── verify.py
│       ├── pyproject.toml
│       └── README.md
├── tests/
│   └── parity/
│       ├── tokenizer.test.ts         # Verify byte-for-byte vs Python reference
│       ├── preprocess.test.ts        # Verify tensor within 0.02 abs-diff
│       └── classification.test.ts    # Verify top-1 agreement on fixed image
├── index.html
├── package.json
├── pnpm-lock.yaml
├── vite.config.ts
├── tailwind.config.ts
├── tsconfig.json
├── README.md
└── LICENSE                           # already exists
```

### Ordered task sequence

1. Scaffold: `pnpm create vite@latest . -- --template react-ts` (in the existing directory alongside `LICENSE`). Install Tailwind, `@fontsource/eb-garamond`, `@fontsource-variable/inter`, `lucide-react`, `vitest`, `@testing-library/react`, `@xenova/transformers`, `onnxruntime-web`.
2. Tailwind config: parchment cream background (`#f5efe3`), ink body (`#2b2622`), walnut/sage accents. Typography plugin configured. Layout helpers for the "single central column, max-width ~980px" shape.
3. Author `src/config.ts` model registry with entries for v1 and v2 (full HF URLs with pinned commit SHAs, extracted `logitScaleExp` from Phase 0, human-readable `name` and `description`, download-size estimate, architecture metadata). Expose a default model id.
4. Implement `lib/tokenizer/clip.ts` and write `tests/parity/tokenizer.test.ts` comparing against a Python-generated reference JSON (produced once by a throwaway script that runs `open_clip.get_tokenizer("ViT-B-16")(fixtures)` and saves to `tests/parity/fixtures/tokens.json`). Test passes = tokenizer is trustable. Tokenizer is shared across all registry models (both BioCLIP v1 and v2 use the same ViT-B/16 tokenizer).
5. Implement `lib/preprocess/image.ts` and write `tests/parity/preprocess.test.ts` against a Python-generated reference tensor for a fixed test image. Max-abs-diff threshold: 0.02. Preprocessing is shared across models (same 224×224 normalize).
6. Implement `lib/model/loader.ts` — takes a model registry entry, fetches vision+text ONNX from the HF URL with streaming progress callback, Cache API read-through (keyed by full URL including commit SHA), creates `InferenceSession` with `executionProviders: ['webgpu', 'wasm']`. Returns a `LoadedModel` object holding both sessions + the entry's metadata. Warmup dummy inference on first load per model.
7. Implement `lib/bioclip/templates.ts` — port the 80 OpenAI ImageNet templates from `predict.py:32–113` verbatim as JS template functions.
8. Implement `lib/bioclip/classifier.ts` orchestrating the flow per **active** `LoadedModel`: preprocess image → encode image → L2 normalize; tokenize N×80 prompts in a single batch → encode text → L2 normalize → reshape to `[N, 80, 512]` → mean over axis 1 → L2 normalize → transpose to `[512, N]`; `logitScaleExp * img @ txt` → softmax → top-k. Memoize text features keyed on `(modelId, sortedClassList)`.
9. Build the UI: four panels (model selector + image dropzone + label editor + results), state machine `idle → downloading-model → model-ready → classifying → results`, with separate state per active model load. Results component: ranked list with horizontal bar indicators (not pie — reads as a natural-history table). Model switching preserves the image and label state; only predictions are recomputed.
10. Classification parity tests: one per model in the registry. Fixed image + fixed class list, assert top-1 matches pybioclip reference output within defined thresholds for both v1 and v2.
11. Deploy target: Cloudflare Pages (excellent static perf, easy custom domain, generous free tier, simple `_headers` config if COEP/COOP ever needed). Repo push triggers build; model fetch from HF Hub happens on first user visit per selected model.

### Visual-design brief

- Tone: botanical illustration / natural-history journal. Restrained. Specimen dominant, UI recedes.
- Palette: warm parchment (`#f5efe3`), deep ink (`#2b2622`), muted sage (`#7f8a6a`) or walnut (`#6b4e35`) for accents.
- Typography: EB Garamond for the tagline, page heading, and label headings; Inter for form controls and small labels. Italic Garamond for epigraphs.
- Layout: one centered column, ~980px max. Generous vertical rhythm (1.6 line-height body, large section padding). No cards with shadows. Thin rules between sections.
- Imagery: the uploaded specimen gets center stage at native resolution (within layout bounds), not cropped to a thumbnail.
- Icons: lucide-react, 1.5px stroke, matched to Garamond's fine line weight.
- Dark mode: **not** in v0.1. Light mode only.

## Later Phases (summary only — detailed plans separately when they come up)

- **Phase 2 (v0.2):** Side-by-side comparison UI (both selected models' predictions on the same image at once); custom model-string input (advanced mode for arbitrary open_clip models, iterating toward pybioclip's full model surface); graceful error states for WebGPU-unsupported, download-failed, invalid-image, OOM; iPad-friendly responsive polish; share-a-prediction via URL-encoded hash (no image upload).
- **Phase 3 (v0.3):** TreeOfLife via curated subset artifacts (e.g., "North American fishes", "European butterflies") published to HF Hub as small precomputed species-embedding bundles per domain. Requires port of `create_taxa_filter_from_csv` logic.
- **Phase 4 (v0.4):** Patch-masking tool at `/patch` (or separate route). Ports `pybioclip/src/bioclip/patch_gui.py:17–162` (grid/freeform masking, mean-fill/zero-fill/gaussian-blur strategies, apply_mask_to_image bounding-box crop logic) to JS/Canvas. Reuses the same model-singleton and image-encoder session — no additional model download.
- **Future, if warranted:** separate Astro deployment for long-form docs, methodology, Goethean-method essays, contributor guides. Lives at e.g. `docs.zarte-empirie.org`. Entirely independent from the SPA.

## Verification Strategy

End-to-end test, manual on v0.1:

1. Serve the app locally (`pnpm dev`).
2. Open in Chrome with WebGPU enabled.
3. Select BioCLIP v2 in the model selector, wait for model download and ready state.
4. Drag-drop a known test image (e.g., pybioclip's `tests/images/mycat.jpg` or an equivalent specimen image).
5. Enter three candidate classes (e.g., "Felis catus", "Canis lupus familiaris", "Panthera leo").
6. Run `bioclip predict --cls "Felis catus,Canis lupus familiaris,Panthera leo" --model-str hf-hub:imageomics/bioclip-2 tests/images/mycat.jpg` against pybioclip.
7. Compare: top-1 class must match; probabilities within 1e-2 (int8 quantization tolerance).
8. Switch selector to BioCLIP v1 — expect a second download the first time, instant on subsequent switches.
9. Run the equivalent pybioclip command with `--model-str hf-hub:imageomics/bioclip` and verify top-1 agreement and probability deltas for v1.
10. Switch back to v2 — should come from Cache API, zero network activity.
11. Reload page — both cached models remain cached, selector defaults to the configured default model.
12. Check in DevTools Performance: second classification with same class list should be image-encoder-bound (< 100ms on WebGPU on a modern laptop).

Automated: parity tests in `tests/parity/` run on every commit via `pnpm test`. Classification parity runs per registered model.

## Critical Files

### Reference files in pybioclip (to port from)

- `C:/Users/thompson.4509/projects/pybioclip/src/bioclip/predict.py`
  - Lines 32–113: OpenAI ImageNet 80 prompt templates (verbatim port to TS)
  - Lines 157–166: image preprocessing (reproduce numerically in JS)
  - Line 242: `logit_scale` usage pattern
  - Lines 334–404: `CustomLabelsClassifier` orchestration (reference for `classifier.ts` design)
- `C:/Users/thompson.4509/projects/pybioclip/src/bioclip/_constants.py`
  - Lines 8–10: `BIOCLIP_V2_MODEL_STR = "hf-hub:imageomics/bioclip-2"`
- `C:/Users/thompson.4509/projects/pybioclip/.venv312/Lib/site-packages/open_clip/tokenizer.py`
  - Lines 133–170: canonical `SimpleTokenizer` behavior — JS output must match this
- `C:/Users/thompson.4509/projects/pybioclip/.venv312/Lib/site-packages/open_clip/model_configs/ViT-B-16.json`
  - Architecture config — confirms embed_dim=512, context_length=77, vocab_size=49408, plain GELU
- `C:/Users/thompson.4509/projects/pybioclip/src/bioclip/patch_gui.py` (Phase 4 reference, not MVP)
  - Lines 17–162: mask/crop logic for later patch-masking tool port

### New files to create in zarte-empirie

- `tools/export-bioclip-onnx/export.py` (Phase 0 critical path; parameterized over model string)
- `tools/export-bioclip-onnx/verify.py` (Phase 0 blocking gate; runs per model)
- `src/lib/bioclip/classifier.ts` (orchestration hub, takes an active `LoadedModel`)
- `src/lib/tokenizer/clip.ts` (CLIP BPE wrapper; day-1 parity test)
- `src/lib/preprocess/image.ts` (image → tensor; day-1 parity test)
- `src/lib/model/loader.ts` (HF fetch + Cache API + ORT session; per-registry-entry)
- `src/lib/bioclip/templates.ts` (80 ImageNet templates)
- `src/config.ts` (model registry array + default model id)
- `src/components/ModelSelector.tsx` (reads registry, emits selection change)
- `tests/parity/tokenizer.test.ts`, `preprocess.test.ts`, `classification.test.ts` (classification parity parameterized over registry entries)

## Open Dependencies (non-blocking but worth confirming early)

- Write access to `imageomics/` HF org for publishing `bioclip-onnx-web` artifacts. If unavailable for MVP, scaffold under a personal HF account, retarget in Phase 2.
- Any Imageomics institute style guide / logo / color palette that the visual-design brief should honor. If none exists, proceed with the Goethean botanical palette described above.
- Both BioCLIP v1 and v2 training configs (GELU variant confirmation per model) — inspect each HF model card or architecture JSON during Phase 0 before export. Fail loudly in the export script if QuickGELU is detected on either.
- Cloudflare Pages account (or alternative static host). Free-tier works for MVP; decide the production domain before Phase 1 wrap.
