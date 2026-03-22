# TODO

## Multimodal Roadmap

- [x] Light level: image OCR + audio ASR in `basic_v1` extractor.
- [ ] Video: add keyframe extraction.
- [ ] Video: add OCR over keyframes.
- [ ] Video: add audio-track ASR.
- [ ] Video: add provenance fields (timestamps/frame ranges) for evidence.
- [ ] Medium: better OCR quality controls (language packs, confidence thresholds).
- [ ] Medium: better ASR controls (model/device/per-language tuning).
- [ ] Medium: add extraction quality flags to metadata and surface in API/UI.
- [ ] Heavy: add native multimodal embeddings (image/audio/video) instead of text-only fallback.
- [ ] Heavy: add cross-modal retrieval/ranking policy and score fusion.
- [ ] Heavy: add performance profile + batching + async workers for heavy media.

## Quality and Validation Backlog

- [ ] Add UI e2e coverage for critical flows (graph/node CRUD, explore, edge actions).
- [ ] Add performance/load tests for ingestion, FTS, and vector query latencies.
- [ ] Add CI smoke jobs against live external services (OpenAI API, PP-OCRv5/Whisper runtime checks).

## Graph Correctness Backlog

- [ ] Implement evidence-aware implication staleness: mark implication edge `stale` only when its provenance evidence chunks changed, not for every edge touching an updated node.
