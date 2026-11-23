# Unified dataset schema

See `datasets/unified_schema/layout.md` and the JSON Schemas beside it. Each public dataset gets an
adapter in `src/data/adapters/` that converts its units, coordinate frame and skeleton into this schema.
Splits are recording-level and subject-independent (`scripts/make_splits.py`); never split by frame.
