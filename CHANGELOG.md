# Changelog

## [1.2.7] - 2026-09-02

### Added
- Add Qwen PLE engine support:
  - `EngramConfig.engine="qwen_ple"`
  - `table_spec="PLE_QWEN_V1"`
  - `table_source="engramdb:store"`
- Add disk-backed PLE table configuration fields:
  - `prime_sizes`
  - `table_store_path`
  - `table_model_dir`
  - `table_shards`, `table_rows_per_shard`, `table_width`
  - `table_dtype`, `table_scale`, `table_cache_size`
- Add `QwenPleHashMapping` and `create_hash_mapping()` factory.
- Add automatic EngramDB Store-I PLE table installation through
  `_install_engramdb_table_if_requested()` when `table_source="engramdb:store"`.

### Changed
- Keep `engine` defaulting to `"deepseek"` so existing users are unaffected.
- For `qwen_ple` / `PLE_QWEN_V1`, skip tokenizer compression and hash original
  Qwen token ids directly.
- Route `EngramLayer`, `EngramModel`, data collator, and weight transfer through
  the new hash-mapping factory.

### Fixed
- Make `examples/engram_knowledge_memory.py` use CPU when CUDA is unavailable,
  avoiding bf16 validation failure on CPU-only machines.
