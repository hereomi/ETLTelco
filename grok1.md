# ETLTelco Architecture and Security Audit Blueprint

## Module Structure and Classification

### Core Pipeline
- **etl/**: Main ETL engine components (ops.py: ETL class, load_execute.py, casting_pipeline.py, ddl_sync.py, io.py, context.py, pk.py, dir_tosql.py, gen_config.py)
- **main.py**: CLI entry point
- **etl.py**: Package initialization

### Utility
- **utils/**: Shared utilities (casting.py, cast.py, data_cleaner.py, data_profiler.py, logger.py, sanitization.py, schema_manager.py, crud_v2.py, crud_v3.py, crud_harness.py, call_crud.py, built_where.py, where_build.py, upsert.py)
- **crud/**: Parameterized SQL operations (common.py, core.py, query_builder.py, sanitization.py)
- **schema_align/**: Data alignment and validation (api.py, core.py, config.py, validators.py, performance.py, diagnostics.py, schema_analyzer.py, utils.py)

### Adapter
- **crud/dialects/**: Database-specific CRUD implementations (postgres.py, oracle.py, mysql.py, sqlite.py, mssql.py)
- **ddl/**: Dialect-aware DDL generation (ddl_common.py, ddl_database.py, ddl_func.py, ddl_io.py, ddl_mssql.py, ddl_mysql.py, ddl_oracle.py, ddl_postgre.py, ddl_sqlite.py)

### Config
- **config.yml**: Global configuration
- **gen_config.py**: Configuration generation utility

### Legacy
- **legacy/**: Deprecated implementations (schema_mgr/, crud_v1.py, ddl_create.py, ddl_minimal.py, ddl_oracle.py, df_to_schema.py, schema_analyzer.py, sql_corrector.py)

## Dependency Graph (Text-Based)

```
main.py → etl.py → etl/ops.py
                    ↓
                    etl/ (internal dependencies)
                    ↓
crud/ ←→ schema_align/
  ↓         ↓
utils/     utils/
  ↓         ↓
ddl/       ddl/
```

Key couplings:
- etl/ops.py imports crud/, utils/, ddl/ (via ddl_sync.py)
- crud/__init__.py imports schema_align/, utils/
- schema_align/ imports utils/ (indirectly via logger)
- ddl/ is self-contained with dialect submodules
- utils/ has internal dependencies but no external imports

No circular dependencies detected.

## Module Analysis Table

| Module | Classification | Coupling Type | Risk Level | Notes |
|--------|----------------|---------------|------------|-------|
| etl/ | Core Pipeline | High (imports crud, utils, ddl) | Medium | Central orchestration, complex dependencies |
| crud/ | Utility | High (imports schema_align, utils) | Medium | Core data operations, dialect adapters |
| utils/ | Utility | Medium (internal) | Low | Shared utilities, some unused modules |
| schema_align/ | Utility | Medium (imported by crud) | Low | Data validation, no external deps |
| ddl/ | Adapter | Low (self-contained) | Low | DDL generation, dialect-specific |
| legacy/ | Legacy | None | High | No active imports, archival candidate |
| main.py | Core Pipeline | Low (imports etl) | Low | Simple CLI wrapper |
| config.yml | Config | None | Low | Static configuration |

## Findings

### Explicit Coupling Analysis
- **etl/ops.py** heavily coupled to crud/, utils/, ddl/ via direct imports and function calls
- **crud/__init__.py** instantiates DataAligner from schema_align/ and uses logger from utils/
- No circular dependencies found, but tight coupling between core pipeline and utilities
- Dialect adapters in crud/dialects/ and ddl/ follow consistent patterns but increase maintenance burden

### Implicit Coupling Analysis
- **Shared mutable state**: `utils/logger.py` global `ENABLE_LOGGING` modified by `etl/ops.py` - creates hidden configuration dependencies
- **Side-effectful functions**: Logging writes to filesystem, DB operations modify external state without clear contracts
- **Duck-typed interfaces**: Assumes SQLAlchemy engine interface throughout, no explicit protocols - ambiguous coupling requiring manual review
- **Global configuration**: Logger loads config.yml at import time, creating initialization order dependencies

### Data Flow Analysis
**Entry Points**: main.py (CLI), run_etl() (programmatic)

**Flow Path**:
1. Ingestion: `_to_dataframe()` → pandas.read_csv/excel (no path sanitization)
2. Transformation: `quick_clean()` → `cast_df()` → `sanitize_cols()` (validation via type inference thresholds)
3. DDL Generation: `_prepare_ddl_and_sync()` → dialect-specific DDL functions
4. Loading: `_load_and_validate()` → crud auto_insert/upsert/update

**Validation**: Type inference with configurable thresholds, NOT NULL checks, but no comprehensive schema validation
**Error Handling**: Basic try/except in main.py, inconsistent in pipeline (some return False, some raise)
**Secrets Handling**: db_uri passed as plain string, no encryption or secure storage mechanisms

### Security Risk Assessment
- **SQL Injection**: Low risk - uses parameterized queries (`sa.text(sql), params`) throughout crud operations
- **Unsafe Deserialization**: Low risk - `yaml.safe_load()` used for config, `json.loads()` for CLI args
- **Hardcoded Credentials**: None found - db_uri provided by user
- **Missing Input Sanitization**: Medium risk - file paths in `_to_dataframe()` not validated, potential path traversal if malicious input
- **External Data Processing**: Medium risk - pandas CSV/Excel readers can execute formulas or load malicious content

### Archival Recommendations
- **legacy/**: Complete archival - zero inbound calls, redundant with current implementations
- **utils/crud_v3.py**: Archival - only imported by unused `call_crud.py`
- **utils/crud_v2.py**: Archival - only used in test files
- **utils/upsert.py**: Archival - no inbound imports outside tests
- **utils/call_crud.py**: Archival - no inbound imports
- **ddl/demo_etl.py**: Archival - appears to be example code with no production usage

### Actionable Recommendations
1. Extract logger configuration to avoid global state mutations
2. Add path validation in `_to_dataframe()` to prevent directory traversal
3. Implement consistent error handling patterns across pipeline
4. Add comprehensive input validation for external data sources
5. Remove identified archival candidates to reduce maintenance burden
6. Consider explicit interfaces for SQLAlchemy dependencies to reduce duck typing risks