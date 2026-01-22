---
description: Repository Information Overview
alwaysApply: true
---

# ETLTelco Repository Information Overview

## Repository Summary
ETLTelco is a high-performance ETL engine for bridging messy data to structured databases, supporting multiple SQL dialects (Oracle, PostgreSQL, SQLite, MySQL, MSSQL). It includes data sanitization, intelligent type inference, auto-DDL generation, and schema evolution.

## Repository Structure
- **crud/**: Utilities for parameterized SQL statements (SELECT/UPDATE/INSERT) with DataFrame-aware column alignment.
- **ddl/**: Dialect-aware DDL generation for creating and altering database tables.
- **schema_align/**: Schema matching and alignment tools for data ingestion.
- **utils/**: Shared utilities for casting, data cleaning, profiling, and CRUD operations.
- **legacy/**: Deprecated code and older implementations.
- **docs/**: Documentation, usage guides, and implementation notes.
- **log/**: Log files from test runs and operations.
- **schema/**: Sample JSON schemas for testing.
- Root files: Main ETL class, configuration, tests, and entry points.

## Projects

### ETLTelco (Main ETL Engine)
**Configuration File**: etl.py

#### Language & Runtime
**Language**: Python  
**Version**: 3.9+  
**Build System**: None (direct Python execution)  
**Package Manager**: pip

#### Dependencies
**Main Dependencies**:
- pandas: Data manipulation and analysis
- sqlalchemy: SQL toolkit and ORM
- typing: Type hints
- json: JSON handling
- pathlib: Path operations

**Development Dependencies**:
- argparse: Command-line argument parsing
- sys, warnings: Standard library utilities

#### Build & Installation
```bash
pip install pandas sqlalchemy
```

#### Testing
**Framework**: Custom test scripts (no formal framework)  
**Test Location**: Root directory (test_*.py files)  
**Naming Convention**: test_*.py  
**Configuration**: Inline in scripts

**Run Command**:
```bash
python test_etl_evolution.py
```

### crud
**Configuration File**: crud/__init__.py

#### Language & Runtime
**Language**: Python  
**Version**: 3.9+  
**Build System**: None  
**Package Manager**: pip

#### Dependencies
**Main Dependencies**:
- pandas
- sqlalchemy

#### Build & Installation
```bash
pip install pandas
```

#### Testing
**Framework**: None (integrated in main tests)  
**Test Location**: N/A  
**Naming Convention**: N/A  
**Configuration**: N/A

### ddl
**Configuration File**: ddl/__init__.py

#### Language & Runtime
**Language**: Python  
**Version**: 3.9+  
**Build System**: None  
**Package Manager**: pip

#### Dependencies
**Main Dependencies**:
- sqlalchemy

#### Build & Installation
```bash
pip install sqlalchemy
```

#### Testing
**Framework**: None  
**Test Location**: N/A  
**Naming Convention**: N/A  
**Configuration**: N/A

### schema_align
**Configuration File**: schema_align/__init__.py

#### Language & Runtime
**Language**: Python  
**Version**: 3.9+  
**Build System**: None  
**Package Manager**: pip

#### Dependencies
**Main Dependencies**:
- pandas
- sqlalchemy

#### Build & Installation
```bash
pip install pandas sqlalchemy
```

#### Testing
**Framework**: None  
**Test Location**: N/A  
**Naming Convention**: N/A  
**Configuration**: N/A

### utils
**Configuration File**: utils/ (various .py files)

#### Language & Runtime
**Language**: Python  
**Version**: 3.9+  
**Build System**: None  
**Package Manager**: pip

#### Dependencies
**Main Dependencies**:
- pandas
- sqlalchemy

#### Build & Installation
```bash
pip install pandas sqlalchemy
```

#### Testing
**Framework**: None  
**Test Location**: N/A  
**Naming Convention**: N/A  
**Configuration**: N/A

## AST Analysis Summary
Using AST parsing, the repository contains:
- **Functions**: 500+ functions across modules, including ETL methods (create_load, upsert, auto_etl), CRUD builders (build_select, build_insert), DDL generators, and utility functions.
- **Classes**: Key classes like ETL (main engine), with methods for data processing, schema management, and database operations.
- **Imports**: Primarily standard library (typing, json, pathlib), pandas, sqlalchemy, and internal modules. Relationships show tight coupling between ETL core, CRUD operations, DDL generation, and schema alignment utilities.
- **Relationships**: ETL class depends on utils (casting, cleaning), crud (insert/upsert), ddl (table creation), and schema_align (alignment). CRUD dialects extend base functionality for each database type.