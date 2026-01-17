# Implementation Plan: Future Roadmap (Usability, Integration & Growth)

This document outlines the technical plan for implementing the **Usability** (CLI, Web UI), **Integration** (Airflow, dbt), and **Deployment** strategy for the ETL engine.

## Goal

Transform the ETL engine into a consumable product hosted at `aips.pro`, aiming for **50 GitHub stars** within 5 months.

---

## Phase 1: Usability (The Interface Layer)

### 1. CLI Tool (`etl_cli.py`)

* **Technology**: `click` or `argparse`.
* **Commands**: `load`, `inspect`, `init`.
* **Interface**: `telcoetl load source.csv --db ...`

### 2. Web UI (`etl_ui.py`)

* **Technology**: `streamlit`.
* **Tabs**: Auto ETL, Create, Append, Upsert, Update.
* **Features**: Connection Manager, Data Preview, Real-time Logs.

---

## Phase 1.5: Security & Governance (Anti-Abuse)

### 1. User Registration & Auth

**Goal**: Track usage to prevent abuse.

* **Implementation**: Simple `sqlite` user DB (`users.db`).
* **Flow**: Signup (Email/Pass) -> Login -> Session State.

### 2. Rate Limiting (The "5 Tests Rule")

**Goal**: Enforce 5 ETL executions per user per day.

* **Logic**:
  * Table `usage_logs`: `user_id`, `date`, `count`.
  * **Pre-Execution Check**: `if usage_count >= 5: raise error`.
  * **Reset**: Count resets at 00:00 UTC.

### 3. Data Retention & Privacy (7-Day TTL)

**Goal**: STRICT Governance. "We never look at your data. It is for testing."

* **Policy**: All data is purely for testing purposes. We strictly DO NOT inspect, sell, or use public data.
* **Mechanism (7-Day Cleanup)**:
  * All created tables are tagged with `created_at` timestamp (or tracked in `users.db`).
  * **Garbage Collector**: A daily Cron job runs:

        ```sql
        -- Pseudo-code
        FOR table IN user_tables:
            IF created_at < NOW() - 7 DAYS:
                DROP TABLE table CASCADE;
        ```

* **User Action**: Users can still click "Delete My Data" to wipe it instantly if they chose.

### 4. Ephemeral Logs

**Goal**: "Clear log by 3 hours".

* **Mechanism**: A background thread or Cron job.
* **Command**: `find logs/ -mmin +180 -delete`.

---

## Phase 2: Integration (The Enterprise Layer)

### 3. Airflow Operator (`telco_provider.py`)

* **Code**: Custom `BaseOperator` wrapping `ETL` class.

### 4. dbt Integration (`dbt_runner.py`)

* **Pattern**: Pre-hook seed loader for large datasets.

---

## Phase 3: Deployment (Hybrid Cloud Architecture)

**Context**: User has valid Oracle/MSSQL licenses/setup on a local workstation. The VPS (2GB) is too small for these DBs.
**Solution**: **Cloudflare Tunnel** (Zero Trust) to securely expose local DB ports to the VPS.

### 1. Architecture Components

#### A. The VPS (`aips.pro`) - The "Frontend"

* **Resources**: 1 vCPU, 2GB RAM.
* **Services**:
  * `Streamlit App` (TelcoETL UI).
  * `Cloudflared` (To ingest traffic from the world).
* **Connection**: Connects to DBs via the Tunnel Network.

#### B. The Workstation - The "Backend"

* **Resources**: Powerful local machine.
* **Services**:
  * `Oracle DB` (Container/Native).
  * `MSSQL` (Container/Native).
  * `Cloudflared` (To expose DB ports *only* to the VPS).

### 2. Podman / Docker Composition

**File 1: `docker-compose.vps.yml` (For `aips.pro`)**

```yaml
version: '3.8'
services:
  app:
    build: .
    restart: always
    volumes:
       - ./users.db:/app/users.db # Persist auth data
```

**File 2: `docker-compose.local.yml` (For Workstation)**

```yaml
services:
  oracle:
    image: container-registry.oracle.com/database/express:latest
    ports: ["1521:1521"]
```

---

## Phase 4: Growth Strategy (The "50 Stars" Plan)

To reach 50 stars, the project needs **Visibility** and **Documentation**.

### 1. Documentation Site (`docs.aips.pro`)

* **Tool**: `MkDocs`.
* **Content**: "Hyrid Cloud ETL: Connecting Cloud Apps to On-Prem Data".

### 2. Content Marketing

* **Month 1**: Release v1.0.
* **Month 2**: Article: *"How I ran a production ETL pipeline for $5/mo using my gaming PC as the Database"*.
* **Month 3**: Video tutorial.

---

## Next Steps

1. **Phase 1**: Build the UI.
2. **Phase 1.5**: Implement Auth and Rate Limiting.
