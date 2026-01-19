---
tags: [sql]
title: Harness
created: '2026-01-18T17:40:02.925Z'
modified: '2026-01-18T17:41:39.895Z'
---

# Harness

I have built a crud.py module with functions for generating insert, upsert and update sql query and param from dataframe. Now I want to test whether those function can really clean a DataFrame and whether they can produce the correct CRUD syntax from it by executing generated statement with real sql database.

I want to create a utility module that will hand me three DataFrames prepared for INSERT, UPSERT and UPDATE, all derived from one original DataFrame.

### A. Base Dataset

1. The original DataFrame (`df_src`) contains **exactly 10 rows**, indexed `0–9`.
2. Column categories are known:

   * **Primary key column(s)** uniquely identify a row.
   * **Constraint columns** (non-PK) are used for UPDATE filtering.
   * **Mutable columns** are safe to change during testing.
3. Primary key values are **stable and unique** across the DataFrame.
4. Column dtypes are preserved when modifying values.

---

### B. Row Selection Rules

| Operation | Rows Selected | Purpose                    |
| --------- | ------------- | -------------------------- |
| INSERT    | `0–5`         | New rows to be inserted    |
| UPSERT    | `3–7`         | Mix of existing + new rows |
| UPDATE    | `1, 7, 8`     | Rows for update testing    |

---

### C. Effective Database State Assumption

1. The database is assumed to be **empty before testing begins**.
2. INSERT is executed **first**, inserting rows `0–5`.
3. After INSERT:

   * Rows `0–5` exist in the database.
   * Rows `6–9` do not exist yet.

---

### D. UPSERT Behavior Assumptions

1. UPSERT DataFrame includes rows `3–7`.
2. Row classification during UPSERT:

   * Rows `3–5` → **existing rows** (already inserted)
   * Rows `6–7` → **new rows**
3. For existing rows (`3–5`):

   * Only **non-PK, non-constraint columns** may be modified.
   * Modified values must:

     * Respect original dtype
     * Be deterministic (no randomness)
4. For new rows (`6–7`):

   * Values remain unchanged from source DataFrame.
5. Primary key columns are **never modified**.
6. UPSERT execution results in:

   * Rows `0–7` existing in the database.

---

### E. UPDATE Behavior Assumptions

1. UPDATE DataFrame contains rows `1`, `7`, and `8`.
2. Row classification:

   * Rows `1` and `7` → **exist in DB**
   * Row `8` → **does not exist in DB**
3. UPDATE operation:

   * Uses **constraint columns only** for row matching.
   * Modifies only **mutable columns**.
4. Expected outcomes:

   * Rows `1` and `7` are updated successfully.
   * Row `8` produces **no change** (zero rows affected).
   * No error is raised for row `8`.

---

### F. Validation & SELECT Assumptions

1. Each operation returns a **deterministic SELECT statement**:

   * INSERT → select rows `0–5` by PK
   * UPSERT → select rows `3–7` by PK
   * UPDATE → select rows `1`, `7`, `8` by constraint columns
2. Validation compares:

   * Source DataFrame
   * Prepared test DataFrame
   * Database-selected DataFrame
3. Validation checks:

   * Correct rows affected
   * Correct columns modified
   * No unintended changes

---

### G. Final Expected Database State

| Row Index | Exists | Modified            |
| --------- | ------ | ------------------- |
| 0         | Yes    | No                  |
| 1         | Yes    | Yes (UPDATE)        |
| 2         | Yes    | No                  |
| 3         | Yes    | Yes (UPSERT)        |
| 4         | Yes    | Yes (UPSERT)        |
| 5         | Yes    | Yes (UPSERT)        |
| 6         | Yes    | Inserted via UPSERT |
| 7         | Yes    | Updated via UPDATE  |
| 8         | No     | No                  |
| 9         | No     | No                  |

---

### H. Non-Goals (Explicitly Out of Scope)

* No delete testing
* No schema evolution
* No random data mutation
* No concurrency testing

