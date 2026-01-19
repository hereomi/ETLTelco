from sqlalchemy import Column, Integer, String, MetaData, Table, create_engine, text
from crud import auto_insert, auto_upsert

def test_refactor():
    print("Initializing In-Memory SQLite DB...")
    eng = create_engine("sqlite:///:memory:")
    meta = MetaData()
    users = Table(
        "users",
        meta,
        Column("id", Integer, primary_key=True),
        Column("username", String(50), unique=True),
        Column("city", String(50)),
    )
    meta.create_all(eng)

    print("Testing auto_insert...")
    initial = [
        {"id": 1, "username": "alice", "city": "Paris"},
        {"id": 2, "username": "bob", "city": "London"},
    ]
    # auto_insert(engine, data, table, ...)
    rows_inserted = auto_insert(eng, initial, users)
    print(f"Inserted {rows_inserted} rows.")

    print("Testing auto_upsert...")
    upserts = [
        {"id": 1, "username": "alice", "city": "Lyon"},     # update
        {"id": 3, "username": "charlie", "city": "Berlin"}, # insert
    ]
    # auto_upsert(engine, data, table, constrain=[...])
    stats = auto_upsert(eng, upserts, users, constrain=["id"])
    print(f"Upsert Stats: {stats}")

    print("Verifying DB Content...")
    with eng.connect() as c:
        rows = c.execute(text("select id, username, city from users order by id")).fetchall()
        for r in rows:
            print(dict(r._mapping))
            
    # Quick assert
    expected_alice_city = "Lyon"
    found = False
    for r in rows:
        if r.id == 1 and r.city == expected_alice_city:
            found = True
            break
    
    if found:
        print("✅ SUCCESS: Data updated correctly using new CRUD module.")
    else:
        print("❌ FAILURE: Data mismatch.")

if __name__ == "__main__":
    test_refactor()
