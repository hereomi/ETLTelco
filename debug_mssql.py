import sqlalchemy.dialects.mssql as mssql
print(dir(mssql))
try:
    print(mssql.dml)
except AttributeError:
    print("mssql.dml not found")
