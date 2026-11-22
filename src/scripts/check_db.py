"""Creates the tables and prints what the database contains"""

from sqlalchemy import inspect, text

from mobi.create_db import get_engine, init_db

def main() -> None:
    """Creates missing tables, then prints server version and table names"""
    init_db()
    engine = get_engine()
    print("dialect:", engine.dialect.name)
    if engine.dialect.name == "postgresql":
        with engine.connect() as conn:
            print("servier:", conn.execute(text("SELECT version()")).scalar())
    print("tables:", sorted(inspect(engine).get_table_names()))

if __name__ == "__main__":
    main()