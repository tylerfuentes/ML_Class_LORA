#!/usr/bin/env python3
from pathlib import Path

import psycopg2


def load_pgpass(path: str = "/home/nathanaelguitar/.pgpass") -> tuple[str, str, str, str, str]:
    raw = Path(path).read_text(encoding="utf-8").strip()
    parts = raw.split(":", 4)
    if len(parts) != 5:
        raise ValueError(f"{path} must contain exactly 5 colon-separated fields")
    return tuple(parts)  # type: ignore[return-value]


def main() -> None:
    host, port, dbname, username, password = load_pgpass()
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=username,
        password=password,
        sslmode="require",
        connect_timeout=15,
    )
    cur = conn.cursor()
    cur.execute("select current_user, current_database()")
    current_user, current_database = cur.fetchone()
    print("connected", True)
    print("current_user", current_user)
    print("current_database", current_database)
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
