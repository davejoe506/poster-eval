"""Pin the Airflow UI password before `airflow standalone` starts.

`airflow standalone` always mints a random admin password (and ignores
`_AIRFLOW_WWW_USER_PASSWORD`). Create/reset the user first so README
credentials (`admin` / `admin` by default) actually work.
"""

from __future__ import annotations

import os
import subprocess
import sys


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=False, text=True)


def ensure_www_user(username: str, password: str, airflow_home: str) -> None:
    migrate = _run(["airflow", "db", "migrate"])
    if migrate.returncode != 0:
        sys.exit(migrate.returncode)

    created = _run(
        [
            "airflow",
            "users",
            "create",
            "--username",
            username,
            "--firstname",
            "Admin",
            "--lastname",
            "User",
            "--role",
            "Admin",
            "--email",
            f"{username}@example.com",
            "--password",
            password,
        ]
    )
    if created.returncode != 0:
        reset = _run(
            [
                "airflow",
                "users",
                "reset-password",
                "--username",
                username,
                "--password",
                password,
            ]
        )
        if reset.returncode != 0:
            sys.exit(reset.returncode)

    password_path = os.path.join(airflow_home, "standalone_admin_password.txt")
    with open(password_path, "w", encoding="utf-8") as handle:
        handle.write(password)


def main() -> None:
    username = os.environ.get("_AIRFLOW_WWW_USER_USERNAME", "admin")
    password = os.environ.get("_AIRFLOW_WWW_USER_PASSWORD", "admin")
    airflow_home = os.environ.get("AIRFLOW_HOME", "/opt/airflow")
    print(
        f"standalone-entrypoint | ensuring UI user {username!r} matches "
        "_AIRFLOW_WWW_USER_PASSWORD",
        flush=True,
    )
    ensure_www_user(username, password, airflow_home)
    os.execvp("airflow", ["airflow", "standalone"])


if __name__ == "__main__":
    main()
