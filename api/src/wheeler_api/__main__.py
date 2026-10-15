"""`wheeler-api`: serwer HTTP.

W kontenerze nasłuchuje na 0.0.0.0, ale port jest publikowany wyłącznie na 127.0.0.1.
"""

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "wheeler_api.app:create_app",
        factory=True,
        host=os.environ.get("WHEELER_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("WHEELER_API_PORT", "8000")),
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
