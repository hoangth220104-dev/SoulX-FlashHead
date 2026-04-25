from __future__ import annotations

import uvicorn

from flashhead_server.app import create_app
from flashhead_server.config import Settings

settings = Settings.from_env()
app = create_app(settings)
# def main() -> None:
#     settings = Settings.from_env()
#     app = create_app(settings)
#     uvicorn.run(
#         app,
#         host=settings.host,
#         port=settings.port,
#         reload=settings.debug,
#     )


# if __name__ == "__main__":
#     main()
