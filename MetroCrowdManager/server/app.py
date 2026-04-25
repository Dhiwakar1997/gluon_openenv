"""
FastAPI app for the MetroCrowdManager MCP environment.

`create_app` is given the generic `Action` base class as `action_cls` so
that incoming MCP payloads (`list_tools`, `call_tool`) are routed by the
serializer to their built-in classes. Our own `SubmitResponseAction` is
validated inside the environment's `_step_impl`.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install with `uv sync`."
    ) from e

try:
    from ..models import MetrocrowdmanagerObservation, SubmitResponseAction
    from .MetroCrowdManager_environment import MetrocrowdmanagerEnvironment
except (ImportError, ModuleNotFoundError):
    from models import MetrocrowdmanagerObservation, SubmitResponseAction
    from server.MetroCrowdManager_environment import MetrocrowdmanagerEnvironment


# `SubmitResponseAction` has `extra="allow"`, so the HTTP serializer
# happily validates *any* JSON payload (including MCP `list_tools` /
# `call_tool` shapes). The env's `step()` then re-coerces those into the
# right MCP action type before invoking the parent dispatcher.
app = create_app(
    MetrocrowdmanagerEnvironment,
    SubmitResponseAction,
    MetrocrowdmanagerObservation,
    env_name="MetroCrowdManager",
    max_concurrent_envs=1,
)


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
