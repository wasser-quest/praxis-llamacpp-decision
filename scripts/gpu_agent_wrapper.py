"""gpu-agent launcher (supervisord program, priority 15).

Starts the pinned gpu-agent binary (baked in via COPY --from of the
praxis-gpu-router image) when this box is router-provisioned:
ROUTER_URL and PRAXIS_NODE_TOKEN must both be set. Boxes started
standalone (no router) exit 0 immediately — supervisord treats that as
expected (autorestart=unexpected) and stays quiet.
"""

import os
import sys

BINARY = "/opt/praxis/bin/gpu-agent"


def main() -> int:
    missing = [
        name
        for name in ("ROUTER_URL", "PRAXIS_NODE_TOKEN")
        if not os.environ.get(name, "").strip()
    ]
    if missing:
        print(
            "gpu-agent disabled (missing env: %s) — standalone box, no router call-home"
            % ", ".join(missing)
        )
        return 0
    if not os.access(BINARY, os.X_OK):
        print("gpu-agent binary missing at %s — rebuild image with PGPU_IMAGE" % BINARY)
        return 0
    print("gpu-agent: call-home to %s" % os.environ["ROUTER_URL"])
    sys.stdout.flush()
    os.execv(BINARY, [BINARY])


if __name__ == "__main__":
    raise SystemExit(main())