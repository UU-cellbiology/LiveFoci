# Linux/ARM fix for the `lift` ImageJ trackers (NND_utils + NGMA_utils).
#
# Problem: at runtime `lift` downloads two bundles from Zenodo, each
# containing a *Windows* ImageJ + Windows JRE, then calls
#   /root/.lift-foci/<bundle>/ImageJ/jre/bin/java
# which can't exist/run on Linux. The ImageJ bytecode (ij.jar, macros,
# plugins) is fine — only the bundled `jre` folders are Windows-specific.
#
# Fix: bring in a real multi-arch Linux Java 8, download + extract BOTH
# bundles at build time so lift's "does the dir exist?" cache check passes
# and it never downloads at runtime, then replace every bundled Windows
# `jre` with a symlink to the Linux JVM.
#
# Fully self-contained — no files needed in the build context.
#
# Build:  podman build -t lift-gpu-fixed -f Dockerfile.fix .
# Run:    podman run --rm --gpus all -v "%CD%\data:/data" lift-gpu-fixed \
#             run /data/20241120_Xray_25_Gy --steps 2

ARG BASE=krijns/lift:gpu-base
FROM ${BASE}

# A real Linux Java 8, matching the Windows JRE 8 the bundles shipped.
# Temurin is multi-arch, so buildx pulls arm64 or amd64 to match the target.
COPY --from=eclipse-temurin:8-jre /opt/java/openjdk /opt/java/openjdk

# Pre-seed both bundles and swap their Windows JREs for the Linux one.
# NOTE: this pulls from Zenodo *Sandbox*, which is ephemeral — if the record
# has been purged, download the tarballs from wherever they now live and use
# the COPY variant at the bottom of this file instead.
ARG ZENODO="https://sandbox.zenodo.org/records/547985/files"
RUN set -eux; \
    mkdir -p /root/.lift-foci; \
    for b in NND_utils NGMA_utils; do \
        python3 -c "import urllib.request,sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
            "${ZENODO}/${b}.tar.gz?download=1" "/tmp/${b}.tar.gz"; \
        tar -xzf "/tmp/${b}.tar.gz" -C /root/.lift-foci; \
        rm "/tmp/${b}.tar.gz"; \
    done; \
    # Replace every bundled Windows JRE with the Linux one (handles both
    # bundles generically, whatever the exact folder depth):
    find /root/.lift-foci -type d -name jre | while read -r d; do \
        echo "swapping Windows JRE -> Linux: $d"; \
        rm -rf "$d"; ln -s /opt/java/openjdk "$d"; \
    done; \
    # Sanity: the java the tracker calls must now work...
    for j in $(find /root/.lift-foci -path '*/jre/bin/java'); do "$j" -version; done; \
    # ...and warn about any *other* Windows-native code left in the bundles
    # (a plugin .dll would need its own Linux build — investigate if found):
    echo "Remaining native .dll files (should be none):"; \
    find /root/.lift-foci -name '*.dll' -print

# ── Offline / sandbox-is-gone alternative ────────────────────────────────
# If you can't rely on the Zenodo download at build time, extract the two
# bundles once, `rm -rf */ImageJ/jre` from each, drop them next to this file
# as ./NND_utils and ./NGMA_utils, and replace the RUN block above with:
#
#   COPY NND_utils  /root/.lift-foci/NND_utils
#   COPY NGMA_utils /root/.lift-foci/NGMA_utils
#   RUN find /root/.lift-foci -type d -name jre -exec rm -rf {} + ; \
#       ln -s /opt/java/openjdk /root/.lift-foci/NND_utils/ImageJ/jre; \
#       ln -s /opt/java/openjdk /root/.lift-foci/NGMA_utils/ImageJ/jre