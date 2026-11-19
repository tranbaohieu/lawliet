# syntax=docker/dockerfile:1

# ---- build ----------------------------------------------------------------
FROM golang:1.27-alpine AS build
WORKDIR /src
COPY . .
ARG VERSION=dev
# SIMD=1: AVX2 distance kernels (GOEXPERIMENT=simd, experimental); default scalar.
ARG SIMD=
RUN if [ "$SIMD" = 1 ]; then export GOEXPERIMENT=simd; fi \
 && CGO_ENABLED=0 go build -trimpath \
      -ldflags "-s -w -X github.com/tranbaohieu/lawliet/internal/version.Version=${VERSION}" \
      -o /out/lawliet ./cmd/lawliet \
 && mkdir -p /out/data

# ---- run ------------------------------------------------------------------
# Static binary on distroless: no shell, runs as a non-root user.
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/lawliet /lawliet
COPY --from=build --chown=nonroot:nonroot /out/data /data
EXPOSE 6380
ENTRYPOINT ["/lawliet"]
# Listen on all interfaces inside the container (the binary defaults to 127.0.0.1).
# Persistence: add "--dir", "/data" and mount a volume on /data.
CMD ["--bind", "0.0.0.0"]
