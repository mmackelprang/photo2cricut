# photo2cricut -- containerized converter
# Build:  docker build -t photo2cricut .
# Run:    docker run --rm -v "$PWD:/work" photo2cricut /work/photo.jpg /work/out.svg --method xdog
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir .

# default entrypoint is the converter; args are passed through
ENTRYPOINT ["photo2cricut"]
CMD ["--help"]
