# NOTE: This Dockerfile is only used in development. It provides a runtime
# environment where the JavaScript build process can run. In production the
# files built by this process are served from disk, while in development a HTTP
# server that's distributed with the UI build tools is used.
FROM node:24.22.1-alpine

# Setup a spot for our code
WORKDIR /ui

# Install dependencies
COPY package.json yarn.lock ./
RUN --mount=type=cache,target=/root/.yarn YARN_CACHE_FOLDER=/root/.yarn yarn install --frozen-lockfile

# Copy in the source code
COPY . .

ENTRYPOINT [ "yarn" ]
CMD [ "start" ]
