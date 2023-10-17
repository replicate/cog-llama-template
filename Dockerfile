# # from https://github.com/vercel/next.js/blob/canary/examples/with-docker/Dockerfile
# # Install dependencies only when needed
# FROM node:16-alpine AS jsdeps
# # Check https://github.com/nodejs/docker-node/tree/b4117f9333da4138b03a546ec926ef50a31506c3#nodealpine to understand why libc6-compat might be needed.
# RUN apk add --no-cache libc6-compat
# WORKDIR /app

# # Install dependencies 
# COPY mirror/package.json mirror/package-lock.json ./
# RUN npm ci;

# # Rebuild the source code only when needed
# FROM node:16-alpine AS next
# WORKDIR /app
# COPY --from=jsdeps /app/node_modules ./node_modules/
# COPY ./mirror /app/

# # Next.js collects completely anonymous telemetry data about general usage.
# # Learn more here: https://nextjs.org/telemetry
# # Uncomment the following line in case you want to disable telemetry during the build.
# ENV NEXT_TELEMETRY_DISABLED 1
# RUN npm run export

#####################################################

# FROM appropriate/curl as model
# RUN curl -o nya.tar https://r2-public-worker.drysys.workers.dev/nya-sd-8-threads-2023-04-04-.tar
# #RUN curl -o stable-hf-cache.tar https://r2-public-worker.drysys.workers.dev/hf-cache-ait-verdant-2022-11-30.tar
# # &&
# RUN tar -xf nya.tar -C / # /model/nya?
# RUN ls /model

# FROM appropriate/curl as ait
# RUN curl -o ait-verdant.tar https://r2-public-worker.drysys.workers.dev/ait-verdant-2022-11-30.tar
# # &&
# RUN mkdir /workdir
# RUN tar -xf ait-verdant.tar -C /workdir

FROM appropriate/curl as downloaders
RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.0.3/pget#cacheburst" && chmod +x /usr/local/bin/pget
RUN curl -o /usr/local/bin/remotefile -L "https://r2.drysys.workers.dev/remotefile" && chmod +x /usr/local/bin/remotefile

FROM python:3.11 as libbuilder
WORKDIR /app
RUN pip install poetry git+https://github.com/python-poetry/poetry.git git+https://github.com/python-poetry/poetry-core.git
RUN python3.11 -m venv /app/venv 
WORKDIR /app/
COPY ./pyproject.toml /app/
RUN VIRTUAL_ENV=/app/venv poetry install 
RUN mkdir nya
RUN pip install -t nya https://r2.drysys.workers.dev/torch/torch-2.1.0a0+git3af011b-cp311-cp311-linux_x86_64.whl
#RUN pip install -t nya https://r2.drysys.workers.dev/nyacomp-0.0.5-cp311-cp311-linux_x86_64.whl

FROM python:3.11-slim
WORKDIR /app
# COPY --from=ait /workdir/tmp /app/tmp
# COPY --from=model /model /app/model
COPY --link --from=downloaders /usr/local/bin/pget /usr/local/bin
COPY --link --from=downloaders /usr/local/bin/remotefile /usr/local/bin
COPY --link --from=libbuilder /app/venv/lib/python3.11/site-packages /app/
COPY --link --from=libbuilder /app/nya/ /app/
#COPY --from=next /app/out /app/next


ENV DISABLE_TELEMETRY=YES
ENV PRELOAD_PATH=/app/model/nya/meta.csv
ARG DEBUG=
ENV DEBUG=$DEBUG
RUN if [ -n "$DEBUG" ]; then apt update && apt install -yy kakoune gdb; fi
# unmangle this particular version
RUN ln -s /app/torch/lib/libcudart-9335f6a2.so.12 /app/torch/lib/libcudart.so.12
ENV LD_LIBRARY_PATH=/app/torch/lib

COPY --link ./exllama/ /app/exllama/
COPY --link ./src/ /app/src/
COPY --link ./predict.py ./config.py ./subclass.py /app/
COPY --link ./client.js ./index.html ./server.py /app/
# ew
COPY .env /app/

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/python3.11", "/app/server.py"]
