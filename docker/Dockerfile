# syntax=docker/dockerfile:1

FROM alpine:latest AS build
LABEL maintainer="ikhlef_ali@outlook.com" description="IKHLEF Ali"
LABEL maintainer="younesyuuyuu@gmail.com" description="HAMZA Younes"

# Install the required tools 

RUN apt-get update && apt-get install -y \
    git \
    tree

# do stuff

RUN echo "hello" > /opt/file

# do stuff


# This results in a single layer image
FROM scratch
COPY --from=build /opt/file /opt/file
# ENTRYPOINT [ "executable" ]
# CMD [ "executable" ]
