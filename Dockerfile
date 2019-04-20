FROM python:3.6.2

WORKDIR /brain/src
ADD . /brain/src

ENV ENVIRONMENT production
ENV PYTHONUSERBASE /venv
ENV PATH "/venv/bin:${PATH}"
ENV PYTHONPATH "$PYTHONPATH:/brain/src:/brain/src/tests"
ARG GITHUB_TOKEN
RUN ./docker/setup.sh

EXPOSE 5000

ENTRYPOINT ["./docker/entrypoint.sh"]
CMD ["./manage.py", "run_production_server"]
