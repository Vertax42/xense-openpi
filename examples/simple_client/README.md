# Simple Client

A minimal client that sends observations to the server and prints the inference rate.

You can specify which runtime environment to use using the `--env` flag. You can see the available options by running:

```bash
python examples/simple_client/main.py --help
```

## With Docker

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/simple_client/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
python examples/simple_client/main.py --env DROID
```

Terminal window 2:

```bash
python scripts/serve_policy.py --env DROID
```
