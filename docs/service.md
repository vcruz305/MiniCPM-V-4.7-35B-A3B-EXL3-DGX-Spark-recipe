# Running safely

Serving is foreground-only by default. No service is installed or enabled by
setup, and no existing GPU process is stopped. Authentication stays enabled,
CORS defaults to an empty browser allowlist, and remote content fetching is
disabled. For remote access, keep the loopback bind and use an SSH tunnel:

```bash
ssh -L 8898:127.0.0.1:8898 USER@YOUR_SPARK
```

Set `TABBY_API_KEY` to the inference key on the client. Never publish the
admin key, api_tokens.yml, or a private deployment receipt. The server runs in
the adopted TabbyAPI directory, so that server's private key store remains there.

`STATE_DIR` is locked for the life of the launched server. Busy ports fail instead
of silently changing numbers. Optional `GPU_LOCK_FILE` provides cooperative
exclusion. Stop the child with Ctrl-C. The launcher forwards termination only to
its own child; it never uses pkill or kills unrelated workloads.

A preview uses DRY_RUN=1 and only writes config.preview.yml. Actual config and
runtime receipts are stored privately under STATE_DIR. No request prompts or
response contents are enabled in server debug logs by this recipe. Benchmark
helpers save the explicit public test prompts and replies in your chosen local
results directory; review them before publication.
