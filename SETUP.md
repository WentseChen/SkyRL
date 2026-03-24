# SkyRL Mini SWE Agent Training Setup

## Training Script

```bash
cd ~/SkyRL && source setup_env.sh && bash examples/train/mini_swe_agent/run_mini_swe_8B.sh 2>&1 | tee ~/training.log
```

Log is written to `~/training.log`.

## Known Issues & Fixes

### Docker Hub Rate Limit (podman exit status 125)

**Symptom:** Training crashes during eval with:
```
ValueError: Found no valid responses for this step. This means that generation failed
for all trajectories, likely due to errors in environment setup.
```
Underlying cause in logs:
```
Command '['podman', 'run', '-d', '--name', '...', 'docker.io/swebench/...', 'sleep', '2h']'
returned non-zero exit status 125.
toomanyrequests: You have reached your unauthenticated pull rate limit.
```

**Root cause:** The SWE-Bench evaluation containers are pulled from Docker Hub on demand at runtime.
Without authentication, Docker Hub enforces a 100 pulls/6h rate limit, which is quickly exhausted
when running 50+ parallel eval trajectories.

**Fix:** Log in to Docker Hub before starting training:
```bash
podman login docker.io -u <username> -p <password>
```
Credentials are cached and persist across sessions. Authenticated pulls have a higher rate limit
(200/6h for free accounts, unlimited for paid).

**Note:** The training data (getmoto/moto, python/mypy) and most validation data images are not
pre-pulled locally. Only django/django and astropy/astropy images (250 total) are cached on disk.
All other images will be pulled on first use.

### Prompt/Response Count Mismatch (AssertionError)

**Symptom:** Training crashes with:
```
AssertionError: Mismatch between prompts (50) and responses (16)
```

**Root cause:** Individual trajectories can fail during eval (e.g., context length exceeded),
causing them to be filtered out. The original validation asserted a strict equality between
the number of input prompts and returned responses, which fails when some trajectories error out.

**Fix:** Changed the assertion in `skyrl/train/utils/trainer_utils.py:613` from `==` to `<=`,
allowing fewer responses than prompts as long as at least one valid response exists.
