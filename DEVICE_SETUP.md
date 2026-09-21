# Moving this project between devices

**Do not zip the folder.** Use git. The repo is the transport.

## Leaving a device

```bash
git add -A
git commit -m "wip: <what you did>"
git push
```

## Arriving at a device

First time on that machine:

```bash
git clone <your-repo-url> AI_Sandbox
cd AI_Sandbox
py -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Every later time:

```bash
git pull
```

Then recreate `.env` (see below). That's it.

## Why not zip?

| Thing | Survives a zip/move? | Why |
|---|---|---|
| Source code | Yes | Plain text |
| `.venv/` | **No** | Hardcodes absolute paths to this exact folder. Move it and the interpreter breaks. |
| `node_modules/` | No | Same, plus compiled native binaries |
| `.env` | Yes, but | It is gitignored on purpose -- keys must not reach GitHub |

`.venv/` is rebuilt from `requirements.txt` in about a minute. That is the whole
point of `requirements.txt` -- it is the portable form of the environment.

## The .env file

`.env` is gitignored and will **not** travel with the repo. Recreate it on each
device:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Keep the key in a password manager, not in a chat or a screenshot. If it ever
lands in a commit, rotate it -- git history keeps it forever otherwise.

## Rules that keep this working

1. Never write an absolute path (`C:/Users/Ridham/...`) in code. Import paths
   from `legal_ai.config`, which derives everything from `PROJECT_ROOT`.
2. Commit before you close the laptop, even if the code is broken. A `wip:`
   commit beats losing the day.
3. If `git pull` complains about divergent branches, you committed on two
   devices without pushing. `git pull --rebase` usually sorts it.
