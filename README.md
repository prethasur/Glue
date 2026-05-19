# Glue

BigEarthNet-S2 compositional generalisation utilities and NCI/Gadi run scaffolding.

## Legacy MVP Notes

Functional MVP for a postgraduate abroad consultancy content stack.

It discovers fresh scholarship and PhD vacancy calls from configured web feeds/pages, consolidates them against a predefined content calendar, drafts platform-specific content, sends approval cards to Telegram, and posts approved content through social adapters. By default, posting runs in safe `DRY_RUN` mode.

## What This MVP Does

- Collects opportunities from RSS feeds and simple web pages.
- Scores and deduplicates scholarship / PhD vacancy calls.
- Matches discoveries to a weekly content calendar.
- Creates drafts for LinkedIn, X, Instagram/Facebook video scripts, and Shorts.
- Sends drafts to Telegram for approval.
- Posts approved drafts through adapters, or logs the payload when `DRY_RUN=true`.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m abroad_social_stack run-once --dry-run
```

If you do not want a virtual environment, this project can also run with system Python 3.11+ because the MVP only uses Python standard library modules.

## Telegram Approval Loop

1. Create a Telegram bot with BotFather and copy the token.
2. Message your bot once from your Telegram account.
3. Add these values to `.env`:

```text
TELEGRAM_BOT_TOKEN=123456:abc...
TELEGRAM_CHAT_ID=your_chat_id
```

To discover your chat ID after messaging the bot:

```powershell
python -m abroad_social_stack telegram-updates
```

Then run:

```powershell
python -m abroad_social_stack run-once
python -m abroad_social_stack approvals
```

Approve in Telegram by replying:

```text
approve 12
reject 12 needs a more India-focused CTA
```

## Social Posting

Keep `DRY_RUN=true` until API credentials are ready.

Supported MVP adapters:

- LinkedIn: creates an organization/member UGC post when `LINKEDIN_ACCESS_TOKEN` and `LINKEDIN_AUTHOR_URN` are set.
- X: exposes a credential-ready adapter, but remains dry-run unless `X_BEARER_TOKEN` is set and the endpoint is enabled.
- Meta / Shorts: stores approved video scripts and prints handoff payloads. Direct publishing requires Meta Graph API permissions and YouTube Data API OAuth, which should be added after account verification.

## Core Commands

```powershell
python -m abroad_social_stack run-once --dry-run
python -m abroad_social_stack list-drafts
python -m abroad_social_stack send 1
python -m abroad_social_stack send-pending
python -m abroad_social_stack approve 1
python -m abroad_social_stack reject 1 "needs stronger CTA"
python -m abroad_social_stack approvals
python -m abroad_social_stack post-approved
python -m abroad_social_stack telegram-updates
```

## Files

- `config/sources.json`: discovery sources.
- `config/content_calendar.json`: predefined calendar pillars and schedule.
- `abroad_social_stack/`: app code.
- `data/stack.sqlite3`: generated local database.
- `docs/architecture.md`: system design and production path.

## Production Upgrade Path

1. Add a scheduler: GitHub Actions, cron, or a small VPS systemd timer.
2. Replace simple web-page scraping with search APIs such as SerpAPI, Tavily, Google Programmable Search, or institution APIs.
3. Add LLM generation by implementing `ContentGenerator.generate_with_llm`.
4. Add OAuth flows for LinkedIn, Meta, X, and YouTube.
5. Add a human content review dashboard if Telegram-only approval becomes too cramped.
