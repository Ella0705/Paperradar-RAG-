# Paper Radar : A Research Agent that Personalized Paper Discovery for Academics

## The Problem

Academics do not just have a paper overload problem. They have a late-discovery problem.

By the time a researcher notices a truly relevant new paper, they may already have written the wrong literature review, weakened their novelty claim, or spent weeks moving in the wrong direction.

Most existing paper tools monitor the literature with generic keywords, categories, or citation graphs. But that is not how real research works. Your actual research agenda lives across your Overleaf drafts, your GitHub repos, and your local notes.

## What We Built

We built a personalized research radar for academics. Our agent reads that live research context, surveys new arXiv papers on a schedule, filters for what is actually relevant, summarizes the paper, and most importantly explains why it matters to your current work.

So the value is not just faster reading. The value is earlier awareness. We help researchers catch the papers that could change what they write next, before those papers become costly surprises.

This means stronger literature coverage, better novelty positioning, and less wasted time reading papers that do not matter. Instead of another generic feed, the researcher gets a personalized early-warning system for their actual projects.

The system integrates multiple research sources such as GitHub repositories, Overleaf drafts, and academic papers. By combining retrieval, novelty analysis, and local context comparison, the agent can highlight overlapping ideas, surface relevant work, and help researchers decide whether a paper is worth reading in full.

PaperRadar demonstrates how AI agents can move beyond simple automation and become decision-support systems. It transforms the chaotic research workflow into an actionable pipeline: monitor sources, analyze papers, compare with ongoing work, and generate insights.

This project aligns with the “Build Agents That Think” track by implementing a research-focused AI agent system that reasons over multiple sources and produces structured decisions. Instead of just summarizing papers, PaperRadar helps researchers think — identifying novelty, detecting overlap, and guiding research priorities.

## First-Time Experience

When you first interact with the skill, it runs a 2-minute onboarding:

```
Step 1  → Asks for your GitHub username
Step 2  → Fetches your repos (private + public via gh CLI)
Step 3  → Asks about Overleaf (3 connection options)
Step 4  → Optional: extra keywords, arXiv categories
Step 5  → Shows your draft profile for confirmation
Step 6  → Identifies your research strands (2-4 thematic clusters)
```

After onboarding, the agent knows your research strands and maps each discovered paper to your specific projects.
Even without connecting to GitHub or Overleaf, PaperRadar can analyze locally stored research drafts and provide reading recommendations for newly published papers.

## Feed Format

Each feed delivers exactly **3 papers** — one per research strand — with deep connection analysis:

```
📑 [STRAND TAG] — Must-read
Title · Authors · Date · category
2-3 sentence summary

🔗 Connection to your work (→ repo_name):
4-6 sentences: what it shares with your project, whether it is a
potential citation, competing approach, or useful method to adopt.

→ https://arxiv.org/abs/...
```

## Architecture

```text
Input (arXiv URL/ID or "pull me papers")
    |
    v
[Onboarding] --> profile.json (GitHub + Overleaf + manual)
    |
    v
[Discovery] --> arXiv API + Semantic Scholar + SSRN
    |
    v
[Orchestrator] -- parallel per paper:
    |
    +-- [Retriever Agent]       --> RelatedPapers
    +-- [Novelty Checker Agent] --> NoveltyReport
    +-- [Local Overlap Agent]   --> LocalOverlapReport
    |
    v
[Feed Assembler] --> 3 papers × strand-tagged connection analysis
    |
    v
Output (Discord / Telegram / CLI)
```

## Recent improvements (local RAG layer)

The multi-agent triage pipeline (Retriever / Novelty / Local Overlap + Assembler) is unchanged from the hackathon baseline. **Retrieval for `LocalOverlapAgent` was upgraded** in three ways:

1. **Structure-aware LaTeX chunking** — `.tex` sources are split on `\section` boundaries before secondary character splitting; section titles are stored in metadata and prefixed in chunk text so hits can be attributed (e.g. Related Work).
2. **MMR retrieval** — vector search uses Maximal Marginal Relevance so top-k chunks stay relevant to the query without repeating near-duplicate passages.
3. **Hybrid BM25 + vector (RRF)** — dense embeddings and BM25 keyword search run in parallel; results are fused with reciprocal rank fusion so terms like “LoRA” or “BitFit” remain discoverable alongside semantic matches.

Rebuild the index after changing sources: `python -m triage_agent.rag.build_index` (requires `pip install -e ".[rag]"` and `OPENAI_API_KEY`).

## Setup

```bash
git clone https://github.com/AustinJunyuLi/hackathon-research-agent.git
cd hackathon-research-agent
bash scripts/bootstrap_openclaw.sh
python scripts/verify_openclaw_install.py
```

### GitHub CLI (recommended)

For private repo access during onboarding:

```bash
# Install
brew install gh    # macOS
sudo apt install gh  # Linux

# Authenticate
gh auth login
```

Without `gh`, the agent falls back to the public GitHub API (public repos only).

## Configuration

- **Default backend is OpenClaw runtime** (`LLM_BACKEND=openclaw`) when no provider key is set.
- **When `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set**, the skill uses that provider (so Docker/OpenClaw can run real LLM instead of stub).
- **Standalone CLI**: optionally set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`.
- `SEMANTIC_SCHOLAR_API_KEY` optional but recommended.

### Using real OpenAI in OpenClaw / Discord (Docker)

If the bot runs in stub/offline mode (“TRIAGE_STUB_SUMMARIES” or “no usable LLM backend”), do the following:

1. **Pass your key into the container**  
   In the service that runs the skill (e.g. OpenClaw Docker Compose), add environment variables:
   - `OPENAI_API_KEY=<your-key>`  
   - Optional but recommended: `LLM_BACKEND=openai` (the skill will prefer OpenAI when the key is present anyway).

2. **Where to set them**  
   - **Docker Compose**: under the relevant service, add `environment:` with `OPENAI_API_KEY` and optionally `LLM_BACKEND=openai`, or use `env_file: .env` and put the key in a `.env` file (do not commit it).
   - **OpenClaw gateway/config**: if your Discord bot is started by OpenClaw, set the same variables in the environment that starts the OpenClaw process (e.g. in the same `docker-compose` service or in a `.env` loaded by Compose).

3. **Restart and test**  
   Restart the container/compose, then trigger the skill again (e.g. `/research-agent 2301.07041`). Check logs: you should see real LLM calls (and no “stub” or “TRIAGE_STUB_SUMMARIES” from this repo; if that message still appears, it is from OpenClaw’s side and may need to be turned off in OpenClaw’s config).

4. **Quick local check**  
   To confirm the skill uses OpenAI when the key is set, run locally:
   ```bash
   export OPENAI_API_KEY=sk-...
   python skill/scripts/run_triage.py 2301.07041 --format markdown
   ```
   You should get a full memo (not stub text).

## Usage

### Via OpenClaw

```bash
openclaw agent --agent main --message '/research-agent 2106.09685'
```

### Via CLI

```bash
source .venv/bin/activate
triage 2106.09685
triage https://arxiv.org/abs/2106.09685
triage --batch-file papers.txt --format json --output-dir out
```

### Source Enrollment

```bash
# Local drafts
python skill/scripts/enroll.py enroll local "Drafts" --path /path/to/drafts

# GitHub repo
python skill/scripts/enroll.py enroll github "My Repo" --url https://github.com/user/repo

# Overleaf project
python skill/scripts/enroll.py enroll overleaf "My Paper" --url https://git.overleaf.com/abc123 --token ol_xxx

# Sync all sources
python skill/scripts/enroll.py sync
```

## OpenClaw Skill

Skill folder under `skill/`. Install into your OpenClaw workspace:

```bash
cp -r skill/ ~/.openclaw/workspace/skills/research-agent/
```

### Enable Automated Survey

```bash
python3 skill/scripts/setup_daily_cron.py \
  --project-root "$(pwd)" \
  --time "08:00" --tz "Europe/London"
```

Default cadence: Mon/Wed/Fri at 08:00.

## Commands (inside OpenClaw)

```
/research-agent <arxiv-url-or-id>       — triage a specific paper
/research-agent batch <id1> <id2> ...   — triage multiple papers
/research-agent survey                  — run a fresh feed now
/research-agent interests               — show current profile
/research-agent setup                   — re-run onboarding
```

## Testing

```bash
bash scripts/bootstrap_openclaw.sh
python scripts/verify_openclaw_install.py
openclaw skills info research-agent
bash scripts/smoke_openclaw_install.sh
pytest tests/ -q
```

## Known Limitations

- Paper understanding is abstract-first (no PDF parsing)
- Semantic Scholar may throttle without an API key
- Overleaf enrollment requires an explicit Git mirror URL and token
