// server/clustering.js
// ─────────────────────────────────────────────────────────────────────────────
// AI-powered clustering for Tab Out.
//
// This module takes browsing data (either history or live open tabs),
// sends it to an LLM, and asks it to group the URLs into meaningful
// "missions" — clusters of intent like "researching AI tools" or
// "planning Tokyo trip".
//
// Works with any OpenAI-compatible API (DeepSeek, OpenAI, Groq, Ollama, etc.)
// configured via ~/.mission-control/config.json.
// ─────────────────────────────────────────────────────────────────────────────

const OpenAI = require("openai");
const config = require("./config");

// ─────────────────────────────────────────────────────────────────────────────
// LLM Client
// ─────────────────────────────────────────────────────────────────────────────

function getClient() {
  // If the API key is empty, the user probably added it to config.json after
  // the server started (e.g. during first-time setup). Try re-reading the
  // file before giving up — this avoids a confusing 401 error.
  if (!config.apiKey) {
    config.reloadFromDisk();
  }

  if (!config.apiKey) {
    throw new Error(
      "No API key configured. Add your key to ~/.mission-control/config.json " +
        '(set the "apiKey" field) and try again — no restart needed.',
    );
  }

  return new OpenAI({
    apiKey: config.apiKey,
    baseURL: config.baseUrl,
  });
}

// Call the configured LLM and return the text response
async function callLLM(prompt, maxTokens = 4000) {
  const client = getClient();
  const completion = await client.chat.completions.create({
    model: config.model,
    temperature: 0.3,
    max_tokens: maxTokens,
    messages: [{ role: "user", content: prompt }],
  });

  const text = completion.choices?.[0]?.message?.content;
  if (!text) throw new Error("LLM returned an empty response");
  return text;
}

// Parse JSON from LLM response (strips markdown fences if present)
function parseJSON(text) {
  let cleaned = text
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/, "")
    .trim();
  return JSON.parse(cleaned);
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared prompt rules (used by both history and open-tab clustering)
// ─────────────────────────────────────────────────────────────────────────────

const CORE_RULES = `
- Group by INTENT, not by domain. A research project might span GitHub, Stack Overflow, and blog posts — they belong in the same mission.
- Be SPECIFIC and descriptive in mission names. "Learning about Next.js Server Components" is better than "Web Development".
- Ignore social media noise (Twitter/X feeds, Reddit front page, YouTube home) unless they're clearly part of a specific research task.`.trim();

const GRANULARITY_RULES = `
- **Individual content pieces get their own mission.** Each YouTube video, blog post, article, or podcast is its own mission — named after the content (e.g., "Watching: Building Is the Easy Part Now | Mike Knoop"). Do NOT lump unrelated content tabs together under a generic name like "Watching AI Videos."
- **localhost tabs are vibe coding project previews.** Group localhost tabs by PORT number — each port is a different project. Name the mission after the page title or what the project appears to be (e.g., "Brand DNA Tool (localhost:5173)").
- **Tools and platforms that are clearly separate activities should not be lumped together.** Give each its own mission unless they're clearly part of the same research session.
- Prefer MORE granular missions over fewer broad ones. 15 specific missions is better than 5 vague ones.`.trim();

function getUserRules() {
  return config.customPromptRules
    ? `\nAdditional rules from the user:\n${config.customPromptRules}\n`
    : "";
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. OPEN TABS CLUSTERING (on demand, ephemeral)
//
// Groups currently open tabs into missions. NOT stored in DB.
// Includes a witty personal message about the user's tab habits.
// ─────────────────────────────────────────────────────────────────────────────

function buildOpenTabsPrompt(tabs) {
  const tabLines = tabs
    .map((tab, i) => {
      return `${i + 1}. [${tab.title || "(no title)"}] ${tab.url || ""}`;
    })
    .join("\n");

  return `You are an AI assistant that organizes currently open browser tabs into meaningful "missions."

Here are the user's currently open browser tabs:

${tabLines}

Group ALL of these tabs into missions. Every single tab must appear in exactly one mission.

Rules:
${CORE_RULES}
- Every tab must appear in exactly one mission — no orphans. If a tab is genuinely miscellaneous, create a "Miscellaneous" mission.
- Write a concise 1-sentence summary for each mission. Sound like a casual human note to yourself — never say "the user" or "the reader." Examples: "Digging into how constitutional AI actually works." or "Setting up Stripe billing for the new project."

Also write a "personalMessage" — a single witty, slightly humorous observation about this person based on their open tabs. Write it as if you're a friend glancing at their screen. Keep it warm, specific to their actual tabs (reference real things you see), and under 20 words. Examples:
- "12 AI tools open and counting. You're either building the future or procrastinating on it."
- "Three localhost projects, two research rabbit holes, and a recipe. Classic."
- "The USCIS tab has been there for weeks, hasn't it? Hang in there."

${GRANULARITY_RULES}
${getUserRules()}
Respond ONLY with valid JSON (no markdown, no explanation):
{
  "personalMessage": "Your witty one-liner here.",
  "missions": [
    {
      "name": "Short, specific mission name",
      "summary": "One sentence about this mission",
      "tab_indices": [1, 2, 5]
    }
  ]
}`;
}

function parseOpenTabsResponse(responseText, tabs) {
  const parsed = parseJSON(responseText);

  if (!parsed || !Array.isArray(parsed.missions)) {
    throw new Error(`Response missing "missions" array`);
  }

  const personalMessage = parsed.personalMessage || null;

  const missions = parsed.missions.map((mission) => {
    const tabIndices = Array.isArray(mission.tab_indices)
      ? mission.tab_indices
      : [];
    const resolvedTabs = tabIndices.map((idx) => tabs[idx - 1]).filter(Boolean);
    return {
      name: mission.name || "Unnamed Mission",
      summary: mission.summary || "",
      tabs: resolvedTabs,
    };
  });

  return { missions, personalMessage };
}

async function clusterOpenTabs(tabs) {
  if (!tabs || tabs.length === 0)
    return { missions: [], personalMessage: null };

  console.log(`[clustering] Clustering ${tabs.length} open tabs...`);

  const responseText = await callLLM(buildOpenTabsPrompt(tabs), 3000);
  console.log(`[clustering] LLM responded (${responseText.length} chars)`);

  const { missions, personalMessage } = parseOpenTabsResponse(
    responseText,
    tabs,
  );
  console.log(`[clustering] Got ${missions.length} missions`);

  return { missions, personalMessage };
}

// ─────────────────────────────────────────────────────────────────────────────
module.exports = { clusterOpenTabs };
