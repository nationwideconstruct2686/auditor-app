// Auditor AI: the only place the app talks to Claude. Keeps the API key on the server,
// checks the caller is a family member, caps daily use, and records real token counts.
import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "npm:@supabase/supabase-js@2";
import Anthropic from "npm:@anthropic-ai/sdk";

const MODEL = "claude-haiku-5-5";
const DAILY_CALL_CAP = 800; // about 4 hours of auditing per person per day

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

const RULES = `You assist a scripted somatic-processing session. The script asks every question;
you never ask questions, give advice, interpret, diagnose, or add your own ideas.
Use plain, calm, warm language. Never mention being an AI.`;

const TASKS: Record<string, (p: any) => { prompt: string; maxTokens: number; json: boolean }> = {
  // A short, neutral acknowledgment that repeats the person's own words back.
  reflect: (p) => ({
    json: true,
    maxTokens: 120,
    prompt: `${RULES}

The auditor asked: "${p.question}"
The person answered: "${p.answer}"

Reply with only JSON: {"ack": string, "distress": boolean}
- ack: at most 10 words. Acknowledge by echoing their own words, e.g. "Okay, a heavy gray stone." or "Thank you." No question, no advice. Empty string if the answer is blank or unclear.
- distress: true only if the answer expresses intent to harm themselves or others, or a crisis.`,
  }),
  // Pull a 0-10 number or a yes/no out of an answer the app's own parser couldn't read.
  classify: (p) => ({
    json: true,
    maxTokens: 60,
    prompt: `The auditor asked: "${p.question}"
The person answered: "${p.answer}"
${p.kind === "number"
  ? 'Reply with only JSON: {"value": <integer 0-10 or null>}. "Gone", "nothing", "it disappeared" mean 0. null if no intensity was given.'
  : 'Reply with only JSON: {"value": <true for yes, false for no, null if unclear>}.'}`,
  }),
  // Choose which extra perceptic questions fit what the person has described so far.
  pick: (p) => ({
    json: true,
    maxTokens: 80,
    prompt: `In a somatic-processing session the person has described a feeling like this:
${JSON.stringify(p.described ?? {}).slice(0, 4000)}

Available extra perceptic questions (key: question):
${Object.entries(p.options ?? {}).map(([k, q]) => `${k}: ${q}`).join("\n")}

Pick the ${p.count ?? 3} that make the most sense to ask next, to help the person perceive it more fully.
Reply with only JSON: {"keys": ["key1", "key2", "key3"]}`,
  }),
  // The auditor's observation: do readiness answers read as awake, coherent and engaged?
  observe: (p) => ({
    json: true,
    maxTokens: 40,
    prompt: `These are a person's spoken answers to a readiness checklist before a session (speech-to-text, so allow for small transcription errors):
${JSON.stringify(p.answers ?? []).slice(0, 6000)}

Judge only whether the answers read as reasonably awake, coherent and communicative. Short answers like "yes" or "no" are fine.
Reply with only JSON: {"ok": true} or {"ok": false}`,
  }),
  // End-of-session summary for the log.
  summary: (p) => ({
    json: false,
    maxTokens: 400,
    prompt: `${RULES}

Write a 3-4 sentence plain summary of this session for the person's own log: what was addressed,
how the intensity changed, the counter-intentions canceled, and the positive statements planted.
Use the person's own words. No advice. Session data:
${JSON.stringify(p.log).slice(0, 20000)}`,
  }),
};

function reply(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { ...CORS, "Content-Type": "application/json" } });
}

function parseJson(text: string) {
  const start = text.indexOf("{"), end = text.lastIndexOf("}");
  if (start < 0 || end < start) return null;
  try { return JSON.parse(text.slice(start, end + 1)); } catch { return null; }
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS });
  if (req.method !== "POST") return reply({ error: "POST only" }, 405);

  const admin = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!);
  const jwt = (req.headers.get("Authorization") ?? "").replace(/^Bearer\s+/i, "");
  const { data: { user } } = await admin.auth.getUser(jwt);
  if (!user?.email) return reply({ error: "not_signed_in" }, 401);

  const { data: member } = await admin.from("auditor_family").select("email").eq("email", user.email.toLowerCase()).maybeSingle();
  if (!member) return reply({ error: "not_family" }, 403);

  // The admin can switch AI off for everyone from the app; the app then runs fully scripted.
  const { data: settings } = await admin.from("auditor_settings").select("ai_enabled").eq("id", 1).maybeSingle();
  if (settings && !settings.ai_enabled) return reply({ error: "ai_off" }, 503);

  const { data: usage } = await admin.from("auditor_usage").select("calls")
    .eq("user_id", user.id).eq("day", new Date().toISOString().slice(0, 10)).maybeSingle();
  if ((usage?.calls ?? 0) >= DAILY_CALL_CAP) return reply({ error: "daily_cap" }, 429);

  const apiKey = Deno.env.get("ANTHROPIC_API_KEY");
  if (!apiKey) return reply({ error: "no_api_key" }, 500);

  let body: any;
  try { body = await req.json(); } catch { return reply({ error: "bad_json" }, 400); }
  const task = TASKS[body?.task];
  if (!task) return reply({ error: "unknown_task" }, 400);
  const { prompt, maxTokens, json } = task(body.payload ?? {});

  try {
    const client = new Anthropic({ apiKey });
    const msg = await client.messages.create({
      model: MODEL,
      max_tokens: maxTokens,
      thinking: { type: "disabled" },
      output_config: { effort: "low" },
      messages: [{ role: "user", content: prompt }],
    } as any);
    const text = msg.content.filter((b: any) => b.type === "text").map((b: any) => b.text).join("").trim();
    await admin.rpc("auditor_add_usage", {
      p_user: user.id, p_in: msg.usage.input_tokens, p_out: msg.usage.output_tokens,
    });
    return reply({
      result: json ? parseJson(text) : text,
      usage: { input: msg.usage.input_tokens, output: msg.usage.output_tokens },
    });
  } catch (e) {
    console.error("anthropic call failed", e);
    return reply({ error: "ai_unavailable" }, 502);
  }
});
