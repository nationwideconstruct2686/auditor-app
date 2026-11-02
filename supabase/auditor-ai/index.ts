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

const RULES = `You assist a scripted self-processing session. The script asks every question;
you never ask questions, give advice, interpret, diagnose, or add your own ideas.
Use plain, calm, warm language. Never mention being an AI.`;

const PROCESS = `How the process works: the person locates a somatic, negative emotion or limiting feeling, rates its intensity,
describes its perceptics (smell, color, texture, shape, location and so on), personifies it, contacts it, then finds
what it is saying (the counter-intention). They say the counter-intention 10 times, say "cancel, cancel", choose a
positive statement or word in alignment with absolute truth and their infinite potential, say it 10 times, and repeat
until the intensity is zero. Deep processing also draws it up by describing situations, and processes any memory that
comes up: go to the beginning of the memory, perceptics, go through it to the end, and repeat until the charge is zero.
Experience processing works on a recent suboptimal experience with no somatic, emotion or feeling attached: the person
names it and the ideal outcome, recalls its percepts (where, what they saw, smelled, heard, felt), then finds the
limiting thought acting as a counter-intention to a more optimal experience, and repeats until nothing about it is suboptimal.`;

// Shared by every task that hears the person's reply: spot a question to the app and answer it fully.
const questionCheck = (p: any) => `You are the voice of a guided self-processing app (session type: ${p.mode ?? "processing"}).
The app just asked: "${p.question}"
The person replied: "${p.answer}"
(This is speech-to-text, so allow for small transcription errors.)

${PROCESS}

The app owner's own words on the method:
${(p.concepts ?? []).join(" ").slice(0, 4000)}

First decide whether the reply is a question or request directed at the app rather than an answer to the question asked:
for example asking about the process, a step or a word, saying they don't understand, asking to repeat, asking how to
answer, or asking anything else, even outside the process.
If it is, set is_question true and write reply: a complete answer to what they asked, in at most 4 short sentences to be
spoken aloud. Answer it fully and directly; never just thank them. Never give example answers, never interpret what the
person's experience means, never give advice, diagnosis, or medical or psychological guidance, and never suggest what their
answer should be. If the question is outside the process, say kindly that it's outside what this session covers. If they
asked to repeat, reply "Of course." Don't ask a question; the app repeats its own question after you. Never mention being an AI.${inSpanish(p)}`;

// The app can run in Spanish; then everything the person hears or reads comes back in Spanish.
function inSpanish(p: any) {
  return p.lang === "es"
    ? `\nThe person is using the app in Spanish. Write reply, ack and any other text in natural Latin American Spanish, using tú. JSON keys stay in English.`
    : "";
}

const TASKS: Record<string, (p: any) => { prompt: string; maxTokens: number; json: boolean }> = {
  // Either answers a question the person asked, or gives a short acknowledgment that echoes their words.
  reflect: (p) => ({
    json: true,
    maxTokens: 300,
    prompt: `${questionCheck(p)}

Otherwise set is_question false and write ack: at most 10 words acknowledging by echoing their own words,
e.g. "Okay, a heavy gray stone." No question, no advice. Empty string if the answer is blank or unclear.
Reply with only JSON: {"is_question": boolean, "reply": string, "ack": string, "distress": boolean}
- distress: true only if the reply expresses intent to harm themselves or others, or a crisis.`,
  }),
  // Pull a 0-10 number or a yes/no out of an answer the app's own parser couldn't read, unless it was a question.
  classify: (p) => ({
    json: true,
    maxTokens: 300,
    prompt: `${questionCheck(p)}

Otherwise set is_question false and read the value from their reply.
${p.kind === "number"
  ? 'value: an integer 0-10, or null if no intensity was given. "Gone", "nothing", "it disappeared" mean 0.'
  : "value: true for yes, false for no, null if unclear."}
Reply with only JSON: {"is_question": boolean, "reply": string, "value": ${p.kind === "number" ? "number|null" : "boolean|null"}, "distress": boolean}
- distress: true only if the reply expresses intent to harm themselves or others, or a crisis.`,
  }),
  // A reply that looks like a question, on a step with no echo-back.
  answer: (p) => ({
    json: true,
    maxTokens: 300,
    prompt: `${questionCheck(p)}

Otherwise set is_question false and reply "".
Reply with only JSON: {"is_question": boolean, "reply": string, "distress": boolean}
- distress: true only if the reply expresses intent to harm themselves or others, or a crisis.`,
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
  // End-of-session summary for the log.
  summary: (p) => ({
    json: false,
    maxTokens: 400,
    prompt: `${RULES}

Write a 3-4 sentence plain summary of this session for the person's own log: what was addressed,
how the intensity changed, the counter-intentions canceled, and the positive statements planted.
Use the person's own words. No advice.${inSpanish(p)} Session data:
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
