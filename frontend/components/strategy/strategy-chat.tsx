"use client";

import { useEffect, useState } from "react";
import { ChevronDown, Loader2 } from "lucide-react";
import { fetchChatStatus, sendChat, type ChatMessage, type ChatStatus, type ChatToolCall } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface Turn extends ChatMessage {
  tools?: ChatToolCall[];
  caveats?: string[];
  failed?: boolean;
}

/** T-058: local-LLM (Ollama) chat. Optional: without Ollama it shows setup steps instead of the input. */
export function StrategyChat({ trackId }: { trackId: string }) {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<ChatStatus | null | "error">(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let live = true;
    fetchChatStatus()
      .then((s) => live && setStatus(s))
      .catch(() => live && setStatus("error"));
    return () => {
      live = false;
    };
  }, []);

  async function send() {
    const content = text.trim();
    if (!content || busy) return;
    const history = [...turns.filter((t) => !t.failed), { role: "user" as const, content }];
    setTurns([...turns, { role: "user", content }]);
    setText("");
    setBusy(true);
    try {
      const r = await sendChat(
        history.map(({ role, content: c }) => ({ role, content: c })),
        trackId ? { track_id: trackId } : undefined,
      );
      setTurns((t) => [...t, { role: "assistant", content: r.reply, tools: r.tool_calls, caveats: r.caveats }]);
    } catch (e) {
      setTurns((t) => [...t, { role: "assistant", content: e instanceof Error ? e.message : "Chat failed.", failed: true }]);
    } finally {
      setBusy(false);
    }
  }

  const available = status !== null && status !== "error" && status.available;

  return (
    <section aria-labelledby="chat-h" className="space-y-3">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex items-center gap-2 text-left"
      >
        <ChevronDown className={`size-4 text-muted-fg transition-transform ${open ? "" : "-rotate-90"}`} />
        <h2 id="chat-h" className="display text-3xl text-fg">
          Ask about this strategy
        </h2>
      </button>

      {open && (
        <div className="panel space-y-3 p-4">
          {status === null && <p className="text-[13px] text-muted-fg">Checking for a local model...</p>}
          {status === "error" && (
            <p className="text-[13px] text-muted-fg">Could not reach the backend to check for a local model.</p>
          )}
          {status !== null && status !== "error" && !available && (
            <p className="text-[13px] leading-5 text-muted-fg">
              The chat runs on a local model and is not available here. To enable it on your machine: install Ollama
              (ollama.com), then run <code className="font-mono">ollama pull {status.model}</code>, and reload.
              {status.message ? ` (${status.message})` : ""}
            </p>
          )}

          {available && (
            <>
              <p className="text-[13px] text-muted-fg">
                Answers come from the simulator&apos;s own tools{trackId ? ` (track: ${trackId})` : ""}. A small local
                model can be slow and can be wrong, so check the numbers.
              </p>
              <div className="space-y-3" aria-live="polite">
                {turns.map((t, i) => (
                  <div key={i} className={t.role === "user" ? "text-right" : ""}>
                    <p
                      className={`inline-block max-w-[90%] whitespace-pre-wrap rounded-lg px-3 py-2 text-left text-[14px] ${
                        t.role === "user" ? "bg-sunken text-fg" : t.failed ? "text-bad" : "text-fg"
                      }`}
                    >
                      {t.content}
                    </p>
                    {t.tools && t.tools.length > 0 && (
                      <p className="mt-1 font-mono text-xs text-muted-fg">
                        used tools: {t.tools.map((c) => c.name).join(", ")}
                      </p>
                    )}
                    {t.caveats && t.caveats.length > 0 && (
                      <ul className="mt-1 list-disc space-y-0.5 pl-5 text-xs text-muted-fg">
                        {t.caveats.map((c) => (
                          <li key={c}>{c}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
                {busy && (
                  <p className="flex items-center gap-2 text-[13px] text-muted-fg">
                    <Loader2 className="size-4 animate-spin" /> Thinking (a local model can take a minute)...
                  </p>
                )}
              </div>
              <form
                className="flex gap-2"
                onSubmit={(e) => {
                  e.preventDefault();
                  void send();
                }}
              >
                <Input
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="e.g. Why is a 1-stop best here?"
                  aria-label="Ask about this strategy"
                  className="font-sans"
                  disabled={busy}
                />
                <Button type="submit" disabled={busy || !text.trim()}>
                  Ask
                </Button>
              </form>
            </>
          )}
        </div>
      )}
    </section>
  );
}
