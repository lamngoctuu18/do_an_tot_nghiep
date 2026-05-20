import { useState, type FormEvent } from "react";
import { useChatbot } from "./ChatbotContext";

export default function ChatbotInput() {
  const { sendUser, typing } = useChatbot();
  const [text, setText] = useState("");

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!text.trim() || typing) return;
    sendUser(text);
    setText("");
  };

  return (
    <form
      onSubmit={submit}
      className="flex items-center gap-2 border-t border-[var(--color-line)] bg-white px-3 py-2.5"
    >
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Nhập tin nhắn..."
        className="flex-1 rounded-full border border-[var(--color-line)] bg-[var(--color-bg-soft)] px-4 py-2 text-[13px] text-[var(--color-ink)] placeholder:text-[var(--color-ink-disabled)] outline-none focus:border-[var(--color-accent)] focus:bg-white"
      />
      <button
        type="submit"
        disabled={!text.trim() || typing}
        aria-label="Gửi"
        className="h-9 w-9 flex-shrink-0 rounded-full bg-black text-white grid place-items-center disabled:opacity-40 disabled:cursor-not-allowed hover:bg-[var(--color-primary-hover)] active:scale-95 transition"
      >
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M22 2L11 13" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M22 2l-7 20-4-9-9-4 20-7z" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
    </form>
  );
}
