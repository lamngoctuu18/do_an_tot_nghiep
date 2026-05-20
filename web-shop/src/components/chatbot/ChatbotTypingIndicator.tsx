export default function ChatbotTypingIndicator() {
  return (
    <div className="flex items-end gap-2">
      <div className="h-7 w-7 rounded-full bg-black grid place-items-center text-white text-[10px] font-bold">
        AI
      </div>
      <div className="rounded-2xl rounded-bl-sm bg-[#F8F9FA] px-3 py-2 flex items-center gap-1">
        <span className="h-1.5 w-1.5 rounded-full bg-[var(--color-ink-muted)] animate-bounce [animation-delay:-0.3s]" />
        <span className="h-1.5 w-1.5 rounded-full bg-[var(--color-ink-muted)] animate-bounce [animation-delay:-0.15s]" />
        <span className="h-1.5 w-1.5 rounded-full bg-[var(--color-ink-muted)] animate-bounce" />
      </div>
    </div>
  );
}
