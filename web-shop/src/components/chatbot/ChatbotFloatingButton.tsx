import { useChatbot } from "./ChatbotContext";

export default function ChatbotFloatingButton() {
  const { open, minimized, toggle, restore, unread } = useChatbot();
  if (open && !minimized) return null;

  const handleClick = () => {
    if (open && minimized) restore();
    else toggle();
  };

  return (
    <button
      type="button"
      aria-label={`Mở trợ lý thời trang AI${unread ? `, ${unread} tin nhắn mới` : ""}`}
      aria-expanded={open}
      onClick={handleClick}
      className="fixed bottom-6 right-6 z-[60] group focus:outline-none focus-visible:ring-4 focus-visible:ring-[#D6B98C]/40 rounded-full"
    >
      <span className="absolute inset-0 rounded-full bg-black/30 animate-ping opacity-60" aria-hidden />
      <span className="relative flex h-14 w-14 items-center justify-center rounded-full bg-black text-white shadow-[0_12px_28px_rgba(0,0,0,0.28)] transition-transform hover:scale-105 active:scale-95">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className="h-6 w-6" aria-hidden>
          <path strokeLinecap="round" strokeLinejoin="round" d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
        </svg>
        <span className="absolute -bottom-0.5 -right-0.5 h-3.5 w-3.5 rounded-full border-2 border-white bg-emerald-500" aria-hidden />
        {unread > 0 && (
          <span
            className="absolute -top-1 -right-1 min-w-[20px] h-5 px-1 rounded-full bg-[#D6B98C] text-[11px] font-bold text-black grid place-items-center"
            aria-hidden
          >
            {unread > 9 ? "9+" : unread}
          </span>
        )}
      </span>
      <span
        role="tooltip"
        className="pointer-events-none absolute right-16 top-1/2 -translate-y-1/2 whitespace-nowrap rounded-lg bg-black px-3 py-1.5 text-xs font-medium text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100"
      >
        Trợ lý thời trang AI
      </span>
    </button>
  );
}
