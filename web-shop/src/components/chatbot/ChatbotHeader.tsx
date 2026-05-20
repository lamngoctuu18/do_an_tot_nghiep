import { useChatbot } from "./ChatbotContext";

export default function ChatbotHeader() {
  const { setOpen, minimize, clearHistory } = useChatbot();
  return (
    <div className="flex items-center gap-3 border-b border-[var(--color-line)] bg-white px-4 py-3">
      <div className="relative">
        <div className="h-10 w-10 rounded-full bg-black grid place-items-center text-white font-bold" aria-hidden>
          AI
        </div>
        <span className="absolute -bottom-0.5 -right-0.5 h-3 w-3 rounded-full border-2 border-white bg-emerald-500" aria-hidden />
      </div>
      <div className="flex-1 min-w-0">
        <div className="font-display text-[15px] font-semibold text-[var(--color-ink)] leading-tight">
          Trợ lý thời trang AI
        </div>
        <div className="text-[11px] text-[var(--color-ink-muted)] flex items-center gap-1">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" aria-hidden />
          Đang hoạt động
        </div>
      </div>
      <button
        type="button"
        aria-label="Xoá lịch sử trò chuyện"
        title="Xoá lịch sử"
        onClick={() => {
          if (confirm("Xoá toàn bộ lịch sử trò chuyện?")) clearHistory();
        }}
        className="h-8 w-8 rounded-lg text-[var(--color-ink-muted)] hover:bg-[var(--color-bg-soft)] grid place-items-center focus:outline-none focus-visible:ring-2 focus-visible:ring-[#D6B98C]"
      >
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
          <path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6h14z" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      <button
        type="button"
        aria-label="Thu nhỏ"
        title="Thu nhỏ"
        onClick={minimize}
        className="h-8 w-8 rounded-lg text-[var(--color-ink-muted)] hover:bg-[var(--color-bg-soft)] grid place-items-center focus:outline-none focus-visible:ring-2 focus-visible:ring-[#D6B98C]"
      >
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
          <path d="M5 12h14" strokeLinecap="round" />
        </svg>
      </button>
      <button
        type="button"
        aria-label="Đóng"
        title="Đóng"
        onClick={() => setOpen(false)}
        className="h-8 w-8 rounded-lg text-[var(--color-ink-muted)] hover:bg-[var(--color-bg-soft)] grid place-items-center focus:outline-none focus-visible:ring-2 focus-visible:ring-[#D6B98C]"
      >
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
          <path d="M6 6l12 12M18 6l-12 12" strokeLinecap="round" />
        </svg>
      </button>
    </div>
  );
}
