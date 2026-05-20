import type { ChatOrderInfo } from "./types";

const fmt = (n: number) => new Intl.NumberFormat("vi-VN").format(n) + "₫";

const STATUS_COLORS: Record<string, string> = {
  PENDING: "bg-amber-100 text-amber-700",
  CONFIRMED: "bg-blue-100 text-blue-700",
  SHIPPING: "bg-indigo-100 text-indigo-700",
  COMPLETED: "bg-emerald-100 text-emerald-700",
  CANCELLED: "bg-rose-100 text-rose-700",
};

export default function ChatbotOrderCard({ order }: { order: ChatOrderInfo }) {
  const color = STATUS_COLORS[order.status] || "bg-gray-100 text-gray-700";
  return (
    <div className="rounded-xl border border-[var(--color-line)] bg-white p-3">
      <div className="flex items-center justify-between">
        <div className="text-[12px] font-bold text-[var(--color-ink)]">#{order.code}</div>
        <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${color}`}>
          {order.status}
        </span>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-[var(--color-ink-muted)]">
        <div>Số SP: <span className="text-[var(--color-ink)] font-medium">{order.itemsCount}</span></div>
        <div>Tổng: <span className="text-[var(--color-ink)] font-medium">{fmt(order.total)}</span></div>
        <div className="col-span-2">Ngày đặt: <span className="text-[var(--color-ink)] font-medium">{order.createdAt}</span></div>
      </div>
    </div>
  );
}
