import { Link } from "react-router-dom";
import type { ChatProductSuggestion } from "./types";

const fmt = (n: number) => new Intl.NumberFormat("vi-VN").format(n) + "₫";

export default function ChatbotProductCard({ product }: { product: ChatProductSuggestion }) {
  return (
    <div className="flex gap-3 rounded-xl border border-[var(--color-line)] bg-white p-2.5 hover:border-[var(--color-accent)] transition-colors">
      <img
        src={product.image}
        alt={product.name}
        className="h-16 w-16 rounded-lg object-cover flex-shrink-0"
      />
      <div className="flex-1 min-w-0 flex flex-col justify-between">
        <div>
          <div className="text-[13px] font-semibold text-[var(--color-ink)] line-clamp-1">
            {product.name}
          </div>
          {product.reason && (
            <div className="text-[11px] text-[var(--color-ink-muted)] line-clamp-1">
              {product.reason}
            </div>
          )}
          <div className="text-[13px] font-bold text-[var(--color-ink)] mt-0.5">
            {fmt(product.price)}
          </div>
        </div>
        <div className="flex gap-1.5 mt-1">
          <Link
            to={`/product/${product.id}`}
            className="flex-1 rounded-md border border-[var(--color-line)] px-2 py-1 text-center text-[11px] font-medium hover:border-[var(--color-accent)] hover:bg-[var(--color-bg-warm)]"
          >
            Xem chi tiết
          </Link>
          <Link
            to={`/try-on/${product.id}`}
            className="flex-1 rounded-md bg-black px-2 py-1 text-center text-[11px] font-medium text-white hover:bg-[var(--color-primary-hover)]"
          >
            Thử đồ
          </Link>
        </div>
      </div>
    </div>
  );
}
