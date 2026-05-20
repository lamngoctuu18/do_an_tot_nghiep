import { useEffect, useState } from 'react';
import { Star } from 'lucide-react';
import { reviewsApi, ApiError } from '../lib/api';
import { useAuth } from '../context/AuthContext';

const formatDate = (s: string) =>
  new Date(s).toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric' });

export default function ProductReviews({ productId }: { productId: number }) {
  const { user } = useAuth();
  const [items, setItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [rating, setRating] = useState(5);
  const [comment, setComment] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [ok, setOk] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    reviewsApi
      .list(productId)
      .then(setItems)
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (productId) load();
  }, [productId]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    setOk(null);
    setSubmitting(true);
    try {
      await reviewsApi.create(productId, { rating, comment: comment.trim() || undefined });
      setOk('Cảm ơn bạn đã đánh giá!');
      setComment('');
      setRating(5);
      load();
    } catch (e: any) {
      setErr(e instanceof ApiError ? e.message : 'Lỗi gửi đánh giá');
    } finally {
      setSubmitting(false);
    }
  };

  const visible = items.filter((r) => r.status !== 'HIDDEN');

  return (
    <div className="space-y-6">
      {user ? (
        <form onSubmit={submit} className="border border-gray-200 rounded-lg p-4 space-y-3">
          <div className="font-medium text-sm">Viết đánh giá của bạn</div>
          <div className="flex items-center gap-1">
            {[1, 2, 3, 4, 5].map((n) => (
              <button
                key={n}
                type="button"
                onClick={() => setRating(n)}
                aria-label={`${n} sao`}
                className="p-0.5"
              >
                <Star
                  className={`w-6 h-6 ${
                    n <= rating
                      ? 'fill-[var(--color-warning)] text-[var(--color-warning)]'
                      : 'text-gray-300'
                  }`}
                />
              </button>
            ))}
            <span className="ml-2 text-sm text-gray-500">{rating}/5</span>
          </div>
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Chia sẻ cảm nhận của bạn về sản phẩm..."
            rows={3}
            className="input resize-none"
          />
          {err && <div className="text-sm text-red-600">{err}</div>}
          {ok && <div className="text-sm text-green-600">{ok}</div>}
          <div className="flex justify-end">
            <button
              type="submit"
              disabled={submitting}
              className="px-4 py-2 bg-[var(--color-ink)] text-white text-sm rounded hover:opacity-90 disabled:opacity-50"
            >
              {submitting ? 'Đang gửi...' : 'Gửi đánh giá'}
            </button>
          </div>
        </form>
      ) : (
        <div className="text-sm text-gray-500 border border-dashed border-gray-200 rounded p-3">
          Đăng nhập để viết đánh giá.
        </div>
      )}

      {loading ? (
        <div className="text-sm text-gray-500">Đang tải đánh giá...</div>
      ) : visible.length === 0 ? (
        <div className="text-sm text-gray-500">Chưa có đánh giá nào cho sản phẩm này.</div>
      ) : (
        <ul className="space-y-4">
          {visible.map((r) => (
            <li key={r.id} className="border-b border-gray-100 pb-4 last:border-b-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="font-medium text-sm">
                  {r.user?.fullName ?? r.user?.email ?? 'Khách'}
                </span>
                <span className="text-xs text-gray-400">{formatDate(r.createdAt)}</span>
              </div>
              <div className="flex items-center gap-0.5 mb-1.5">
                {[1, 2, 3, 4, 5].map((n) => (
                  <Star
                    key={n}
                    className={`w-3.5 h-3.5 ${
                      n <= r.rating
                        ? 'fill-[var(--color-warning)] text-[var(--color-warning)]'
                        : 'text-gray-200'
                    }`}
                  />
                ))}
              </div>
              {r.comment && <p className="text-sm text-gray-700 whitespace-pre-line">{r.comment}</p>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
