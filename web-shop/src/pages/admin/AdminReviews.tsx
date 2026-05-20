import { useEffect, useState } from 'react';
import { EyeOff, Eye, Star } from 'lucide-react';
import { adminApi, ApiError } from '../../lib/api';

const STATUS_FILTERS = [
  { value: '', label: 'Tất cả' },
  { value: 'VISIBLE', label: 'Hiển thị' },
  { value: 'HIDDEN', label: 'Đã ẩn' },
];

const formatDate = (s: string) =>
  new Date(s).toLocaleString('vi-VN', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

export default function AdminReviews() {
  const [items, setItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState('');

  const load = async (status: string) => {
    setLoading(true);
    try {
      setItems(await adminApi.reviews(status || undefined));
    } catch (e: any) {
      alert(e instanceof ApiError ? e.message : 'Lỗi tải');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load(statusFilter);
  }, [statusFilter]);

  const toggle = async (r: any) => {
    try {
      if (r.status === 'HIDDEN') await adminApi.unhideReview(r.id);
      else await adminApi.hideReview(r.id);
      await load(statusFilter);
    } catch (e: any) {
      alert(e instanceof ApiError ? e.message : 'Lỗi');
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Đánh giá sản phẩm</h2>
        <div className="flex gap-1.5">
          {STATUS_FILTERS.map((s) => (
            <button
              key={s.value}
              onClick={() => setStatusFilter(s.value)}
              className={`text-xs px-3 py-1.5 rounded border ${
                statusFilter === s.value
                  ? 'bg-[var(--color-ink)] text-white border-[var(--color-ink)]'
                  : 'border-gray-200 hover:bg-gray-50'
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="text-gray-500">Đang tải...</div>
      ) : items.length === 0 ? (
        <div className="text-gray-500 border border-dashed border-gray-200 rounded-md p-10 text-center">
          Không có đánh giá nào.
        </div>
      ) : (
        <ul className="space-y-3">
          {items.map((r) => (
            <li
              key={r.id}
              className="border border-gray-200 rounded-md p-3 flex flex-col gap-2 md:flex-row md:items-start md:justify-between"
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1 text-sm">
                  <span className="font-medium">
                    {r.user?.fullName ?? r.user?.email ?? 'Khách'}
                  </span>
                  <span className="text-gray-400 text-xs">{formatDate(r.createdAt)}</span>
                  <span
                    className={`text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded ${
                      r.status === 'HIDDEN' ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                    }`}
                  >
                    {r.status}
                  </span>
                </div>
                <div className="text-xs text-gray-500 mb-1.5">
                  Sản phẩm: <span className="font-medium">{r.product?.name ?? '-'}</span>
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
                {r.comment && (
                  <p className="text-sm text-gray-700 whitespace-pre-line">{r.comment}</p>
                )}
              </div>
              <button
                onClick={() => toggle(r)}
                className="inline-flex items-center gap-1 text-xs px-2.5 py-1.5 border border-gray-200 rounded hover:bg-gray-50 self-start whitespace-nowrap"
              >
                {r.status === 'HIDDEN' ? (
                  <>
                    <Eye className="w-3.5 h-3.5" /> Hiện lại
                  </>
                ) : (
                  <>
                    <EyeOff className="w-3.5 h-3.5" /> Ẩn đánh giá
                  </>
                )}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
