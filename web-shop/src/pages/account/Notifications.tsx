import { useEffect, useState } from 'react';
import { Bell, Check, CheckCheck } from 'lucide-react';
import { notificationsApi } from '../../lib/api';

const TYPE_BADGE: Record<string, string> = {
  ORDER: 'bg-blue-100 text-blue-700',
  PRODUCT: 'bg-purple-100 text-purple-700',
  SHOP: 'bg-amber-100 text-amber-700',
  REVIEW: 'bg-pink-100 text-pink-700',
  SYSTEM: 'bg-gray-100 text-gray-700',
};

export default function Notifications() {
  const [items, setItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'unread'>('all');

  const load = async () => {
    setLoading(true);
    try {
      setItems(await notificationsApi.list(filter === 'unread' ? false : undefined));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [filter]);

  const markRead = async (id: number) => {
    await notificationsApi.read(id);
    setItems((l) => l.map((n) => (n.id === id ? { ...n, isRead: true } : n)));
  };

  const markAll = async () => {
    await notificationsApi.readAll();
    setItems((l) => l.map((n) => ({ ...n, isRead: true })));
  };

  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold">Thông báo</h2>
        <button
          onClick={markAll}
          className="text-sm inline-flex items-center gap-1.5 text-gray-600 hover:text-black"
        >
          <CheckCheck className="w-4 h-4" /> Đánh dấu đã đọc tất cả
        </button>
      </div>

      <div className="flex gap-2 mb-4">
        {(['all', 'unread'] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1.5 text-sm rounded-md border ${
              filter === f
                ? 'bg-[var(--color-ink)] text-white border-[var(--color-ink)]'
                : 'border-gray-200 text-gray-700'
            }`}
          >
            {f === 'all' ? 'Tất cả' : 'Chưa đọc'}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="text-gray-500">Đang tải...</div>
      ) : items.length === 0 ? (
        <div className="text-gray-500 border border-dashed border-gray-200 rounded-md p-8 text-center">
          <Bell className="w-10 h-10 mx-auto mb-3 text-gray-300" />
          Không có thông báo.
        </div>
      ) : (
        <div className="space-y-2">
          {items.map((n) => (
            <div
              key={n.id}
              className={`border rounded-md p-3 flex gap-3 ${
                n.isRead ? 'border-gray-200 bg-white' : 'border-[var(--color-accent)]/40 bg-amber-50/30'
              }`}
            >
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <span className={`text-xs px-2 py-0.5 rounded ${TYPE_BADGE[n.type] ?? 'bg-gray-100'}`}>
                    {n.type}
                  </span>
                  <span className="text-xs text-gray-500">
                    {new Date(n.createdAt).toLocaleString('vi-VN')}
                  </span>
                </div>
                <div className="font-medium text-sm">{n.title}</div>
                <div className="text-sm text-gray-600 mt-0.5">{n.content}</div>
              </div>
              {!n.isRead && (
                <button
                  onClick={() => markRead(n.id)}
                  className="text-xs inline-flex items-center gap-1 text-[var(--color-accent)] shrink-0"
                >
                  <Check className="w-3.5 h-3.5" /> Đã đọc
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
