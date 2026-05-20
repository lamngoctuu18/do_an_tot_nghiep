import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ordersApi } from '../../lib/api';

const STATUS_LABEL: Record<string, string> = {
  PENDING: 'Chờ xác nhận',
  CONFIRMED: 'Đã xác nhận',
  PACKING: 'Đang đóng gói',
  SHIPPING: 'Đang giao',
  COMPLETED: 'Hoàn thành',
  CANCELLED: 'Đã huỷ',
};

const STATUS_COLOR: Record<string, string> = {
  PENDING: 'bg-amber-100 text-amber-700',
  CONFIRMED: 'bg-blue-100 text-blue-700',
  PACKING: 'bg-indigo-100 text-indigo-700',
  SHIPPING: 'bg-sky-100 text-sky-700',
  COMPLETED: 'bg-green-100 text-green-700',
  CANCELLED: 'bg-red-100 text-red-700',
};

const fmt = (v: any) => new Intl.NumberFormat('vi-VN').format(+v) + 'đ';

export default function Orders() {
  const [list, setList] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    ordersApi.list().then(setList).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-gray-500">Đang tải...</div>;

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Đơn hàng của tôi</h2>
      {list.length === 0 ? (
        <div className="text-gray-500 border border-dashed border-gray-200 rounded-md p-8 text-center">
          Chưa có đơn hàng nào.
        </div>
      ) : (
        <div className="space-y-3">
          {list.map((o) => (
            <Link
              key={o.id}
              to={`/account/orders/${o.code}`}
              className="block border border-gray-200 rounded-md p-4 hover:border-[var(--color-accent)] transition-colors"
            >
              <div className="flex justify-between items-start mb-2">
                <div>
                  <div className="font-semibold">#{o.code}</div>
                  <div className="text-xs text-gray-500">
                    {new Date(o.placedAt).toLocaleString('vi-VN')}
                  </div>
                </div>
                <span className={`text-xs px-2 py-1 rounded ${STATUS_COLOR[o.status] ?? 'bg-gray-100'}`}>
                  {STATUS_LABEL[o.status] ?? o.status}
                </span>
              </div>
              <div className="text-sm text-gray-600">
                {o.items?.length ?? 0} sản phẩm · Tổng {fmt(o.total)}
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
