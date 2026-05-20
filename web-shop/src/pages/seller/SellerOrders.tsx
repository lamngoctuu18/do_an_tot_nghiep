import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { sellerApi, ApiError } from '../../lib/api';

const STATUS_LABEL: Record<string, string> = {
  PENDING: 'Chờ xác nhận',
  CONFIRMED: 'Đã xác nhận',
  PACKING: 'Đang đóng gói',
  SHIPPING: 'Đang giao',
  COMPLETED: 'Hoàn thành',
  CANCELLED: 'Đã huỷ',
};

const NEXT: Record<string, string[]> = {
  PENDING: [],
  CONFIRMED: ['PACKING', 'CANCELLED'],
  PACKING: ['SHIPPING'],
  SHIPPING: ['COMPLETED'],
};

const fmt = (v: any) => new Intl.NumberFormat('vi-VN').format(+v) + 'đ';

export default function SellerOrders() {
  const [items, setItems] = useState<any[]>([]);
  const [status, setStatus] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setErr(null);
    try {
      const data = await sellerApi.orders({ status: status || undefined });
      setItems(data.items);
    } catch (e: any) {
      setErr(e instanceof ApiError ? e.message : 'Lỗi tải đơn hàng');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [status]);

  const act = async (code: string, action: 'confirm' | string) => {
    try {
      if (action === 'confirm') await sellerApi.confirm(code);
      else await sellerApi.updateStatus(code, action);
      await load();
    } catch (e: any) {
      alert(e instanceof ApiError ? e.message : 'Lỗi cập nhật');
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Đơn hàng</h2>
        <select
          value={status}
          onChange={(e) => setStatus(e.target.value)}
          className="border border-gray-200 rounded-md px-3 py-2 text-sm bg-white"
        >
          <option value="">Tất cả</option>
          {Object.entries(STATUS_LABEL).map(([k, v]) => (
            <option key={k} value={k}>
              {v}
            </option>
          ))}
        </select>
      </div>

      {err && <div className="text-red-600 text-sm mb-3">{err}</div>}
      {loading ? (
        <div className="text-gray-500">Đang tải...</div>
      ) : items.length === 0 ? (
        <div className="text-gray-500 border border-dashed border-gray-200 rounded-md p-8 text-center">
          Không có đơn hàng.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-left">
              <tr>
                <th className="p-2.5">Mã đơn</th>
                <th className="p-2.5">Khách hàng</th>
                <th className="p-2.5">Tổng</th>
                <th className="p-2.5">Trạng thái</th>
                <th className="p-2.5">Hành động</th>
              </tr>
            </thead>
            <tbody>
              {items.map((o) => (
                <tr key={o.id} className="border-t border-gray-100">
                  <td className="p-2.5 font-medium">
                    <Link to={`/seller/orders/${o.code}`} className="hover:underline">
                      #{o.code}
                    </Link>
                  </td>
                  <td className="p-2.5">{o.user?.fullName ?? '-'}</td>
                  <td className="p-2.5">{fmt(o.total)}</td>
                  <td className="p-2.5">{STATUS_LABEL[o.status] ?? o.status}</td>
                  <td className="p-2.5">
                    <div className="flex flex-wrap gap-1.5">
                      {o.status === 'PENDING' && (
                        <button
                          onClick={() => act(o.code, 'confirm')}
                          className="px-2 py-1 bg-[var(--color-ink)] text-white rounded text-xs"
                        >
                          Xác nhận
                        </button>
                      )}
                      {NEXT[o.status]?.map((next) => (
                        <button
                          key={next}
                          onClick={() => act(o.code, next)}
                          className="px-2 py-1 border border-gray-200 rounded text-xs hover:bg-gray-50"
                        >
                          → {STATUS_LABEL[next]}
                        </button>
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
