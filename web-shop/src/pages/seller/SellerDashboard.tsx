import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Sparkles, AlertTriangle } from 'lucide-react';
import { sellerApi, ApiError } from '../../lib/api';

const fmt = (v: any) => new Intl.NumberFormat('vi-VN').format(+v) + 'đ';

function Stat({ label, value, accent }: { label: string; value: any; accent?: boolean }) {
  return (
    <div
      className={`p-4 rounded-lg border ${
        accent ? 'border-[var(--color-accent)] bg-amber-50/30' : 'border-gray-200'
      }`}
    >
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className="text-2xl font-semibold">{value}</div>
    </div>
  );
}

export default function SellerDashboard() {
  const [data, setData] = useState<any>(null);
  const [tryonTop, setTryonTop] = useState<any[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      sellerApi.dashboard().catch((e) => {
        setErr(e instanceof ApiError ? e.message : 'Lỗi tải dashboard');
        return null;
      }),
      sellerApi.tryonStats().catch(() => []),
    ])
      .then(([d, t]) => {
        setData(d);
        setTryonTop((t ?? []).slice(0, 5));
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-gray-500">Đang tải...</div>;
  if (err && !data) return <div className="text-red-600 text-sm">{err}</div>;
  if (!data) return null;

  const lowStock = data.lowStockVariants ?? 0;

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Tổng quan</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="Chờ xác nhận" value={data.pendingOrders} accent />
        <Stat label="Đã xác nhận" value={data.confirmedOrders} />
        <Stat label="Đang giao" value={data.shippingOrders} />
        <Stat label="Hoàn thành" value={data.completedOrders} />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-3">
        <Stat label="Doanh thu hôm nay" value={fmt(data.todayRevenue)} />
        <Stat label="Doanh thu tháng" value={fmt(data.monthRevenue)} />
        <Stat label="Variant sắp hết hàng" value={lowStock} accent />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-6">
        <section className="border border-gray-200 rounded-lg overflow-hidden">
          <div className="flex items-center justify-between bg-gray-50 px-4 py-2.5 text-sm">
            <div className="inline-flex items-center gap-1.5 font-semibold">
              <AlertTriangle className="w-4 h-4 text-amber-600" /> Sắp hết hàng
            </div>
            <Link to="/seller/products" className="text-xs text-gray-500 hover:underline">
              Quản lý sản phẩm →
            </Link>
          </div>
          <div className="p-4 text-sm">
            {lowStock === 0 ? (
              <div className="text-gray-500">Không có variant nào sắp hết.</div>
            ) : (
              <div className="text-gray-700">
                Có <strong>{lowStock}</strong> variant tồn kho ≤ 5. Vui lòng kiểm tra để nhập thêm
                hàng.
              </div>
            )}
          </div>
        </section>

        <section className="border border-gray-200 rounded-lg overflow-hidden">
          <div className="flex items-center justify-between bg-gray-50 px-4 py-2.5 text-sm">
            <div className="inline-flex items-center gap-1.5 font-semibold">
              <Sparkles className="w-4 h-4 text-[var(--color-accent)]" /> Top thử đồ ảo
            </div>
            <Link to="/seller/tryon-stats" className="text-xs text-gray-500 hover:underline">
              Xem tất cả →
            </Link>
          </div>
          {tryonTop.length === 0 ? (
            <div className="p-4 text-sm text-gray-500">Chưa có dữ liệu thử đồ.</div>
          ) : (
            <ul className="divide-y divide-gray-100">
              {tryonTop.map((t, idx) => (
                <li key={t.productId} className="flex items-center gap-3 px-4 py-2.5 text-sm">
                  <span className="w-6 h-6 inline-flex items-center justify-center rounded-full bg-[var(--color-bg-soft)] text-xs font-semibold">
                    {idx + 1}
                  </span>
                  <span className="flex-1 truncate">{t.productName}</span>
                  <span className="text-xs text-gray-500">{t.tryonCount} lượt</span>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </div>
  );
}
