import { useEffect, useState } from 'react';
import { Sparkles, CheckCircle2, XCircle, Activity } from 'lucide-react';
import { adminApi } from '../../lib/api';

const fmt = (v: any) => new Intl.NumberFormat('vi-VN').format(+v) + 'đ';
const fmtNum = (v: any) => new Intl.NumberFormat('vi-VN').format(+v);
const todayStr = () => new Date().toISOString().slice(0, 10);
const daysAgoStr = (n: number) => {
  const d = new Date();
  d.setDate(d.getDate() - n);
  return d.toISOString().slice(0, 10);
};

export default function AdminReports() {
  const [from, setFrom] = useState(daysAgoStr(29));
  const [to, setTo] = useState(todayStr());
  const [revenue, setRevenue] = useState<any[]>([]);
  const [topProducts, setTopProducts] = useState<any[]>([]);
  const [tryon, setTryon] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const loadAll = () => {
    setLoading(true);
    Promise.all([
      adminApi.reportRevenue(from, to).catch(() => []),
      adminApi.topProducts(10).catch(() => []),
      adminApi.tryonReport().catch(() => null),
    ]).then(([rev, top, ty]) => {
      setRevenue(rev as any[]);
      setTopProducts(top as any[]);
      setTryon(ty);
      setLoading(false);
    });
  };

  useEffect(() => {
    loadAll();
  }, []);

  const totalRev = revenue.reduce((s, r) => s + +(r.revenue ?? r.total ?? 0), 0);
  const totalOrders = revenue.reduce((s, r) => s + +(r.orders ?? r.count ?? 0), 0);
  const successRate = tryon?.total
    ? ((+tryon.success / +tryon.total) * 100).toFixed(1)
    : '0';

  return (
    <div className="space-y-6">
      <section>
        <div className="flex items-end justify-between gap-3 flex-wrap mb-3">
          <h2 className="text-lg font-semibold">Doanh thu theo ngày</h2>
          <div className="flex items-end gap-2">
            <label className="text-xs">
              Từ
              <input
                type="date"
                value={from}
                onChange={(e) => setFrom(e.target.value)}
                className="block border border-gray-200 rounded px-2 py-1 text-sm"
              />
            </label>
            <label className="text-xs">
              Đến
              <input
                type="date"
                value={to}
                onChange={(e) => setTo(e.target.value)}
                className="block border border-gray-200 rounded px-2 py-1 text-sm"
              />
            </label>
            <button
              onClick={loadAll}
              className="px-3 py-1.5 bg-[var(--color-ink)] text-white rounded text-xs"
            >
              Lọc
            </button>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3 mb-3">
          <div className="border border-gray-200 rounded-md p-3">
            <div className="text-xs text-gray-500">Tổng doanh thu</div>
            <div className="text-xl font-semibold">{fmt(totalRev)}</div>
          </div>
          <div className="border border-gray-200 rounded-md p-3">
            <div className="text-xs text-gray-500">Tổng đơn</div>
            <div className="text-xl font-semibold">{fmtNum(totalOrders)}</div>
          </div>
        </div>
        {loading ? (
          <div className="text-gray-500 text-sm">Đang tải...</div>
        ) : revenue.length === 0 ? (
          <div className="text-gray-500 text-sm border border-dashed border-gray-200 rounded p-6 text-center">
            Chưa có dữ liệu trong khoảng thời gian này.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm border border-gray-200 rounded-md min-w-[480px]">
              <thead className="bg-gray-50 text-left">
                <tr>
                  <th className="p-2.5">Ngày</th>
                  <th className="p-2.5 text-right">Số đơn</th>
                  <th className="p-2.5 text-right">Doanh thu</th>
                </tr>
              </thead>
              <tbody>
                {revenue.map((r, i) => (
                  <tr key={i} className="border-t border-gray-100">
                    <td className="p-2.5">{r.date ?? r.day}</td>
                    <td className="p-2.5 text-right">{fmtNum(r.orders ?? r.count ?? 0)}</td>
                    <td className="p-2.5 text-right">{fmt(r.revenue ?? r.total ?? 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-3">Sản phẩm bán chạy (Top 10)</h2>
        {topProducts.length === 0 ? (
          <div className="text-gray-500 text-sm">Chưa có dữ liệu.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm border border-gray-200 rounded-md min-w-[520px]">
              <thead className="bg-gray-50 text-left">
                <tr>
                  <th className="p-2.5 w-10">#</th>
                  <th className="p-2.5">Sản phẩm</th>
                  <th className="p-2.5 text-right">Đã bán</th>
                  <th className="p-2.5 text-right">Doanh thu</th>
                </tr>
              </thead>
              <tbody>
                {topProducts.map((p, i) => (
                  <tr key={i} className="border-t border-gray-100">
                    <td className="p-2.5 text-gray-500">{i + 1}</td>
                    <td className="p-2.5">{p.name ?? p.productName}</td>
                    <td className="p-2.5 text-right">{fmtNum(p.sold ?? p.quantity ?? 0)}</td>
                    <td className="p-2.5 text-right">{fmt(p.revenue ?? 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-3">Báo cáo thử đồ ảo</h2>
        {!tryon ? (
          <div className="text-gray-500 text-sm">Chưa có dữ liệu.</div>
        ) : (
          <div className="space-y-3">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="border border-gray-200 rounded-md p-3">
                <div className="inline-flex items-center gap-1.5 text-xs text-gray-500 mb-1">
                  <Activity className="w-3.5 h-3.5" /> Tổng lượt
                </div>
                <div className="text-xl font-semibold">{fmtNum(tryon.total)}</div>
              </div>
              <div className="border border-gray-200 rounded-md p-3">
                <div className="inline-flex items-center gap-1.5 text-xs text-gray-500 mb-1">
                  <CheckCircle2 className="w-3.5 h-3.5 text-green-600" /> Thành công
                </div>
                <div className="text-xl font-semibold text-green-700">{fmtNum(tryon.success)}</div>
              </div>
              <div className="border border-gray-200 rounded-md p-3">
                <div className="inline-flex items-center gap-1.5 text-xs text-gray-500 mb-1">
                  <XCircle className="w-3.5 h-3.5 text-red-600" /> Thất bại
                </div>
                <div className="text-xl font-semibold text-red-700">{fmtNum(tryon.failed)}</div>
              </div>
              <div className="border border-gray-200 rounded-md p-3">
                <div className="inline-flex items-center gap-1.5 text-xs text-gray-500 mb-1">
                  <Sparkles className="w-3.5 h-3.5 text-[var(--color-accent)]" /> Tỉ lệ thành công
                </div>
                <div className="text-xl font-semibold">{successRate}%</div>
              </div>
            </div>

            {tryon.byBackend?.length > 0 && (
              <div className="border border-gray-200 rounded-md overflow-hidden">
                <div className="bg-gray-50 px-3 py-2 text-xs uppercase tracking-wider text-gray-500">
                  Theo backend
                </div>
                <table className="w-full text-sm">
                  <tbody>
                    {tryon.byBackend.map((b: any) => (
                      <tr key={b.backend ?? '-'} className="border-t border-gray-100">
                        <td className="p-2.5">{b.backend ?? '(none)'}</td>
                        <td className="p-2.5 text-right">{fmtNum(b.count)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}

