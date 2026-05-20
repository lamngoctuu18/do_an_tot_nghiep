import { useEffect, useState } from 'react';
import { adminApi } from '../../lib/api';

const fmt = (v: any) => new Intl.NumberFormat('vi-VN').format(+v) + 'đ';

function Stat({ label, value }: { label: string; value: any }) {
  return (
    <div className="p-4 rounded-lg border border-gray-200">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className="text-2xl font-semibold">{value}</div>
    </div>
  );
}

export default function AdminDashboard() {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    adminApi.stats().then(setData).catch(() => setData({}));
  }, []);

  if (!data) return <div className="text-gray-500">Đang tải...</div>;

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Tổng quan hệ thống</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="Người dùng" value={data.totalUsers ?? '-'} />
        <Stat label="Shop" value={data.totalSellers ?? data.totalShops ?? '-'} />
        <Stat label="Sản phẩm" value={data.totalProducts ?? '-'} />
        <Stat label="Đơn hàng" value={data.totalOrders ?? '-'} />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-3">
        <Stat label="Shop chờ duyệt" value={data.pendingSellers ?? data.pendingShops ?? '-'} />
        <Stat label="SP chờ duyệt" value={data.pendingProducts ?? '-'} />
        <Stat label="Doanh thu" value={data.totalRevenue !== undefined ? fmt(data.totalRevenue) : '-'} />
      </div>
    </div>
  );
}
