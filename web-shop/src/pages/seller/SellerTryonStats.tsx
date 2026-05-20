import { useEffect, useState } from 'react';
import { sellerApi } from '../../lib/api';

export default function SellerTryonStats() {
  const [rows, setRows] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    sellerApi.tryonStats().then(setRows).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-gray-500">Đang tải...</div>;

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Thống kê thử đồ ảo</h2>
      {rows.length === 0 ? (
        <div className="text-gray-500 border border-dashed border-gray-200 rounded-md p-8 text-center">
          Chưa có dữ liệu thử đồ.
        </div>
      ) : (
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-left">
            <tr>
              <th className="p-2.5">Sản phẩm</th>
              <th className="p-2.5 text-right">Lượt thử</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.productId} className="border-t border-gray-100">
                <td className="p-2.5">{r.productName}</td>
                <td className="p-2.5 text-right">{r.tryonCount}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
