import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ArrowLeft, Package, User, MapPin, CreditCard } from 'lucide-react';
import { sellerApi, ApiError } from '../../lib/api';
import { useToast } from '../../components/Toast';

const STATUS_LABEL: Record<string, string> = {
  PENDING: 'Chờ xác nhận',
  CONFIRMED: 'Đã xác nhận',
  PACKING: 'Đang đóng gói',
  SHIPPING: 'Đang giao',
  COMPLETED: 'Hoàn thành',
  CANCELLED: 'Đã huỷ',
};

const STATUS_COLOR: Record<string, string> = {
  PENDING: 'bg-yellow-100 text-yellow-700',
  CONFIRMED: 'bg-blue-100 text-blue-700',
  PACKING: 'bg-indigo-100 text-indigo-700',
  SHIPPING: 'bg-purple-100 text-purple-700',
  COMPLETED: 'bg-green-100 text-green-700',
  CANCELLED: 'bg-red-100 text-red-700',
};

const NEXT: Record<string, string[]> = {
  PENDING: [],
  CONFIRMED: ['PACKING', 'CANCELLED'],
  PACKING: ['SHIPPING'],
  SHIPPING: ['COMPLETED'],
};

const fmt = (v: any) => new Intl.NumberFormat('vi-VN').format(+v) + 'đ';
const dt = (s?: string) =>
  s ? new Date(s).toLocaleString('vi-VN', { dateStyle: 'short', timeStyle: 'short' }) : '-';

export default function SellerOrderDetail() {
  const { code = '' } = useParams();
  const toast = useToast();
  const [order, setOrder] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [acting, setActing] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      setOrder(await sellerApi.orderDetail(code));
    } catch (e: any) {
      toast.show(e instanceof ApiError ? e.message : 'Lỗi tải đơn', 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [code]);

  const act = async (action: 'confirm' | string) => {
    setActing(true);
    try {
      if (action === 'confirm') await sellerApi.confirm(code);
      else await sellerApi.updateStatus(code, action);
      toast.show('Cập nhật thành công', 'success');
      await load();
    } catch (e: any) {
      toast.show(e instanceof ApiError ? e.message : 'Lỗi cập nhật', 'error');
    } finally {
      setActing(false);
    }
  };

  if (loading) return <div className="text-gray-500">Đang tải...</div>;
  if (!order)
    return (
      <div className="text-gray-500">
        Không tìm thấy đơn.{' '}
        <Link to="/seller/orders" className="underline">
          Quay lại
        </Link>
      </div>
    );

  const items: any[] = order.items ?? [];
  const addr = order.shippingAddress ?? {};

  return (
    <div>
      <Link
        to="/seller/orders"
        className="inline-flex items-center gap-1 text-sm text-gray-600 hover:text-black mb-4"
      >
        <ArrowLeft className="w-4 h-4" /> Quay lại danh sách
      </Link>

      <div className="flex items-start justify-between flex-wrap gap-3 mb-5">
        <div>
          <h2 className="text-xl font-semibold">Đơn #{order.code}</h2>
          <p className="text-xs text-gray-500 mt-0.5">Đặt lúc {dt(order.placedAt)}</p>
        </div>
        <span
          className={`text-xs px-2.5 py-1 rounded font-medium ${
            STATUS_COLOR[order.status] ?? 'bg-gray-100 text-gray-700'
          }`}
        >
          {STATUS_LABEL[order.status] ?? order.status}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
        <div className="border border-gray-200 rounded-md p-3 text-sm">
          <div className="flex items-center gap-1.5 text-gray-500 mb-1.5 text-xs uppercase tracking-wider">
            <User className="w-3.5 h-3.5" /> Khách hàng
          </div>
          <div className="font-medium">{order.user?.fullName ?? '-'}</div>
          <div className="text-xs text-gray-600">{order.user?.email}</div>
        </div>
        <div className="border border-gray-200 rounded-md p-3 text-sm">
          <div className="flex items-center gap-1.5 text-gray-500 mb-1.5 text-xs uppercase tracking-wider">
            <MapPin className="w-3.5 h-3.5" /> Giao đến
          </div>
          <div className="font-medium">{addr.recipient ?? '-'}</div>
          <div className="text-xs text-gray-600">{addr.phone}</div>
          <div className="text-xs text-gray-600 mt-0.5">
            {[addr.line1, addr.ward, addr.district, addr.city].filter(Boolean).join(', ')}
          </div>
        </div>
        <div className="border border-gray-200 rounded-md p-3 text-sm">
          <div className="flex items-center gap-1.5 text-gray-500 mb-1.5 text-xs uppercase tracking-wider">
            <CreditCard className="w-3.5 h-3.5" /> Thanh toán
          </div>
          <div className="font-medium">{order.paymentMethod ?? '-'}</div>
          <div className="text-xs text-gray-600">{order.paymentStatus ?? '-'}</div>
        </div>
      </div>

      <div className="border border-gray-200 rounded-md overflow-hidden mb-5">
        <div className="bg-gray-50 px-3 py-2 text-xs uppercase tracking-wider text-gray-500 flex items-center gap-1.5">
          <Package className="w-3.5 h-3.5" /> Sản phẩm
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm min-w-[640px]">
          <thead className="text-left bg-white">
            <tr className="border-b border-gray-100">
              <th className="p-2.5 font-medium">Tên</th>
              <th className="p-2.5 font-medium">Phân loại</th>
              <th className="p-2.5 font-medium text-right">SL</th>
              <th className="p-2.5 font-medium text-right">Đơn giá</th>
              <th className="p-2.5 font-medium text-right">Tổng</th>
            </tr>
          </thead>
          <tbody>
            {items.map((it: any) => (
              <tr key={it.id} className="border-t border-gray-100">
                <td className="p-2.5">{it.product?.name ?? it.productName ?? '-'}</td>
                <td className="p-2.5 text-xs text-gray-600">
                  {[it.variant?.color, it.variant?.size].filter(Boolean).join(' / ') || '-'}
                </td>
                <td className="p-2.5 text-right">{it.quantity}</td>
                <td className="p-2.5 text-right">{fmt(it.unitPrice)}</td>
                <td className="p-2.5 text-right font-medium">
                  {fmt(+it.unitPrice * +it.quantity)}
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot className="bg-gray-50">
            <tr>
              <td colSpan={4} className="p-2.5 text-right text-gray-600">
                Tạm tính
              </td>
              <td className="p-2.5 text-right">{fmt(order.subtotal ?? order.total)}</td>
            </tr>
            {order.shippingFee != null && (
              <tr>
                <td colSpan={4} className="p-2.5 text-right text-gray-600">
                  Phí ship
                </td>
                <td className="p-2.5 text-right">{fmt(order.shippingFee)}</td>
              </tr>
            )}
            <tr className="border-t border-gray-200">
              <td colSpan={4} className="p-2.5 text-right font-semibold">
                Tổng cộng
              </td>
              <td className="p-2.5 text-right font-semibold">{fmt(order.total)}</td>
            </tr>
          </tfoot>
          </table>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        {order.status === 'PENDING' && (
          <button
            onClick={() => act('confirm')}
            disabled={acting}
            className="px-3 py-2 bg-[var(--color-ink)] text-white rounded-md text-sm disabled:opacity-50"
          >
            Xác nhận đơn
          </button>
        )}
        {NEXT[order.status]?.map((next) => (
          <button
            key={next}
            onClick={() => act(next)}
            disabled={acting}
            className="px-3 py-2 border border-gray-200 rounded-md text-sm hover:bg-gray-50 disabled:opacity-50"
          >
            → {STATUS_LABEL[next]}
          </button>
        ))}
      </div>
    </div>
  );
}
