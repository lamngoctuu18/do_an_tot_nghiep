import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ordersApi, ApiError } from '../../lib/api';
import { ChevronLeft, XCircle } from 'lucide-react';
import { useToast } from '../../components/Toast';
import ReasonModal from '../../components/ReasonModal';

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

const fmt = (v: any) => new Intl.NumberFormat('vi-VN').format(+v) + 'đ';

export default function OrderDetail() {
  const { code = '' } = useParams();
  const toast = useToast();
  const [order, setOrder] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [showCancel, setShowCancel] = useState(false);

  const load = () => {
    setLoading(true);
    ordersApi.detail(code).then(setOrder).finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, [code]);

  const cancel = async (reason: string) => {
    try {
      await ordersApi.cancel(code, reason);
      toast.show(`Đã huỷ đơn #${code}`, 'success');
      setShowCancel(false);
      load();
    } catch (e: any) {
      const msg = e instanceof ApiError ? e.message : 'Lỗi huỷ đơn';
      toast.show(msg, 'error');
      throw new Error(msg);
    }
  };

  if (loading) return <div className="text-gray-500">Đang tải...</div>;
  if (!order) return <div>Không tìm thấy đơn hàng.</div>;

  const canCancel = ['PENDING', 'CONFIRMED'].includes(order.status);
  const addr = order.addressSnapshot ?? {};

  return (
    <div>
      <Link to="/account/orders" className="inline-flex items-center text-sm text-gray-600 mb-4">
        <ChevronLeft className="w-4 h-4" /> Quay lại
      </Link>
      <div className="flex justify-between items-start mb-4 gap-3 flex-wrap">
        <div>
          <h2 className="text-xl font-semibold">Đơn #{order.code}</h2>
          <div className="text-sm text-gray-500">
            {new Date(order.placedAt).toLocaleString('vi-VN')}
          </div>
        </div>
        <span
          className={`text-xs px-2.5 py-1 rounded font-medium ${
            STATUS_COLOR[order.status] ?? 'bg-gray-100 text-gray-700'
          }`}
        >
          {STATUS_LABEL[order.status] ?? order.status}
        </span>
      </div>

      {order.cancelReason && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-md p-3 text-sm mb-4">
          <strong>Lý do huỷ:</strong> {order.cancelReason}
        </div>
      )}

      <section className="border border-gray-200 rounded-md p-4 mb-4">
        <h3 className="font-semibold mb-2">Địa chỉ giao hàng</h3>
        <div className="text-sm">
          {addr.recipient} · {addr.phone}
        </div>
        <div className="text-sm text-gray-600">
          {addr.line1}, {addr.ward}, {addr.district}, {addr.city}
        </div>
      </section>

      <section className="border border-gray-200 rounded-md p-4 mb-4">
        <h3 className="font-semibold mb-3">Sản phẩm</h3>
        <div className="space-y-3">
          {order.items?.map((it: any) => (
            <div key={it.id} className="flex justify-between items-start text-sm">
              <div>
                <div>{it.nameSnapshot}</div>
                <div className="text-gray-500">SL: {it.quantity}</div>
              </div>
              <div className="text-right">{fmt(+it.priceSnapshot * it.quantity)}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="border border-gray-200 rounded-md p-4 mb-4 space-y-1 text-sm">
        <div className="flex justify-between">
          <span>Tạm tính</span>
          <span>{fmt(order.subtotal)}</span>
        </div>
        <div className="flex justify-between">
          <span>Phí vận chuyển</span>
          <span>{fmt(order.shippingFee)}</span>
        </div>
        <div className="flex justify-between font-semibold pt-2 border-t mt-2">
          <span>Tổng cộng</span>
          <span>{fmt(order.total)}</span>
        </div>
        <div className="text-xs text-gray-500 pt-1">
          Phương thức: {order.paymentMethod} · Thanh toán: {order.paymentStatus}
        </div>
      </section>

      {canCancel && (
        <button
          onClick={() => setShowCancel(true)}
          className="inline-flex items-center gap-1.5 border border-red-500 text-red-600 px-4 py-2 rounded-md text-sm hover:bg-red-50"
        >
          <XCircle className="w-4 h-4" /> Huỷ đơn
        </button>
      )}

      <ReasonModal
        open={showCancel}
        title="Lý do huỷ đơn"
        placeholder="Vui lòng nhập lý do (tối thiểu 3 ký tự)..."
        confirmLabel="Xác nhận huỷ"
        onClose={() => setShowCancel(false)}
        onConfirm={cancel}
      />
    </div>
  );
}
