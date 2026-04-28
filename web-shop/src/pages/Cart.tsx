import { Link } from "react-router-dom";
import { Trash2, Plus, Minus, ShoppingBag, ArrowLeft } from "lucide-react";
import { useCart } from "../context/CartContext";
import { formatPrice } from "../data/products";

export default function Cart() {
  const { items, removeItem, updateQuantity, clearCart, totalPrice } = useCart();

  if (items.length === 0) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-20 text-center">
        <ShoppingBag className="w-20 h-20 text-gray-600 mx-auto mb-6" />
        <h2 className="text-2xl font-bold text-white mb-3">Giỏ hàng trống</h2>
        <p className="text-gray-400 mb-6">Hãy chọn sản phẩm yêu thích để thêm vào giỏ hàng.</p>
        <Link
          to="/"
          className="inline-flex items-center gap-2 px-6 py-3 bg-primary text-white font-semibold rounded-xl hover:bg-primary-dark transition-colors"
        >
          <ArrowLeft className="w-4 h-4" /> Tiếp tục mua sắm
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <Link to="/" className="inline-flex items-center gap-1 text-gray-400 hover:text-white mb-6 transition-colors">
        <ArrowLeft className="w-4 h-4" /> Tiếp tục mua sắm
      </Link>

      <h1 className="text-2xl font-bold text-white mb-8">
        Giỏ hàng ({items.reduce((s, i) => s + i.quantity, 0)} sản phẩm)
      </h1>

      <div className="space-y-4">
        {items.map((item) => (
          <div
            key={`${item.product.id}-${item.size}-${item.color}`}
            className="flex items-center gap-4 bg-surface rounded-xl p-4 border border-white/5"
          >
            <img
              src={item.product.image}
              alt={item.product.name}
              className="w-20 h-20 rounded-lg object-cover"
            />
            <div className="flex-1 min-w-0">
              <Link to={`/product/${item.product.id}`} className="font-semibold text-white hover:text-primary transition-colors">
                {item.product.name}
              </Link>
              <p className="text-sm text-gray-400 mt-1">
                {item.size} · {item.color}
              </p>
              <p className="text-primary font-bold mt-1">{formatPrice(item.product.price)}</p>
            </div>

            {/* Quantity */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => updateQuantity(item.product.id, item.size, item.color, item.quantity - 1)}
                className="p-1.5 bg-surface-light rounded-lg hover:bg-white/10 transition-colors"
              >
                <Minus className="w-4 h-4" />
              </button>
              <span className="w-8 text-center font-medium">{item.quantity}</span>
              <button
                onClick={() => updateQuantity(item.product.id, item.size, item.color, item.quantity + 1)}
                className="p-1.5 bg-surface-light rounded-lg hover:bg-white/10 transition-colors"
              >
                <Plus className="w-4 h-4" />
              </button>
            </div>

            <p className="font-bold text-white w-28 text-right">
              {formatPrice(item.product.price * item.quantity)}
            </p>

            <button
              onClick={() => removeItem(item.product.id, item.size, item.color)}
              className="p-2 text-gray-400 hover:text-red-400 transition-colors"
            >
              <Trash2 className="w-5 h-5" />
            </button>
          </div>
        ))}
      </div>

      {/* Summary */}
      <div className="mt-8 bg-surface rounded-2xl p-6 border border-white/5">
        <div className="flex justify-between items-center mb-4">
          <span className="text-gray-400">Tạm tính</span>
          <span className="text-white font-medium">{formatPrice(totalPrice)}</span>
        </div>
        <div className="flex justify-between items-center mb-4">
          <span className="text-gray-400">Phí vận chuyển</span>
          <span className="text-green-400 font-medium">Miễn phí</span>
        </div>
        <div className="border-t border-white/10 pt-4 flex justify-between items-center">
          <span className="text-lg font-bold text-white">Tổng cộng</span>
          <span className="text-2xl font-bold text-accent">{formatPrice(totalPrice)}</span>
        </div>

        <div className="flex gap-3 mt-6">
          <button className="flex-1 py-4 bg-gradient-to-r from-primary to-primary-dark text-white font-semibold rounded-xl hover:shadow-lg hover:shadow-primary/25 transition-all">
            Thanh toán
          </button>
          <button
            onClick={clearCart}
            className="px-6 py-4 bg-surface-light text-gray-300 hover:text-red-400 rounded-xl transition-colors"
          >
            Xóa tất cả
          </button>
        </div>
      </div>
    </div>
  );
}
