import { useParams, Link } from "react-router-dom";
import { useState } from "react";
import { Star, ShoppingCart, Shirt, ArrowLeft, Check } from "lucide-react";
import { products, formatPrice } from "../data/products";
import { useCart } from "../context/CartContext";

export default function ProductDetail() {
  const { id } = useParams();
  const product = products.find((p) => p.id === Number(id));
  const { addItem } = useCart();
  const [selectedSize, setSelectedSize] = useState("");
  const [selectedColor, setSelectedColor] = useState("");
  const [added, setAdded] = useState(false);

  if (!product) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-20 text-center">
        <p className="text-gray-400 text-lg">Sản phẩm không tồn tại.</p>
        <Link to="/" className="text-primary hover:underline mt-4 inline-block">← Quay lại</Link>
      </div>
    );
  }

  const handleAddToCart = () => {
    if (!selectedSize || !selectedColor) return;
    addItem(product, selectedSize, selectedColor);
    setAdded(true);
    setTimeout(() => setAdded(false), 2000);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <Link to="/" className="inline-flex items-center gap-1 text-gray-400 hover:text-white mb-6 transition-colors">
        <ArrowLeft className="w-4 h-4" /> Quay lại
      </Link>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
        {/* Image */}
        <div className="relative rounded-2xl overflow-hidden bg-surface">
          <img src={product.image} alt={product.name} className="w-full h-[500px] object-cover" />
          {product.badge && (
            <span className="absolute top-4 left-4 px-4 py-1.5 bg-accent text-black text-sm font-bold rounded-full">
              {product.badge}
            </span>
          )}
        </div>

        {/* Info */}
        <div>
          <p className="text-sm text-primary uppercase tracking-wider mb-2">{product.category}</p>
          <h1 className="text-3xl font-bold text-white mb-4">{product.name}</h1>

          <div className="flex items-center gap-2 mb-6">
            <div className="flex">
              {[...Array(5)].map((_, i) => (
                <Star
                  key={i}
                  className={`w-4 h-4 ${i < Math.round(product.rating) ? "fill-accent text-accent" : "text-gray-600"}`}
                />
              ))}
            </div>
            <span className="text-sm text-gray-400">{product.rating} ({product.reviews} đánh giá)</span>
          </div>

          <div className="flex items-baseline gap-3 mb-6">
            <span className="text-3xl font-bold text-white">{formatPrice(product.price)}</span>
            {product.originalPrice && (
              <span className="text-lg text-gray-500 line-through">{formatPrice(product.originalPrice)}</span>
            )}
          </div>

          <p className="text-gray-400 mb-8 leading-relaxed">{product.description}</p>

          {/* Size */}
          <div className="mb-6">
            <label className="text-sm font-semibold text-gray-300 mb-3 block">Kích thước</label>
            <div className="flex flex-wrap gap-2">
              {product.sizes.map((size) => (
                <button
                  key={size}
                  onClick={() => setSelectedSize(size)}
                  className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-all ${
                    selectedSize === size
                      ? "bg-primary text-white ring-2 ring-primary/50"
                      : "bg-surface-light text-gray-300 hover:bg-white/10"
                  }`}
                >
                  {size}
                </button>
              ))}
            </div>
          </div>

          {/* Color */}
          <div className="mb-8">
            <label className="text-sm font-semibold text-gray-300 mb-3 block">Màu sắc</label>
            <div className="flex flex-wrap gap-2">
              {product.colors.map((color) => (
                <button
                  key={color}
                  onClick={() => setSelectedColor(color)}
                  className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-all ${
                    selectedColor === color
                      ? "bg-primary text-white ring-2 ring-primary/50"
                      : "bg-surface-light text-gray-300 hover:bg-white/10"
                  }`}
                >
                  {color}
                </button>
              ))}
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <button
              onClick={handleAddToCart}
              disabled={!selectedSize || !selectedColor}
              className="flex-1 flex items-center justify-center gap-2 px-6 py-4 bg-gradient-to-r from-primary to-primary-dark text-white font-semibold rounded-xl hover:shadow-lg hover:shadow-primary/25 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {added ? (
                <>
                  <Check className="w-5 h-5" /> Đã thêm!
                </>
              ) : (
                <>
                  <ShoppingCart className="w-5 h-5" /> Thêm vào giỏ
                </>
              )}
            </button>
            <Link
              to={`/try-on/${product.id}`}
              className="flex items-center justify-center gap-2 px-6 py-4 bg-accent/10 text-accent hover:bg-accent/20 font-semibold rounded-xl transition-colors"
            >
              <Shirt className="w-5 h-5" /> Thử đồ AI
            </Link>
          </div>

          {!selectedSize && !selectedColor && (
            <p className="text-xs text-gray-500 mt-3">* Vui lòng chọn kích thước và màu sắc trước khi thêm vào giỏ</p>
          )}
        </div>
      </div>
    </div>
  );
}
