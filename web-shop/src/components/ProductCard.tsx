import { Link } from "react-router-dom";
import { Star, ShoppingCart, Shirt } from "lucide-react";
import { formatPrice } from "../data/products";
import type { Product } from "../data/products";

interface Props {
  product: Product;
}

export default function ProductCard({ product }: Props) {
  return (
    <div className="group bg-surface rounded-2xl overflow-hidden border border-white/5 hover:border-primary/30 transition-all duration-300 hover:shadow-lg hover:shadow-primary/5">
      <Link to={`/product/${product.id}`} className="block relative overflow-hidden">
        <img
          src={product.image}
          alt={product.name}
          className="w-full h-72 object-cover group-hover:scale-105 transition-transform duration-500"
        />
        {product.badge && (
          <span className="absolute top-3 left-3 px-3 py-1 bg-accent text-black text-xs font-bold rounded-full">
            {product.badge}
          </span>
        )}
        {product.originalPrice && (
          <span className="absolute top-3 right-3 px-2 py-1 bg-red-500 text-white text-xs font-bold rounded-full">
            -{Math.round((1 - product.price / product.originalPrice) * 100)}%
          </span>
        )}
      </Link>

      <div className="p-4">
        <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">
          {product.category}
        </p>
        <Link to={`/product/${product.id}`}>
          <h3 className="font-semibold text-white group-hover:text-primary transition-colors line-clamp-1">
            {product.name}
          </h3>
        </Link>

        <div className="flex items-center gap-1 mt-2">
          <Star className="w-3.5 h-3.5 fill-accent text-accent" />
          <span className="text-sm text-gray-300">{product.rating}</span>
          <span className="text-xs text-gray-500">({product.reviews})</span>
        </div>

        <div className="flex items-center justify-between mt-3">
          <div>
            <span className="text-lg font-bold text-white">{formatPrice(product.price)}</span>
            {product.originalPrice && (
              <span className="text-sm text-gray-500 line-through ml-2">
                {formatPrice(product.originalPrice)}
              </span>
            )}
          </div>
        </div>

        <div className="flex gap-2 mt-3">
          <Link
            to={`/try-on/${product.id}`}
            className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 bg-primary/10 text-primary hover:bg-primary/20 rounded-lg text-sm font-medium transition-colors"
          >
            <Shirt className="w-4 h-4" />
            Thử đồ
          </Link>
          <Link
            to={`/product/${product.id}`}
            className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 bg-accent/10 text-accent hover:bg-accent/20 rounded-lg text-sm font-medium transition-colors"
          >
            <ShoppingCart className="w-4 h-4" />
            Chi tiết
          </Link>
        </div>
      </div>
    </div>
  );
}
