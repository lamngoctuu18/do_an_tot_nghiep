interface Props {
  className?: string;
}

export function Skeleton({ className = '' }: Props) {
  return (
    <div
      className={`bg-gray-100 rounded-md animate-pulse ${className}`}
      aria-hidden="true"
    />
  );
}

export function ProductCardSkeleton() {
  return (
    <div className="space-y-3">
      <Skeleton className="aspect-[3/4] w-full rounded-2xl" />
      <Skeleton className="h-3 w-1/3" />
      <Skeleton className="h-4 w-4/5" />
      <Skeleton className="h-4 w-1/2" />
    </div>
  );
}

export function TableRowSkeleton({ cols = 4 }: { cols?: number }) {
  return (
    <tr className="border-t border-gray-100">
      {Array.from({ length: cols }).map((_, i) => (
        <td key={i} className="p-2.5">
          <Skeleton className="h-3 w-full" />
        </td>
      ))}
    </tr>
  );
}
