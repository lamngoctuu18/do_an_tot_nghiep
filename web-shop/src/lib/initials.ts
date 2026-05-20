export function getInitials(name?: string | null): string {
  if (!name) return "?";
  const parts = name
    .trim()
    .split(/\s+/)
    .filter(Boolean);
  if (parts.length === 0) return "?";
  const letters = parts.map((p) => p.charAt(0).toUpperCase());
  return letters.slice(-3).join("");
}
