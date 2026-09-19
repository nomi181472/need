export type ByteRange = { start: number; end: number };

export function parseRange(value: string | null, size: number): ByteRange | null | false {
  if (value === null) return null;
  if (!Number.isSafeInteger(size) || size <= 0) return false;
  const match = /^bytes=(\d*)-(\d*)$/i.exec(value.trim());
  if (!match || (!match[1] && !match[2])) return false;
  const first = match[1] ? Number(match[1]) : null;
  const last = match[2] ? Number(match[2]) : null;
  if ((first !== null && !Number.isSafeInteger(first)) ||
      (last !== null && !Number.isSafeInteger(last))) return false;
  if (first === null) {
    if (last === null || last <= 0) return false;
    return { start: Math.max(0, size - last), end: size - 1 };
  }
  if (first >= size || (last !== null && last < first)) return false;
  return { start: first, end: Math.min(last ?? size - 1, size - 1) };
}
