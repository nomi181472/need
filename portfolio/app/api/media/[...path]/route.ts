import { createReadStream } from "node:fs";
import { realpath, stat } from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";
import manifest from "@/content/manifest.json";
import { parseRange } from "@/lib/range";
import type { NextRequest } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const DATA_ROOT = path.resolve(process.cwd(), "..");
const MANIFEST_SOURCES = new Map(manifest.recordings.map((r) => [r.path, r.source]));

type MediaPath = { filePath: string; root: string; manifestPath: string };

function resolveMediaPath(segments: string[]): MediaPath | null {
  if (segments.length < 2 || segments.some((segment) =>
    !segment || segment === "." || segment === ".." || /[/\\\0]/.test(segment)
  )) return null;
  const source = segments[0];
  if (source !== "data" && source !== "data_2") return null;
  const manifestPath = segments.join("/");
  if (MANIFEST_SOURCES.get(manifestPath) !== source) return null;
  const root = path.join(DATA_ROOT, source);
  return { filePath: path.join(DATA_ROOT, manifestPath), root, manifestPath };
}

function buildMediaTargetUrl(manifestPath: string): string | null {
  const base = process.env.MEDIA_BASE_URL;
  if (!base) return null;
  let parsedBase: URL;
  try {
    parsedBase = new URL(base);
  } catch {
    return null;
  }
  if (parsedBase.protocol !== "https:" && parsedBase.protocol !== "http:") return null;
  const relative = manifestPath.split("/").map(encodeURIComponent).join("/");
  parsedBase.pathname = `${parsedBase.pathname.replace(/\/+$/, "")}/${relative}`;
  parsedBase.hash = "";
  return parsedBase.toString();
}

async function serveLocalFile(request: NextRequest, resolved: MediaPath): Promise<Response> {
  const { filePath, root } = resolved;
  let real: string;
  let realRoot: string;
  let realParent: string;
  try {
    [real, realRoot, realParent] = await Promise.all([
      realpath(filePath), realpath(root), realpath(DATA_ROOT),
    ]);
  } catch {
    return new Response(null, { status: 404 });
  }
  if (!realRoot.startsWith(realParent + path.sep)) {
    return new Response(null, { status: 404 });
  }
  if (real !== realRoot && !real.startsWith(realRoot + path.sep)) {
    return new Response(null, { status: 404 });
  }

  let fileStat: Awaited<ReturnType<typeof stat>>;
  try {
    fileStat = await stat(real);
  } catch {
    return new Response(null, { status: 404 });
  }
  if (!fileStat.isFile()) return new Response(null, { status: 404 });

  const size = fileStat.size;
  const parsed = parseRange(request.headers.get("range"), size);

  if (parsed === false) {
    return new Response(null, {
      status: 416,
      headers: { "Content-Range": `bytes */${size}` },
    });
  }

  const { start, end } = parsed ?? { start: 0, end: size - 1 };
  const status = parsed ? 206 : 200;
  const headers = new Headers();
  headers.set("Content-Type", "video/mp4");
  headers.set("Accept-Ranges", "bytes");
  headers.set("Content-Length", String(end - start + 1));
  headers.set("Cache-Control", "public, max-age=3600");
  headers.set("ETag", `"${size}"`);

  if (parsed) headers.set("Content-Range", `bytes ${start}-${end}/${size}`);

  if (request.method === "HEAD" || size === 0) {
    return new Response(null, { status, headers });
  }

  const nodeStream = createReadStream(real, { start, end });
  return new Response(Readable.toWeb(nodeStream) as ReadableStream<Uint8Array>, {
    status,
    headers,
  });
}

async function mediaResponse(request: NextRequest, segments: string[]): Promise<Response> {
  const resolved = resolveMediaPath(segments);
  if (resolved === null) return new Response(null, { status: 404 });
  const target = buildMediaTargetUrl(resolved.manifestPath);
  if (target !== null) {
    return new Response(null, { status: 302, headers: { Location: target } });
  }
  return serveLocalFile(request, resolved);
}

export async function GET(request: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return mediaResponse(request, (await ctx.params).path);
}

export async function HEAD(request: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return mediaResponse(request, (await ctx.params).path);
}
