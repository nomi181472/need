import assert from "node:assert/strict";
import test from "node:test";
import { parseRange } from "../lib/range.ts";

test("parseRange returns null when no Range header", () => {
  assert.equal(parseRange(null, 100), null);
});

test("parseRange parses simple byte range", () => {
  assert.deepEqual(parseRange("bytes=0-49", 100), { start: 0, end: 49 });
  assert.deepEqual(parseRange("bytes=10-19", 100), { start: 10, end: 19 });
});

test("parseRange clamps end to size - 1", () => {
  assert.deepEqual(parseRange("bytes=90-500", 100), { start: 90, end: 99 });
});

test("parseRange parses open-ended range", () => {
  assert.deepEqual(parseRange("bytes=50-", 100), { start: 50, end: 99 });
});

test("parseRange parses suffix range", () => {
  assert.deepEqual(parseRange("bytes=-10", 100), { start: 90, end: 99 });
  assert.deepEqual(parseRange("bytes=-100", 100), { start: 0, end: 99 });
  assert.deepEqual(parseRange("bytes=-500", 100), { start: 0, end: 99 });
});

test("parseRange rejects invalid ranges", () => {
  assert.equal(parseRange("bytes=abc", 100), false);
  assert.equal(parseRange("bytes=-", 100), false);
  assert.equal(parseRange("items=0-5", 100), false);
  assert.equal(parseRange("bytes=99-10", 100), false);
  assert.equal(parseRange("bytes=100-", 100), false);
  assert.equal(parseRange("bytes=0-49", 0), false);
  assert.equal(parseRange("bytes=0-49", -5), false);
});

test("parseRange rejects malformed unit-only or empty header", () => {
  assert.equal(parseRange("", 100), false);
  assert.equal(parseRange("bytes=", 100), false);
});
