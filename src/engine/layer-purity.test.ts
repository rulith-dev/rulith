import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

/**
 * Layer-purity guard (foundations.md: kernel <- engine <- agent).
 *
 * The deductive core must not depend upward on its drivers: nothing in
 * src/kernel may import engine or agent, and nothing in src/engine may import
 * agent. (Test files are exempt — fixtures legitimately exercise the full stack.)
 *
 * This was RED before the derive-stages split: compile-board(-sql) imported the
 * agent's `deriveStages`. The pure layering now lives in engine/derive-stages.ts,
 * so engine no longer reaches into agent. Keep this green.
 */

const SRC = join(dirname(fileURLToPath(import.meta.url)), '..')

function nonTestSources(dir: string): string[] {
  const out: string[] = []
  for (const entry of readdirSync(dir)) {
    const p = join(dir, entry)
    if (statSync(p).isDirectory()) continue
    if (entry.endsWith('.ts') && !entry.endsWith('.test.ts')) out.push(p)
  }
  return out
}

function upwardImports(file: string, forbidden: string[]): string[] {
  const src = readFileSync(file, 'utf8')
  const hits: string[] = []
  const re = /\bfrom\s+['"]([^'"]+)['"]/g
  let m: RegExpExecArray | null
  while ((m = re.exec(src)) !== null) {
    if (forbidden.some((pre) => m![1] === pre || m![1].startsWith(pre + '/'))) {
      hits.push(m[1])
    }
  }
  return hits
}

test('layer purity: src/kernel imports neither engine nor agent', () => {
  for (const file of nonTestSources(join(SRC, 'kernel'))) {
    const bad = upwardImports(file, ['../engine', '../agent'])
    assert.deepEqual(bad, [], `${file} imports up: ${bad.join(', ')}`)
  }
})

test('layer purity: src/engine does not import agent', () => {
  for (const file of nonTestSources(join(SRC, 'engine'))) {
    const bad = upwardImports(file, ['../agent'])
    assert.deepEqual(bad, [], `${file} imports up into agent: ${bad.join(', ')}`)
  }
})
