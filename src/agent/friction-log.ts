/**
 * friction-log — persistence + prompt-adaptation layer around FrictionProfile.
 *
 * Writes per-model failure counts to a JSONL file (one FrictionRecord per
 * model×kind) and reads them back across runs, so the prompt can be front-
 * loaded with the guard this model keeps tripping.
 *
 * Boundary (foundations.md §区分本任务事实 vs 可迁移经验): the on-disk shape
 * stores ONLY model name, failure kind, and count — no task-specific atoms,
 * no board facts, no op ids.  Everything task-local stays task-local.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { FrictionProfile, fixTemplateFor, type FailureKind } from './failure-taxonomy.js'

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** The on-disk shape: transferable signal only (model + kind + count). */
export type FrictionRecord = {
  model: string
  kind: FailureKind
  count: number
}

// ---------------------------------------------------------------------------
// saveFrictionProfile
// ---------------------------------------------------------------------------

/**
 * Serialize a FrictionProfile to a JSONL file.
 * Each line is one FrictionRecord (model × kind × count).
 * The file is overwritten on every call (full snapshot, not append).
 */
export function saveFrictionProfile(profile: FrictionProfile, path: string): void {
  const lines: string[] = []
  for (const model of profile.models()) {
    for (const { kind, count } of profile.forModel(model)) {
      const record: FrictionRecord = { model, kind, count }
      lines.push(JSON.stringify(record))
    }
  }
  writeFileSync(path, lines.length > 0 ? lines.join('\n') + '\n' : '', 'utf8')
}

// ---------------------------------------------------------------------------
// mergeInto
// ---------------------------------------------------------------------------

/**
 * Fold an array of FrictionRecords into a live FrictionProfile.
 * Each record's count is replayed as that many individual record() calls so
 * the profile's internal frequency ordering stays consistent.
 */
export function mergeInto(profile: FrictionProfile, records: FrictionRecord[]): void {
  for (const { model, kind, count } of records) {
    for (let i = 0; i < count; i++) {
      profile.record(model, kind)
    }
  }
}

// ---------------------------------------------------------------------------
// loadFrictionProfile
// ---------------------------------------------------------------------------

/**
 * Read a JSONL file back into a fresh FrictionProfile.
 * Missing file → returns an empty profile (no throw).
 */
export function loadFrictionProfile(path: string): FrictionProfile {
  const profile = new FrictionProfile()
  let raw: string
  try {
    raw = readFileSync(path, 'utf8')
  } catch {
    return profile
  }
  const records = raw
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line) as FrictionRecord)
  mergeInto(profile, records)
  return profile
}

// ---------------------------------------------------------------------------
// frictionPreamble
// ---------------------------------------------------------------------------

/**
 * A short prompt-injection block listing this model's top friction kinds and
 * their copyable fix templates, suitable for prepending to a system prompt.
 * Returns '' when the model has no recorded friction.
 */
export function frictionPreamble(profile: FrictionProfile, model: string, top = 2): string {
  const hints = profile.adaptationHints(model, top)
  if (hints.length === 0) return ''
  const lines = [
    `[friction-profile for ${model}]`,
    ...hints.map((h) => `- ${h}`),
    `[end friction-profile]`,
  ]
  return lines.join('\n')
}
