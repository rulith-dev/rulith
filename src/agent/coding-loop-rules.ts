import type { WorkingMemoryOperation } from '../engine/working-memory.js'
import type { PredicateAtom } from '../model/types.js'

/**
 * attestedDerivations contract for the coding loop: ANY model-authored rule producing
 * finding(kind=fixed) MUST read the machine evidence in its body — edited(issue) +
 * test_result(status=pass), co-bound to the same issue. attestedPredicates alone (test_result/
 * edited machine-only) does NOT stop a model from writing a rule that derives finding(kind=fixed)
 * from a non-attested fact (diagnosis), laundering the verdict past the evidence — deepseek-flash-v4
 * found exactly this in bench-coding-trust (10% false-cert). The seed AX_FIXED is exempt (system
 * load); only model-authored rules are checked. Shared by runCodingTask and bench-coding-trust.
 */
export const FIXED_DERIVATION_CONTRACT: { head: PredicateAtom; requires: PredicateAtom[] }[] = [
  {
    head: { predicate: 'finding', args: { kind: 'fixed', issue: '?i' } },
    requires: [
      { predicate: 'edited', args: { issue: '?i' } },
      { predicate: 'test_result', args: { test: '?t', status: 'pass' } },
    ],
  },
]

/**
 * coding-loop-rules — the reusable coding-loop rule pack (docs/reference.md).
 * Until now these rules lived only in the example fixtures; the real /task flow
 * needs a single pack to seed so every coding task gets the same load-bearing
 * guards.
 *
 * Predicate family:
 *   issue(id)                     — a bug/feature to resolve.
 *   fix_test(issue, test)         — the test that DEFINES "done" for the issue.
 *   test_result(test, status)     — MACHINE FACT from the real runner (attested
 *                                   by run_check via parseTapResults), never the
 *                                   model's word.
 *   edited(file, line)            — MACHINE FACT: the editor touched a file
 *                                   (attested by edit_file), never the model's word.
 *   touches(issue, file)          — the model's claim of which file an issue lives
 *                                   in (a claim, not a machine fact).
 *   diagnosis(issue, kind)        — the model's understanding of the cause.
 *   baseline_pass(test)           — MACHINE FACT: a test that passed at the
 *                                   PRE-EDIT baseline (attested by runCodingTask's
 *                                   one-shot baseline capture before the model
 *                                   touches anything), never the model's word — so
 *                                   a regression cannot be hidden by omitting it.
 *
 * Derived (the closure earns these — the model cannot assert them past the gate):
 *   verified(issue)            <- fix_test(issue,test) AND test_result(test,pass)
 *   edited(issue)              <- edited(file) AND touches(issue,file)   (the bridge)
 *   finding(kind=fixed, issue) <- issue AND edited(issue) AND verified(issue) AND diagnosis(issue,_)
 *   finding(kind=regression,t) <- baseline_pass(t) AND test_result(t,fail)
 *   needs_acceptance_test(issue) <- issue AND NOT has_fix_test(issue)
 *
 * The last one is the INTAKE GATE: an issue with no fix_test has no checkable
 * "done" (verified can never derive), so the board flags it — the agent must
 * declare, or propose, the test that defines done before it can ever finish.
 * That is the acceptance requirement forcing a checkable target, board-native.
 * (`has_fix_test` is a safe-NAF helper: it projects the test out so the negation
 * is range-restricted on `issue` only.)
 */
export function codingLoopRules(): WorkingMemoryOperation[] {
  return [
    {
      op: 'add_axiom',
      id: 'AX_VERIFIED',
      label: "verified when the fix-test passes (the runner's word, not the model's)",
      when: [
        { predicate: 'fix_test', args: { issue: '?i', test: '?t' } },
        { predicate: 'test_result', args: { test: '?t', status: 'pass' } },
      ],
      then: [{ predicate: 'verified', args: { issue: '?i' } }],
    },
    {
      op: 'add_axiom',
      id: 'AX_FIXED',
      label: 'fixed = an edit landed, the fix-test passes, and the cause is understood',
      when: [
        { predicate: 'issue', args: { id: '?i' } },
        { predicate: 'edited', args: { issue: '?i' } },
        { predicate: 'verified', args: { issue: '?i' } },
        { predicate: 'diagnosis', args: { issue: '?i', kind: '?k' } },
      ],
      then: [{ predicate: 'finding', args: { kind: 'fixed', issue: '?i' } }],
    },
    {
      op: 'add_axiom',
      id: 'AX_REGRESSION',
      label: 'a test that passed before but now fails is a regression',
      when: [
        { predicate: 'baseline_pass', args: { test: '?t' } },
        { predicate: 'test_result', args: { test: '?t', status: 'fail' } },
      ],
      then: [{ predicate: 'finding', args: { kind: 'regression', test: '?t' } }],
    },
    {
      op: 'add_axiom',
      id: 'AX_HAS_FIX_TEST',
      label: 'an issue has an acceptance test (projects the test out for safe negation)',
      when: [{ predicate: 'fix_test', args: { issue: '?i', test: '?t' } }],
      then: [{ predicate: 'has_fix_test', args: { issue: '?i' } }],
    },
    {
      op: 'add_axiom',
      id: 'AX_NEEDS_ACCEPTANCE',
      label: 'an issue with no fix-test has no checkable done — declare/propose the test first',
      when: [
        { predicate: 'issue', args: { id: '?i' } },
        { predicate: 'has_fix_test', args: { issue: '?i' }, naf: true },
      ],
      then: [{ predicate: 'needs_acceptance_test', args: { issue: '?i' } }],
    },
    {
      op: 'add_axiom',
      id: 'AX_EDIT_FOR_ISSUE',
      // Bridge from the editor's machine fact to the issue: edit_file attests
      // edited(file, line); the model declares which file an issue lives in via
      // touches(issue, file) (a claim, not a machine fact). The closure then
      // earns edited(issue) — so AX_FIXED's edited(issue) is satisfied by a REAL
      // edit, never by the model asserting edited(issue) directly (it's attested).
      label: 'an edit counts for an issue when the editor touched a file the issue lives in',
      when: [
        { predicate: 'edited', args: { file: '?f' } },
        { predicate: 'touches', args: { issue: '?i', file: '?f' } },
      ],
      then: [{ predicate: 'edited', args: { issue: '?i' } }],
    },
  ] as WorkingMemoryOperation[]
}

/**
 * codingMultiFileRules — the MULTI-FILE coding pack for a cross-file change
 * (schema → callers → tests). codingLoopRules covers a single file via the
 * edited(issue) bridge; a real repair often touches several files with an
 * ordering dependency, and "done" must mean EVERY file the issue touches was
 * edited — not just the one the model felt like editing.
 *
 * Extra predicate family (on top of issue/fix_test/diagnosis/test_result):
 *   touches(issue, file)      — the model's plan: a file this issue must edit.
 *   file_requires(file, dep)  — the model's ordering claim: `file` may only be
 *                               edited after `dep` (e.g. caller requires schema).
 *   edited(file)              — MACHINE FACT (attested by edit_file/write_file).
 *   baseline_pass(test)       — MACHINE FACT (harness pre-edit baseline).
 *
 * Derived:
 *   has_fix_test / needs_acceptance_test          — the intake gate (as single-file).
 *   edit_blocked(file)  <- file_requires(file,dep) AND NOT edited(dep)
 *   edit_ready(file)    <- touches(_,file) AND NOT edited(file) AND NOT edit_blocked(file)
 *                          — the ORDERED editable frontier the model should follow.
 *   has_touch(issue)    <- touches(issue,_)          — the issue declared a scope.
 *   file_pending(issue) <- touches(issue,file) AND NOT edited(file)   — grounded NAF witness.
 *   all_files_edited(issue) <- issue AND has_touch(issue) AND NOT file_pending(issue)
 *                          — COMPLETENESS: every touched file is edited, AND the
 *                            issue actually declared ≥1 file (so an issue that
 *                            touches nothing can never be vacuously "all edited").
 *   verified(issue)     <- fix_test(issue,test) AND test_result(test,pass)
 *   finding(kind=fixed, issue) <- issue AND all_files_edited(issue) AND verified(issue)
 *                                 AND diagnosis(issue,_)
 *   finding(kind=regression, test) <- baseline_pass(test) AND test_result(test,fail)
 *
 * The cross-file REVIEW is the completeness gate: the closure refuses
 * all_files_edited (hence finding(fixed)) while any touched file is unedited — a
 * model cannot change the schema, forget the callers, and still earn `fixed`.
 */
export function codingMultiFileRules(): WorkingMemoryOperation[] {
  return [
    {
      op: 'add_axiom',
      id: 'AX_HAS_FIX_TEST',
      label: 'an issue has an acceptance test (projects the test out for safe negation)',
      when: [{ predicate: 'fix_test', args: { issue: '?i', test: '?t' } }],
      then: [{ predicate: 'has_fix_test', args: { issue: '?i' } }],
    },
    {
      op: 'add_axiom',
      id: 'AX_NEEDS_ACCEPTANCE',
      label: 'an issue with no fix-test has no checkable done — declare/propose the test first',
      when: [
        { predicate: 'issue', args: { id: '?i' } },
        { predicate: 'has_fix_test', args: { issue: '?i' }, naf: true },
      ],
      then: [{ predicate: 'needs_acceptance_test', args: { issue: '?i' } }],
    },
    {
      op: 'add_axiom',
      id: 'AX_EDIT_BLOCKED',
      label: 'a file whose required dependency is not yet edited may not be edited',
      when: [
        { predicate: 'file_requires', args: { file: '?f', dep: '?d' } },
        { predicate: 'edited', args: { file: '?d' }, naf: true },
      ],
      then: [{ predicate: 'edit_blocked', args: { file: '?f' } }],
    },
    {
      op: 'add_axiom',
      id: 'AX_EDIT_READY',
      label: 'a touched, unedited, unblocked file is the editable frontier',
      when: [
        { predicate: 'touches', args: { issue: '?i', file: '?f' } },
        { predicate: 'edited', args: { file: '?f' }, naf: true },
        { predicate: 'edit_blocked', args: { file: '?f' }, naf: true },
      ],
      then: [{ predicate: 'edit_ready', args: { file: '?f' } }],
    },
    {
      op: 'add_axiom',
      id: 'AX_HAS_TOUCH',
      label: 'an issue that named at least one file to edit has a declared scope',
      when: [{ predicate: 'touches', args: { issue: '?i', file: '?f' } }],
      then: [{ predicate: 'has_touch', args: { issue: '?i' } }],
    },
    {
      op: 'add_axiom',
      id: 'AX_FILE_PENDING',
      label: 'grounded witness: a file the issue touches is not yet edited',
      when: [
        { predicate: 'touches', args: { issue: '?i', file: '?f' } },
        { predicate: 'edited', args: { file: '?f' }, naf: true },
      ],
      then: [{ predicate: 'file_pending', args: { issue: '?i' } }],
    },
    {
      op: 'add_axiom',
      id: 'AX_ALL_FILES_EDITED',
      label: 'every file the issue touches has been edited (and it touched at least one)',
      when: [
        { predicate: 'issue', args: { id: '?i' } },
        { predicate: 'has_touch', args: { issue: '?i' } },
        { predicate: 'file_pending', args: { issue: '?i' }, naf: true },
      ],
      then: [{ predicate: 'all_files_edited', args: { issue: '?i' } }],
    },
    {
      op: 'add_axiom',
      id: 'AX_VERIFIED',
      label: "verified when the fix-test passes (the runner's word, not the model's)",
      when: [
        { predicate: 'fix_test', args: { issue: '?i', test: '?t' } },
        { predicate: 'test_result', args: { test: '?t', status: 'pass' } },
      ],
      then: [{ predicate: 'verified', args: { issue: '?i' } }],
    },
    {
      op: 'add_axiom',
      id: 'AX_FIXED',
      label: 'fixed = every touched file edited, the fix-test passes, and the cause is understood',
      when: [
        { predicate: 'issue', args: { id: '?i' } },
        { predicate: 'all_files_edited', args: { issue: '?i' } },
        { predicate: 'verified', args: { issue: '?i' } },
        { predicate: 'diagnosis', args: { issue: '?i', kind: '?k' } },
      ],
      then: [{ predicate: 'finding', args: { kind: 'fixed', issue: '?i' } }],
    },
    {
      op: 'add_axiom',
      id: 'AX_REGRESSION',
      label: 'a test that passed before but now fails is a regression',
      when: [
        { predicate: 'baseline_pass', args: { test: '?t' } },
        { predicate: 'test_result', args: { test: '?t', status: 'fail' } },
      ],
      then: [{ predicate: 'finding', args: { kind: 'regression', test: '?t' } }],
    },
  ] as WorkingMemoryOperation[]
}
