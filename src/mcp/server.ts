import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { z } from 'zod'
import { formatAtom, formatAtoms as formatAtomList } from '../kernel/predicate.js'
import type { SpaceStore } from '../storage/space-store.js'
import { deriveActionEffects } from '../engine/semantic-derivation.js'
import { simulateActionEffects } from '../engine/simulate.js'
import { validatePlan } from '../engine/validate-plan.js'
import { applyPlan } from '../engine/apply-plan.js'
import { suggestPlanRepairs } from '../engine/plan-repair.js'
import { planToGoal } from '../engine/plan-search.js'
import { solveConstraintsOnBoard } from '../engine/constraint-solve.js'
import { explain, formatExplanation } from '../engine/explain.js'
import { distillSpace, seedSpace } from '../engine/distill.js'
import { formatLogicContextAsText, getLogicContext } from '../engine/logic-context.js'
import {
  applyWorkingMemoryOperations,
  type ApplyOptions,
  type WorkingMemoryOperation,
} from '../engine/working-memory.js'

/** Attestation config for the MCP face: an external MCP agent is model-sourced,
 *  so the same machine-attested guards the task-loop applies must apply here too. */
export type McpServerOptions = {
  attestedPredicates?: string[]
  attestedDerivations?: ApplyOptions['attestedDerivations']
}

const semanticScalarSchema = z.union([z.string(), z.number(), z.boolean()])
const semanticArgsSchema = z.record(z.string(), semanticScalarSchema)

const atomSchema = z.object({
  predicate: z.string(),
  args: semanticArgsSchema.optional(),
  negated: z.boolean().optional().describe('Strong negation: "verified that not p".'),
  naf: z
    .boolean()
    .optional()
    .describe('Negation as failure ("p cannot be proven"). Rule bodies and action preconditions - the way to require ABSENCE of a fact.'),
})

const operationSchema = z.discriminatedUnion('op', [
  z.object({
    op: z.literal('declare_goal'),
    id: z.string().optional(),
    label: z.string(),
    summary: z.string().optional(),
    desired: z.array(atomSchema),
  }),
  z.object({
    op: z.literal('assert_fact'),
    id: z.string().optional(),
    label: z.string().optional(),
    summary: z.string().optional(),
    predicate: z.string(),
    args: semanticArgsSchema.optional(),
    negated: z.boolean().optional(),
    evidenceRefs: z
      .array(z.string())
      .optional()
      .describe('Provenance, e.g. ["tool:grep"] or ids of supporting entries.'),
  }),
  z.object({
    op: z.literal('declare_hypothesis'),
    id: z.string().optional(),
    label: z.string().optional(),
    summary: z.string().optional(),
    predicate: z.string(),
    args: semanticArgsSchema.optional(),
    negated: z.boolean().optional(),
  }),
  z.object({
    op: z.literal('add_axiom'),
    id: z.string().optional(),
    label: z.string(),
    summary: z.string().optional(),
    when: z.array(atomSchema),
    then: z.array(atomSchema),
  }),
  z.object({
    op: z.literal('derive_aggregate'),
    id: z.string().optional(),
    label: z.string().optional(),
    summary: z.string().optional(),
    kind: z
      .enum(['sum', 'count', 'min', 'max', 'avg'])
      .optional()
      .describe('Default "sum". sum/min/max/avg need source.valueArg (avg = sum/count, IEEE); count needs none.'),
    source: z.object({ predicate: z.string(), valueArg: z.string().optional() }),
    into: z.object({ predicate: z.string(), valueArg: z.string() }),
    where: z
      .object({ arg: z.string(), equals: z.union([z.string(), z.number(), z.boolean()]) })
      .optional()
      .describe('Optional equality filter: only aggregate source facts whose args[arg] === equals.'),
    group_by: z
      .string()
      .optional()
      .describe('Optional grouping: one into-fact per distinct args[group_by] value (carried into the result). Composes with where.'),
  }),
  z.object({
    op: z.literal('define_action'),
    id: z.string().optional(),
    label: z.string().optional().describe('Defaults to the action name.'),
    summary: z.string().optional(),
    action: z.string(),
    preconditions: z.array(atomSchema).optional(),
    effects: z
      .array(atomSchema)
      .optional()
      .describe('Positive atoms are asserted; negated atoms delete the matching fact.'),
  }),
  z.object({
    op: z.literal('record_result'),
    id: z.string().optional(),
    label: z.string(),
    summary: z.string().optional(),
    evidenceRefs: z
      .array(z.string())
      .optional()
      .describe('Current board node ids this conclusion rests on. Must resolve to facts/findings/axioms/goals/actions/hypotheses/conflicts from get_logic_context.'),
  }),
  z.object({
    op: z.literal('record_conflict'),
    id: z.string().optional(),
    label: z.string(),
    summary: z.string().optional(),
    evidenceRefs: z.array(z.string()).optional(),
  }),
  z.object({
    op: z.literal('retract_node'),
    nodeId: z.string(),
    reason: z.string().optional(),
  }),
  z.object({
    op: z.literal('revise_fact'),
    nodeId: z.string(),
    reason: z.string().optional(),
    id: z.string().optional(),
    label: z.string().optional(),
    summary: z.string().optional(),
    predicate: z.string(),
    args: semanticArgsSchema.optional(),
    negated: z.boolean().optional(),
    evidenceRefs: z.array(z.string()).optional(),
  }),
])

const PROTOCOL_NOTES = [
  'Predicate conventions: use the same predicate name and argument keys every round',
  '(check the vocabulary section of the working memory; signature mismatches return warnings).',
  'Variables are strings starting with "?" - legal in rule bodies/heads and in action',
  'preconditions/effects (effect variables must be bound by preconditions). Never in facts.',
  'Comparison built-ins eq/neq/lt/lte/gt/gte(left, right), between(value, low, high) [closed interval], and contains(left, right) [string substring] may appear in rule bodies.',
  'Arithmetic built-ins add/sub/mul/div/mod/idiv/imod/pow/min/max(left, right, result) and neg/abs/sqrt/ln/exp(left, result)',
  '(idiv=floor division; imod=mathematical modulo, non-negative for positive divisors; mod=JS remainder keeping dividend sign; sqrt/ln/exp=transcendental, IEEE best-effort, FAIL on domain/overflow e.g. sqrt(-1)/ln(0))',
  'String producer concat(left, right, result) joins two string/integer parts into a string (integers stringified canonically: 5->"5"); chain for 3+ parts: concat(a,b,?ab) then concat(?ab,c,?abc). Floats/booleans/out-of-range integers throw a teaching error.',
  'compute EXACT values in rule bodies and bind result - do not do arithmetic in your head, let the',
  'board derive it (the product becomes a derived fact with an evidence chain). Copyable template:',
  '{"op":"add_axiom","id":"ax_cost","label":"cost = unit*qty","when":[{"predicate":"line","args":{"item":"?i","unit":"?u","qty":"?q"}},{"predicate":"mul","args":{"left":"?u","right":"?q","result":"?t"}}],"then":[{"predicate":"cost","args":{"item":"?i","total":"?t"}}]}',
  'Totals WITHOUT hand-writing a chain rule - the engine expands this to ONE exact chain rule over the facts present now (re-run it after facts change; same id replaces the old rule):',
  '{"op":"derive_aggregate","id":"agg_total","source":{"predicate":"cost","valueArg":"total"},"into":{"predicate":"grand_total","valueArg":"value"}}',
  'kind:"count" needs no valueArg. NEVER write a rule whose head predicate appears in its own body to accumulate - a monotonic closure cannot run loops; use derive_aggregate.',
  'kind:"min"/"max" need source.valueArg (like sum) and fold to the smallest/largest value: {"op":"derive_aggregate","id":"agg_min","kind":"min","source":{"predicate":"cost","valueArg":"total"},"into":{"predicate":"cheapest","valueArg":"value"}}',
  'group_by buckets by a distinct arg value, one result fact per group (group value carried into the result): {"op":"derive_aggregate","id":"agg_region","source":{"predicate":"sale","valueArg":"amount"},"into":{"predicate":"region_total","valueArg":"value"},"group_by":"region"}',
  'Aggregate only a subset with an optional where (facts whose arg equals the value):',
  '{"op":"derive_aggregate","id":"agg_east","source":{"predicate":"sale","valueArg":"amount"},"into":{"predicate":"east_total","valueArg":"value"},"where":{"arg":"region","equals":"east"}}',
  'Built-ins go in rule WHEN bodies only - never in then, never in action effects.',
  'For consume/produce transformations use define_action (negated effect = consume, positive = produce)',
  'then apply_action - rules are monotonic and never delete; actions do.',
  'For COUNTED amounts (stoichiometry, budgets, inventory): bind the current amount in a precondition,',
  'guard with gte, compute the new amount with sub/add (also in preconditions), then swap the amount',
  'fact in the effects (negated old + positive new). Consumed facts are archived with an event record,',
  'not destroyed. Copyable counted-action template:',
  '{"op":"define_action","id":"burn1","action":"consume_two","preconditions":[{"predicate":"amount","args":{"species":"H2","mol":"?h"}},{"predicate":"gte","args":{"left":"?h","right":2}},{"predicate":"sub","args":{"left":"?h","right":2,"result":"?h2"}}],"effects":[{"predicate":"amount","args":{"species":"H2","mol":"?h"},"negated":true},{"predicate":"amount","args":{"species":"H2","mol":"?h2"}}]}',
  'negated:true is STRONG negation (an explicit not-p fact; in effects it deletes the matching fact).',
  'naf:true means "cannot be proven" and is the way to require ABSENCE in rule bodies and action',
  'preconditions - do not use negated for absence checks.',
  'Open goals list "producible via action X" when a defined action could produce the missing atom -',
  'simulate/apply that action instead of asserting the product yourself.',
  'To commit a multi-step plan, prefer apply_plan over a chain of apply_action: it validates the whole',
  'ordered list on a clone first (nothing commits if it fails), then runs each step drift-guarded.',
  'Set requireGoals:true to refuse a plan that runs but does not reach every goal.',
  'To correct a mistake use retract_node or revise_fact - never assert a contradicting',
  'fact on top. Retraction physically removes the entry and everything resting on it,',
  'then re-derives whatever is still supported.',
  'Findings must be derived, not asserted: record_result is rejected while any positive',
  'finding(...) fact is asserted rather than rule-derived. Assert the primitive observation',
  'you verified, add a rule deriving finding(...) from it, and let the closure produce the finding.',
  'record_result.evidenceRefs are machine-checked: cite exact current node ids from get_logic_context',
  '(facts, findings, axioms, goals, actions, hypotheses, or conflicts). Do not cite prose labels,',
  'external file references, invented ids, or other result nodes; assert/derive missing evidence first.',
  'To SOLVE a constraint problem (pick a value per variable so nothing is violated) let the board search:',
  'add_axiom a rule deriving conflict(...) over the bridge facts assignment(var,value), then call the solve',
  'tool with the variables and their finite domains - the board searches, the closure certifies each',
  'candidate, and a satisfying assignment is committed as assignment(var,value) facts (else unsat/budget).',
  'Do not assign the values yourself. Template: {"variables":[{"name":"M1","domain":["s1","s2"]},{"name":"M2","domain":["s1","s2"]}],"conflictPredicate":"conflict"}',
].join(' ')

export function createRulithCoreMcpServer(store: SpaceStore, options: McpServerOptions = {}): McpServer {
  const server = new McpServer({ name: 'rulith', version: '0.2.0' })

  server.registerTool(
    'create_space',
    {
      title: 'create_space',
      description:
        'Create a working-memory space for one task. Do this once per task, then drive everything through update_working_memory. ' +
        'Pass seedFromSpaceId to replant the verified rules and predicate vocabulary of a finished space - reuse experience and keep predicate names consistent across tasks.',
      inputSchema: {
        id: z.string().optional(),
        title: z.string(),
        scopes: z.array(z.string()).optional(),
        seedFromSpaceId: z.string().optional(),
      },
    },
    guard(async ({ seedFromSpaceId, ...input }) => {
      const space = store.createSpace(input)
      const lines = [`created space ${space.id}: ${space.title}`]
      if (seedFromSpaceId) {
        const seeded = seedSpace(store, space.id, distillSpace(store, seedFromSpaceId))
        lines.push(
          `seeded ${seeded.seededAxiomIds.length} rule(s) from ${seedFromSpaceId}: ${seeded.seededAxiomIds.join(', ') || 'none'}`,
          `inherited vocabulary: ${seeded.vocabulary.join(', ') || 'none'}`,
        )
      }
      return textResult(lines.join('\n'))
    }),
  )

  server.registerTool(
    'distill_space',
    {
      title: 'distill_space',
      description:
        'Distill a finished space into an experience capsule: its verified rules, recorded conclusions, and predicate vocabulary. ' +
        'Task-specific facts and hypotheses are not included. Use the capsule to review what was learned, or seed a new space via create_space.seedFromSpaceId.',
      inputSchema: {
        spaceId: z.string(),
      },
    },
    guard(async ({ spaceId }: { spaceId: string }) => {
      return textResult(JSON.stringify(distillSpace(store, spaceId), null, 2))
    }),
  )

  server.registerTool(
    'list_spaces',
    {
      title: 'list_spaces',
      description: 'List existing working-memory spaces.',
      inputSchema: {},
    },
    guard(async () => {
      const spaces = store.listSpaces()
      if (spaces.length === 0) return textResult('no spaces')
      return textResult(spaces.map((space) => `${space.id}: ${space.title}`).join('\n'))
    }),
  )

  server.registerTool(
    'update_working_memory',
    {
      title: 'update_working_memory',
      description:
        'Submit a batch of working-memory operations (declare_goal, assert_fact, declare_hypothesis, add_axiom, derive_aggregate, define_action, record_result, record_conflict, retract_node, revise_fact). ' +
        'The kernel applies them, recomputes the rule closure once, and returns the updated working memory: derived facts, goal satisfaction, ' +
        'hypothesis verdicts (open/supported/refuted), and "needs via <rule>: ..." hints that tell you exactly which facts to observe next. ' +
        'Read the returned state instead of calling get_logic_context again. ' +
        PROTOCOL_NOTES,
      inputSchema: {
        spaceId: z.string(),
        operations: z.array(operationSchema),
      },
    },
    guard(async ({ spaceId, operations }: { spaceId: string; operations: unknown[] }) => {
      // An external MCP agent is MODEL-sourced - apply the same machine-attested
      // guards as the task-loop, so the MCP face is not a bypass (Codex review P1).
      const result = applyWorkingMemoryOperations(
        store,
        spaceId,
        operations as WorkingMemoryOperation[],
        {
          format: 'text',
          source: 'model',
          attestedPredicates: options.attestedPredicates,
          attestedDerivations: options.attestedDerivations,
        },
      )
      const lines: string[] = []
      if (result.warnings.length > 0) {
        lines.push('warnings:', ...result.warnings.map((warning) => `- ${warning}`), '')
      }
      const closure = result.semanticRuleApplication
      lines.push(
        `applied ${operations.length} operation(s); derived +${closure.addedFactNodeIds.length}/-${closure.removedFactNodeIds.length} fact(s)`,
        '',
        result.workingMemoryText ?? '',
      )
      return textResult(lines.join('\n'))
    }),
  )

  server.registerTool(
    'simulate_action',
    {
      title: 'simulate_action',
      description:
        'Try a defined action WITHOUT committing it: returns which facts it would add/remove, which derived conclusions would appear or disappear, ' +
        'which goals it would satisfy, and how hypothesis verdicts would change. Nothing is written. ' +
        'Use this to compare candidate actions side by side, then commit the chosen one with apply_action.',
      inputSchema: {
        spaceId: z.string(),
        actionNodeId: z.string(),
      },
    },
    guard(async ({ spaceId, actionNodeId }: { spaceId: string; actionNodeId: string }) => {
      const result = simulateActionEffects(store, spaceId, actionNodeId)
      if (!result.applicable) {
        return textResult(
          [
            `simulate ${actionNodeId}: NOT applicable`,
            result.failedPrecondition
              ? `first failing precondition: ${formatAtom(result.failedPrecondition)}`
              : '',
            nafHint(result.failedPrecondition),
            `missing facts: ${result.unsatisfiedPreconditions.map(formatAtom).join(' AND ') || 'none (a guard/arithmetic literal failed)'}`,
          ]
            .filter(Boolean)
            .join('\n'),
        )
      }
      return textResult(
        [
          `simulate ${actionNodeId}: applicable`,
          bindingLine(result.binding, result.bindingCandidates),
          `would assert: ${formatAtoms(result.addedAtoms)}`,
          `would delete: ${formatAtoms(result.removedAtoms)}`,
          `new derived: ${formatAtoms(result.newDerivedAtoms)}`,
          `lost derived: ${formatAtoms(result.lostDerivedAtoms)}`,
          `would satisfy goals: ${result.wouldSatisfyGoalIds.join(', ') || 'none'}`,
          `hypothesis verdicts: ${
            result.hypothesisVerdicts.map((v) => `${v.nodeId}=${v.status}`).join(', ') || 'none'
          }`,
          `predicate conflicts: ${result.predicateConflicts.length}`,
          `boardRevision: ${result.boardRevision}  (pass to apply_action.expectedRevision to apply only if the board has not changed)`,
        ]
          .filter(Boolean)
          .join('\n'),
      )
    }),
  )

  server.registerTool(
    'validate_plan',
    {
      title: 'validate_plan',
      description:
        'Dry-run a WHOLE ordered list of actions on a throwaway clone (the real board is untouched). ' +
        'Reports the first step whose preconditions fail (with the unmet precondition), whether all ' +
        'goals end satisfied, and the SHORTEST PREFIX of the plan that already reaches every goal ' +
        '(so you can prune the redundant trailing steps). Use it to check a candidate plan before ' +
        'committing any step: the board grounds the plan, you do not have to trace the ordering by hand.',
      inputSchema: {
        spaceId: z.string(),
        actionNodeIds: z.array(z.string()).describe('The plan: action node ids in intended execution order.'),
      },
    },
    guard(async ({ spaceId, actionNodeIds }: { spaceId: string; actionNodeIds: string[] }) => {
      const v = validatePlan(store, spaceId, actionNodeIds)
      const lines = [
        `validate_plan: ${v.ok ? 'OK - every step applies and all goals are reached' : 'NOT OK'}`,
        ...v.steps.map((step) =>
          step.applicable
            ? `  #${step.index} ${step.actionNodeId}: applies`
            : `  #${step.index} ${step.actionNodeId}: BLOCKED${
                step.error ? ` (${step.error})` : ''
              }${step.failedPrecondition ? ` - first unmet: ${formatAtom(step.failedPrecondition)}` : ''}`,
        ),
      ]
      if (v.firstFailureIndex !== undefined) {
        lines.push(
          `first failure at step #${v.firstFailureIndex} - the plan stops there; fix the ordering ` +
            `or the missing precondition and re-validate.`,
        )
      }
      if (v.unmetGoalIds.length > 0) {
        lines.push(`goals NOT reached: ${v.unmetGoalIds.join(', ')} (satisfied: ${v.satisfiedGoalIds.join(', ') || 'none'})`)
      }
      if (v.shortestPrefixLength !== undefined && v.redundantStepIndices.length > 0) {
        lines.push(
          `shortest prefix: the first ${v.shortestPrefixLength} step(s) already reach every goal; ` +
            `steps #${v.redundantStepIndices.join(', #')} are redundant - drop them.`,
        )
      }
      return textResult(lines.join('\n'))
    }),
  )

  server.registerTool(
    'suggest_plan_repairs',
    {
      title: 'suggest_plan_repairs',
      description:
        'When validate_plan reports a broken plan, propose how to fix it: search the board\'s OWN defined ' +
        'actions for one whose effect produces the unmet precondition, chasing that producer\'s own unmet ' +
        'preconditions multi-hop (bounded). Returns copyable repaired sequences, each RE-VALIDATED on a ' +
        'clone first - it never mutates the board and never auto-applies. Re-run validate_plan/apply_plan ' +
        'on a candidate before trusting it. Suggestion only; not a general planner.',
      inputSchema: {
        spaceId: z.string(),
        actionNodeIds: z.array(z.string()).describe('The broken plan: action node ids in intended order.'),
        maxDepth: z.number().optional().describe('Max producer-chain hops (default 5).'),
        maxActions: z.number().optional().describe('Max actions inserted per repair (default = maxDepth).'),
      },
    },
    guard(
      async ({
        spaceId,
        actionNodeIds,
        maxDepth,
        maxActions,
      }: {
        spaceId: string
        actionNodeIds: string[]
        maxDepth?: number
        maxActions?: number
      }) => {
        const r = suggestPlanRepairs(store, spaceId, actionNodeIds, undefined, { maxDepth, maxActions })
        if (r.repairs.length === 0) {
          return textResult(`suggest_plan_repairs: no repair${r.note ? ` - ${r.note}` : ''}`)
        }
        const lines = [
          `suggest_plan_repairs: ${r.repairs.length} candidate(s) for the gap at step #${r.failedIndex}` +
            (r.failedPrecondition ? ` (unmet: ${formatAtom(r.failedPrecondition)})` : ''),
          ...r.repairs.map(
            (rep, i) =>
              `  [${i}]${rep.validates ? ' (reaches goal)' : ' (runs past failure)'} insert ` +
              `${rep.insertedActionNodeIds.join(' -> ')}: try {"actionNodeIds":${JSON.stringify(rep.actionNodeIds)}}`,
          ),
          're-run validate_plan / apply_plan on a candidate before trusting it.',
        ]
        return textResult(lines.join('\n'))
      },
    ),
  )

  server.registerTool(
    'plan_to_goal',
    {
      title: 'plan_to_goal',
      description:
        'Let the BOARD search its defined actions for an ordered plan that reaches the declared goals, ' +
        'instead of proposing one yourself. Bounded forward search (depth + beam) on throwaway clones; ' +
        'every candidate is re-checked by validate_plan before it is returned, and the real board is never ' +
        'mutated. Returns a copyable plan to commit with apply_plan, or a note if none is found in the bound. ' +
        'Suggestion only - the board proposes, you apply.',
      inputSchema: {
        spaceId: z.string(),
        maxDepth: z.number().optional().describe('Max plan length to search (default 8).'),
        maxBeam: z.number().optional().describe('Max states kept per search level (default 16).'),
      },
    },
    guard(async ({ spaceId, maxDepth, maxBeam }: { spaceId: string; maxDepth?: number; maxBeam?: number }) => {
      const r = planToGoal(store, spaceId, { maxDepth, maxBeam })
      if (!r.found) return textResult(`plan_to_goal: no plan${r.note ? ` - ${r.note}` : ''}`)
      return textResult(
        `plan_to_goal: found a validated plan reaching every goal.\n` +
          `apply it: {"actionNodeIds":${JSON.stringify(r.plan)}}`,
      )
    }),
  )

  server.registerTool(
    'solve',
    {
      title: 'solve',
      description:
        'Finite-domain constraint SOLVER: choose a value for each variable so the closure derives NO conflict. ' +
        'First add_axiom a rule deriving the conflict predicate over the bridge facts assignment(var,value); then call solve. ' +
        'The board SEARCHES the domains, the closure adjudicates every candidate, and a satisfying assignment is committed as ' +
        'assignment(var,value) facts (then closure-re-verified) - or it reports unsat/budget. Do not assign the values ' +
        'yourself: search proposes, the closure certifies. ' +
        'Template: {"variables":[{"name":"M1","domain":["s1","s2"]},{"name":"M2","domain":["s1","s2"]}],"conflictPredicate":"conflict"}',
      inputSchema: {
        spaceId: z.string(),
        variables: z
          .array(
            z.object({
              name: z.string(),
              domain: z.array(z.union([z.string(), z.number(), z.boolean()])).min(1),
            }),
          )
          .min(1),
        conflictPredicate: z.string().optional().describe('Predicate whose derivation marks a violation (default "conflict").'),
        assignPredicate: z.string().optional().describe('Bridge predicate carrying a chosen value (default "assignment").'),
        maxNodes: z.number().optional().describe('Search budget in assignment attempts.'),
      },
    },
    guard(
      async ({
        spaceId,
        variables,
        conflictPredicate,
        assignPredicate,
        maxNodes,
      }: {
        spaceId: string
        variables: { name: string; domain: (string | number | boolean)[] }[]
        conflictPredicate?: string
        assignPredicate?: string
        maxNodes?: number
      }) => {
        const r = solveConstraintsOnBoard(store, spaceId, { variables, conflictPredicate, assignPredicate, maxNodes })
        const cp = conflictPredicate ?? 'conflict'
        if (r.sat) {
          const pairs = Object.entries(r.assignment).map(([k, v]) => `${k}=${String(v)}`).join(', ')
          return textResult(`solved (closure-certified, committed): ${pairs} [searched ${r.nodes}]`)
        }
        if (r.reason === 'unsat') {
          return textResult(`unsat: no assignment over the given domains avoids ${cp} (searched ${r.nodes}).`)
        }
        return textResult(`budget: hit the node cap (searched ${r.nodes}); raise maxNodes or shrink the problem.`)
      },
    ),
  )

  server.registerTool(
    'apply_action',
    {
      title: 'apply_action',
      description:
        'Commit a defined action: asserts its positive effects as facts, deletes facts matched by negated effects, then recomputes the closure. ' +
        'Fails softly when preconditions are unsatisfied (returns the gap, writes nothing). ' +
        'Applying an action is a decision - prefer simulate_action first when choosing between candidates.',
      inputSchema: {
        spaceId: z.string(),
        actionNodeId: z.string(),
        expectedRevision: z
          .string()
          .optional()
          .describe('From simulate_action: apply only if the board still matches, else fail.'),
      },
    },
    guard(
      async ({
        spaceId,
        actionNodeId,
        expectedRevision,
      }: {
        spaceId: string
        actionNodeId: string
        expectedRevision?: string
      }) => {
        const result = deriveActionEffects(store, spaceId, actionNodeId, { expectedRevision })
      if (!result.applied) {
        return textResult(
          [
            `apply ${actionNodeId}: NOT applied`,
            result.failedPrecondition
              ? `first failing precondition: ${formatAtom(result.failedPrecondition)}`
              : '',
            nafHint(result.failedPrecondition),
            `missing facts: ${result.unsatisfiedPreconditions.map(formatAtom).join(' AND ') || 'none (a guard/arithmetic literal failed)'}`,
          ]
            .filter(Boolean)
            .join('\n'),
        )
      }
      const context = formatLogicContextAsText(getLogicContext(store, spaceId))
      return textResult(
        [
          `applied ${actionNodeId}: +${result.addedFactNodeIds.length}/-${result.removedFactNodeIds.length} fact(s); satisfied goals: ${result.satisfiedGoalNodeIds.join(', ') || 'none'}`,
          bindingLine(result.binding, result.bindingCandidates),
          '',
          context,
        ]
          .filter((line, index) => line !== '' || index === 2)
          .join('\n'),
      )
    }),
  )

  server.registerTool(
    'apply_plan',
    {
      title: 'apply_plan',
      description:
        'Commit a WHOLE ordered plan in one call: validate the sequence on a clone first, then apply ' +
        'each step to the real board, every step pinned to the revision the previous step produced ' +
        '(so a concurrent change is caught, never silently applied). If the plan does not validate, ' +
        'NOTHING is committed. If a step fails mid-commit (board drift), it stops there and reports ' +
        'how far it got - the steps before it stay committed. Set requireGoals=true to refuse a plan ' +
        'that runs but does not reach every goal. This is validate_plan + a guarded run of apply_action.',
      inputSchema: {
        spaceId: z.string(),
        actionNodeIds: z.array(z.string()).describe('The plan: action node ids in intended execution order.'),
        requireGoals: z
          .boolean()
          .optional()
          .describe('When true, commit only if the plan reaches every declared goal (default: false).'),
      },
    },
    guard(
      async ({
        spaceId,
        actionNodeIds,
        requireGoals,
      }: {
        spaceId: string
        actionNodeIds: string[]
        requireGoals?: boolean
      }) => {
        const r = applyPlan(store, spaceId, actionNodeIds, { requireGoals })
        if (!r.applied) {
          return textResult(
            [
              `apply_plan: NOT applied${r.failedIndex !== undefined ? ` (stopped at step #${r.failedIndex})` : ''}`,
              r.failureReason ?? '',
              r.appliedActionNodeIds.length > 0
                ? `committed before the failure: ${r.appliedActionNodeIds.join(', ')}`
                : 'nothing was committed.',
              r.validation.unmetGoalIds.length > 0
                ? `goals not reached: ${r.validation.unmetGoalIds.join(', ')}`
                : '',
            ]
              .filter(Boolean)
              .join('\n'),
          )
        }
        const context = formatLogicContextAsText(getLogicContext(store, spaceId))
        return textResult(
          [
            `apply_plan: applied ${r.appliedActionNodeIds.length} step(s): ${r.appliedActionNodeIds.join(' -> ') || 'none'}`,
            `final revision: ${r.finalRevision}`,
            '',
            context,
          ].join('\n'),
        )
      },
    ),
  )

  server.registerTool(
    'board_critique',
    {
      title: 'board_critique',
      description:
        'Review the board for STANDING problems, re-derived from current state (not just the last op): ' +
        'goals satisfied only by a bare assertion, open goals nothing can derive or produce, findings ' +
        'asserted instead of derived, vacuous rules whose body just renames the head, predicate ' +
        'contradictions, actions that can never become applicable, and declared goals that desire ' +
        'mutually contradictory atoms (conflicting_goals). Use it to check your own work - ' +
        'an empty critique means the board is healthy.',
      inputSchema: {
        spaceId: z.string(),
      },
    },
    guard(async ({ spaceId }: { spaceId: string }) => {
      const items = getLogicContext(store, spaceId).critique
      if (items.length === 0) return textResult('critique: none - the board is healthy.')
      return textResult(
        ['critique:', ...items.map((item) => `- [${item.kind}] ${item.nodeId}: ${item.message}`)].join('\n'),
      )
    }),
  )

  server.registerTool(
    'explain_fact',
    {
      title: 'explain_fact',
      description:
        'Explain WHY a derived fact is true: returns the rule that derived it and walks its supporting ' +
        'facts recursively back to the asserted/effect leaves, each marked [derived]/[asserted]/[effect]. ' +
        'Use it to audit a conclusion - an asserted fact returns a single leaf, an unknown id is a teaching error.',
      inputSchema: {
        spaceId: z.string(),
        factNodeId: z.string().describe('The fact node id to explain (e.g. a derived: fact id).'),
      },
    },
    guard(async ({ spaceId, factNodeId }: { spaceId: string; factNodeId: string }) => {
      try {
        return textResult(formatExplanation(explain(store, spaceId, factNodeId)))
      } catch (error) {
        return textResult(`explain_fact: ${error instanceof Error ? error.message : String(error)}`)
      }
    }),
  )

  server.registerTool(
    'get_logic_context',
    {
      title: 'get_logic_context',
      description:
        'Read the current working memory: goals (with satisfied flag and missing-fact hints), facts, hypotheses (with hints), findings, ' +
        'axioms, actions, results, conflicts, and the predicate vocabulary. ' +
        'Write operations already return this - call it only when you need to re-read the state.',
      inputSchema: {
        spaceId: z.string(),
      },
    },
    guard(async ({ spaceId }: { spaceId: string }) => {
      return textResult(formatLogicContextAsText(getLogicContext(store, spaceId)))
    }),
  )

  return server
}

/** Teach naf-vs-negated at the moment a strong-negated literal fails. */
function nafHint(atom?: Parameters<typeof formatAtom>[0]): string {
  if (!atom || atom.negated !== true) return ''
  const positive = formatAtom({ ...atom, negated: undefined })
  return (
    `hint: "negated":true is STRONG negation - it only matches an explicit not-${positive} fact. ` +
    `To require the ABSENCE of ${positive}, use "naf":true instead.`
  )
}

function formatAtoms(atoms: Array<Parameters<typeof formatAtom>[0]>): string {
  return formatAtomList(atoms, ', ')
}

/** Render the precondition binding an action ran (or would run) under. */
function bindingLine(binding: Record<string, unknown>, candidates: number): string {
  const text = Object.entries(binding)
    .map(([name, value]) => `?${name}=${String(value)}`)
    .join(', ')
  if (!text) return ''
  const ambiguity =
    candidates > 1
      ? ` (WARNING: ${candidates} candidate bindings matched - the first is used; add preconditions to pin the instance you mean)`
      : ''
  return `binding: ${text}${ambiguity}`
}

type ToolResult = { content: Array<{ type: 'text'; text: string }>; isError?: boolean }

function textResult(text: string): ToolResult {
  return { content: [{ type: 'text', text }] }
}

/**
 * Errors become structured tool results instead of protocol failures, so
 * the model can read the message (e.g. RuleSafetyError violations,
 * unknown node ids) and self-correct.
 */
function guard<T>(handler: (input: T) => ToolResult | Promise<ToolResult>): (input: T) => Promise<ToolResult> {
  return async (input: T) => {
    try {
      return await handler(input)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      return { content: [{ type: 'text', text: `error: ${message}` }], isError: true }
    }
  }
}
