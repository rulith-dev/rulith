import type { CreateNodeInput, ProblemSpace, SpaceNode } from '../model/types.js'
import type { CreateSpaceInput, NodePatch, SpaceStore } from './space-store.js'

/**
 * TenantGuardSpaceStore — the multi-tenant isolation seam (➕, open-core).
 *
 * Wraps ANY SpaceStore so every read/write is scoped to one tenant: a space is
 * tagged with a reserved scope `tenant:<id>` at creation, and every later
 * operation asserts the space carries that tag; `listSpaces` returns only the
 * tenant's spaces. Same SpaceStore contract — no new methods, only isolation
 * semantics. The hosting layer (deployment form C / strategy H1) composes it;
 * everything else is unchanged (OPT-IN — an un-wrapped store behaves as today).
 *
 * Thin by design (architecture 三铁律: thin seam now, heavy impl later). The
 * load-bearing invariant: a tenant can never read, write, or list another
 * tenant's spaces — cross-tenant zero leakage (see tenant-space-store.test.ts).
 */

export type TenantContext = { tenantId: string }

const tenantScope = (tenantId: string): string => `tenant:${tenantId}`

export class TenantGuardSpaceStore implements SpaceStore {
  constructor(
    private readonly inner: SpaceStore,
    private readonly tenant: TenantContext,
  ) {}

  private assertOwned(spaceId: string): ProblemSpace {
    const space = this.inner.getSpace(spaceId)
    if (!space.scopes.includes(tenantScope(this.tenant.tenantId))) {
      throw new Error(
        `tenant '${this.tenant.tenantId}' may not access space '${spaceId}' (cross-tenant access denied)`,
      )
    }
    return space
  }

  createSpace(input: CreateSpaceInput): ProblemSpace {
    const tag = tenantScope(this.tenant.tenantId)
    const scopes = [...(input.scopes ?? [])]
    if (!scopes.includes(tag)) scopes.push(tag)
    return this.inner.createSpace({ ...input, scopes })
  }

  getSpace(spaceId: string): ProblemSpace {
    return this.assertOwned(spaceId)
  }

  listSpaces(): ProblemSpace[] {
    const tag = tenantScope(this.tenant.tenantId)
    return this.inner.listSpaces().filter((s) => s.scopes.includes(tag))
  }

  addNode(spaceId: string, input: CreateNodeInput): SpaceNode {
    this.assertOwned(spaceId)
    return this.inner.addNode(spaceId, input)
  }

  updateNode(spaceId: string, nodeId: string, patch: NodePatch): SpaceNode {
    this.assertOwned(spaceId)
    return this.inner.updateNode(spaceId, nodeId, patch)
  }

  getNode(spaceId: string, nodeId: string): SpaceNode {
    this.assertOwned(spaceId)
    return this.inner.getNode(spaceId, nodeId)
  }

  listNodes(spaceId: string): SpaceNode[] {
    this.assertOwned(spaceId)
    return this.inner.listNodes(spaceId)
  }

  removeNode(spaceId: string, nodeId: string): void {
    this.assertOwned(spaceId)
    this.inner.removeNode(spaceId, nodeId)
  }
}
