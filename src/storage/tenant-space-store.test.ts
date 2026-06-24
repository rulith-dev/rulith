import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { MemorySpaceStore } from './memory-space-store.js'
import { TenantGuardSpaceStore } from './tenant-space-store.js'

describe('TenantGuardSpaceStore — multi-tenant isolation seam (➕)', () => {
  it('a tenant reads and writes its own spaces', () => {
    const inner = new MemorySpaceStore()
    const a = new TenantGuardSpaceStore(inner, { tenantId: 'A' })
    const sp = a.createSpace({ title: 'A-space' })
    assert.ok(a.getSpace(sp.id))
    const node = a.addNode(sp.id, { type: 'fact', label: 'x' })
    assert.ok(a.getNode(sp.id, node.id))
    assert.equal(a.listNodes(sp.id).length, 1)
  })

  it('a tenant CANNOT read/write/list another tenant\'s spaces (cross-tenant zero leakage)', () => {
    const inner = new MemorySpaceStore()
    const a = new TenantGuardSpaceStore(inner, { tenantId: 'A' })
    const b = new TenantGuardSpaceStore(inner, { tenantId: 'B' })
    const sp = a.createSpace({ title: 'A-only' })

    assert.throws(() => b.getSpace(sp.id), /cross-tenant/)
    assert.throws(() => b.addNode(sp.id, { type: 'fact', label: 'x' }), /cross-tenant/)
    assert.throws(() => b.getNode(sp.id, 'n1'), /cross-tenant/)
    assert.throws(() => b.listNodes(sp.id), /cross-tenant/)
    assert.throws(() => b.removeNode(sp.id, 'n1'), /cross-tenant/)

    // B's own space works and is invisible to A.
    const spB = b.createSpace({ title: 'B-only' })
    assert.ok(b.getSpace(spB.id))
    assert.throws(() => a.getSpace(spB.id), /cross-tenant/)
  })

  it('listSpaces returns only the calling tenant\'s spaces', () => {
    const inner = new MemorySpaceStore()
    const a = new TenantGuardSpaceStore(inner, { tenantId: 'A' })
    const b = new TenantGuardSpaceStore(inner, { tenantId: 'B' })
    a.createSpace({ title: 'A1' })
    a.createSpace({ title: 'A2' })
    b.createSpace({ title: 'B1' })
    assert.equal(a.listSpaces().length, 2)
    assert.equal(b.listSpaces().length, 1)
    assert.equal(inner.listSpaces().length, 3, 'the raw store still sees all tenants')
  })

  it('opt-in: an un-wrapped store behaves exactly as today (no tenant tag, no guard)', () => {
    const inner = new MemorySpaceStore()
    const sp = inner.createSpace({ title: 'plain' })
    assert.ok(inner.getSpace(sp.id))
    assert.equal(sp.scopes.some((s) => s.startsWith('tenant:')), false)
  })
})
