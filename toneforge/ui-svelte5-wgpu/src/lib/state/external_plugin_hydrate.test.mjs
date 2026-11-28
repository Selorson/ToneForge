// Run: node --experimental-strip-types --test src/lib/state/external_plugin_hydrate.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { hydrateExternalEntry } from './external_plugin_hydrate.ts';

const live = {
  trackIdx: 1, slotIdx: 0, isClap: false, isInstrument: false, name: 'Live', vendor: 'V',
  pluginId: 'p1', binaryPath: 'x.vst3', latencySamples: 64, guiOpen: true, parameters: [],
};

test('hydrateExternalEntry never reports a GUI window as open (backend closes all on Undo/Open)', () => {
  const e = hydrateExternalEntry([live], 1, 0, false, false, { plugin_id: 'p1' }, 'External FX');
  assert.equal(e.guiOpen, false);
  assert.equal(e.latencySamples, 64); // other live fields still carried over
});

test('hydrateExternalEntry without a live match is closed too', () => {
  const e = hydrateExternalEntry([], 1, 0, true, false, { plugin_id: 'p1' }, 'External FX');
  assert.equal(e.guiOpen, false);
});
