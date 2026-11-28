// Pure rehydration of one external VST3/CLAP plugin entry from a project snapshot (extracted from
// `daw_master_store.svelte.ts` so it can be unit-tested without Svelte runes / Tauri).
import type { ExternalPluginParamUi, LoadedExternalPluginUi } from './daw_master_store.svelte';

export function hydrateExternalEntry(
  livePlugins: LoadedExternalPluginUi[],
  trackIdx: number,
  slotIdx: number,
  isClap: boolean,
  isInstrument: boolean,
  cfg: any,
  defaultName: string
): LoadedExternalPluginUi {
  const pluginId = String(cfg.plugin_id ?? '');
  const live = livePlugins.find(
    (p) => p.trackIdx === trackIdx && p.slotIdx === slotIdx && p.pluginId === pluginId
  );
  const rawOverrides = Array.isArray(cfg.param_overrides) ? cfg.param_overrides : [];
  const overrideMap = new Map<number, number>();
  for (const pair of rawOverrides) {
    if (Array.isArray(pair) && pair.length >= 2) {
      overrideMap.set(Number(pair[0]), Number(pair[1]));
    }
  }
  const parameters: ExternalPluginParamUi[] =
    live && live.parameters.length > 0
      ? [
          ...live.parameters.map((prm) => {
            const ov = overrideMap.get(prm.id);
            return ov !== undefined ? { ...prm, current_normalized: ov } : prm;
          }),
          ...Array.from(overrideMap.entries())
            .filter(([id]) => !live.parameters.some((prm) => prm.id === id))
            .map(([id, val]) => ({
              id,
              name: `Param #${id}`,
              default_normalized: val,
              current_normalized: val,
              automatable: true,
            })),
        ]
      : rawOverrides.map((pair: any) => ({
          id: Number(pair?.[0] ?? 0),
          name: `Param #${Number(pair?.[0] ?? 0)}`,
          default_normalized: Number(pair?.[1] ?? 0.5),
          current_normalized: Number(pair?.[1] ?? 0.5),
          automatable: true,
        }));
  return {
    trackIdx,
    slotIdx,
    isClap,
    isInstrument,
    name: String(cfg.plugin_name ?? live?.name ?? defaultName),
    vendor: live?.vendor || (isClap ? 'CLAP 1.2' : 'Steinberg VST3'),
    pluginId,
    binaryPath: String(cfg.binary_path ?? live?.binaryPath ?? ''),
    latencySamples: live ? live.latencySamples : 0,
    guiOpen: false, // backend closes every plugin window on Undo/Open, so never trust the stale live flag
    parameters,
  };
}
