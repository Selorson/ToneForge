//! Modular Node Graph (`Patcher_x64.dll`: `TModularMap` `0x00790440`, `TMultiBufferDelay` `0x0078ed08`,
//! and `TSugiyamaLayout` `0x008294b8`).
//!
//! `PatcherGraphNode` implements `UnifiedDspNode` so an entire modular sub-DAG of plugins and
//! parameter cords can be dropped into any Mixer Track FX Slot with automatic internal APDC
//! and **Zero Heap Allocation** during `process_slice`.

use crate::apdc_engine::{ApdcDelayLine, DEFAULT_APDC_CAPACITY};
use crate::{MAX_BLOCK_SIZE, ProcessContext, StereoFrame, UnifiedDspNode};

/// Reserved virtual node index representing `THostModule` Audio Input (`[ModularMap + 0x50]`).
pub const PATCHER_HOST_INPUT_NODE: usize = usize::MAX - 1;
/// Reserved virtual node index representing `THostModule` Audio Output (`[ModularMap + 0x58]`).
pub const PATCHER_HOST_OUTPUT_NODE: usize = usize::MAX;

/// Audio connection cord inside `PatcherGraphNode` (`TModularConnection` `0x00790010`),
/// equipped with an internal `TMultiBufferDelay` (`0x0078ed08`) for parallel branch APDC.
pub struct ModularAudioConnection {
    pub from_node: usize,
    pub to_node: usize,
    pub gain: f32,
    pub edge_apdc_delay: ApdcDelayLine,
    pub edge_scratch: Vec<StereoFrame>,
}

impl ModularAudioConnection {
    pub fn new(from_node: usize, to_node: usize, gain: f32) -> Self {
        Self {
            from_node,
            to_node,
            gain,
            edge_apdc_delay: ApdcDelayLine::with_capacity(DEFAULT_APDC_CAPACITY),
            edge_scratch: vec![[0.0; 2]; MAX_BLOCK_SIZE],
        }
    }
}

/// Parameter modulation cord inside `PatcherGraphNode` (`TModularPluginParamInput` `0x007f7fa8`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModularParamConnection {
    pub source_controller: u16,
    pub target_node: usize,
    pub target_param_id: u32,
    pub scale: f32,
    pub offset: f32,
}

/// 2D node layout coordinates computed by `SugiyamaLayout` (`TSugiyamaLayout` `0x008294b8`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeLayoutPosition {
    pub node_index: usize,
    pub layer: usize,
    pub rank_in_layer: usize,
    pub x_px: f32,
    pub y_px: f32,
}

/// Sugiyama layered DAG auto-layout engine (`TSugiyamaLayout` `0x008294b8` -> `TModularLayout` `0x0082be08`).
pub struct SugiyamaLayout;

impl SugiyamaLayout {
    /// Computes layered coordinates `(x_px, y_px)` with barycentric crossing reduction.
    pub fn compute_layout(
        num_nodes: usize,
        edges: &[(usize, usize)],
        layer_spacing_x: f32,
        node_spacing_y: f32,
    ) -> Vec<NodeLayoutPosition> {
        if num_nodes == 0 {
            return Vec::new();
        }
        let mut in_deg = vec![0usize; num_nodes];
        let mut adj = vec![Vec::<usize>::new(); num_nodes];
        let mut preds = vec![Vec::<usize>::new(); num_nodes];

        for &(u, v) in edges {
            if u < num_nodes && v < num_nodes && u != v {
                adj[u].push(v);
                preds[v].push(u);
                in_deg[v] += 1;
            }
        }

        // Longest-path layer assignment
        let mut layer_of = vec![0usize; num_nodes];
        let mut queue: Vec<usize> = (0..num_nodes).filter(|&i| in_deg[i] == 0).collect();
        let mut head = 0usize;
        while head < queue.len() {
            let u = queue[head];
            head += 1;
            for &v in &adj[u] {
                layer_of[v] = layer_of[v].max(layer_of[u] + 1);
                in_deg[v] -= 1;
                if in_deg[v] == 0 {
                    queue.push(v);
                }
            }
        }

        let max_layer = layer_of.iter().copied().max().unwrap_or(0);
        let mut layers: Vec<Vec<usize>> = vec![Vec::new(); max_layer + 1];
        for (node, &l) in layer_of.iter().enumerate() {
            layers[l].push(node);
        }

        // Barycentric heuristic ordering within each layer to minimize edge crossings
        let mut rank_of = vec![0usize; num_nodes];
        for l in 0..=max_layer {
            if l > 0 {
                layers[l].sort_by(|&a, &b| {
                    let bary_a = if preds[a].is_empty() {
                        a as f32
                    } else {
                        preds[a].iter().map(|&p| rank_of[p] as f32).sum::<f32>()
                            / (preds[a].len() as f32)
                    };
                    let bary_b = if preds[b].is_empty() {
                        b as f32
                    } else {
                        preds[b].iter().map(|&p| rank_of[p] as f32).sum::<f32>()
                            / (preds[b].len() as f32)
                    };
                    bary_a.total_cmp(&bary_b)
                });
            }
            for (r, &node) in layers[l].iter().enumerate() {
                rank_of[node] = r;
            }
        }

        let mut positions = Vec::with_capacity(num_nodes);
        for (l, layer_nodes) in layers.iter().enumerate() {
            for (r, &node) in layer_nodes.iter().enumerate() {
                positions.push(NodeLayoutPosition {
                    node_index: node,
                    layer: l,
                    rank_in_layer: r,
                    x_px: (l as f32) * layer_spacing_x + 80.0,
                    y_px: (r as f32) * node_spacing_y + 60.0,
                });
            }
        }
        positions.sort_by_key(|p| p.node_index);
        positions
    }
}

struct PatcherInternalNode {
    dsp: Box<dyn UnifiedDspNode>,
    in_buf: Vec<StereoFrame>,
    accumulated_latency: usize,
}

/// Modular Node Graph (`Patcher_x64.dll` `TTestPlug` `0x0094d598` & `TModularMap` `0x00790440`).
/// Executes a topological DAG of child `UnifiedDspNode` instances with internal branch APDC
/// (`TMultiBufferDelay`) and **Zero Heap Allocation** in `process_slice`.
pub struct PatcherGraphNode {
    nodes: Vec<PatcherInternalNode>,
    connections: Vec<ModularAudioConnection>,
    param_connections: Vec<ModularParamConnection>,
    topo_order: Vec<usize>,
    output_sum_buf: Vec<StereoFrame>,
    total_latency_samples: usize,
    sample_rate: f64,
    max_block_size: usize,
}

impl Default for PatcherGraphNode {
    fn default() -> Self {
        Self::new()
    }
}

impl PatcherGraphNode {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            connections: Vec::new(),
            param_connections: Vec::new(),
            topo_order: Vec::new(),
            output_sum_buf: vec![[0.0; 2]; MAX_BLOCK_SIZE],
            total_latency_samples: 0,
            sample_rate: 48_000.0,
            max_block_size: MAX_BLOCK_SIZE,
        }
    }

    /// Adds a child `UnifiedDspNode` to the modular map and returns its node index `0..N`.
    pub fn add_node(&mut self, mut node: Box<dyn UnifiedDspNode>) -> usize {
        node.prepare(self.sample_rate, self.max_block_size);
        let idx = self.nodes.len();
        self.nodes.push(PatcherInternalNode {
            dsp: node,
            in_buf: vec![[0.0; 2]; self.max_block_size],
            accumulated_latency: 0,
        });
        self.rebuild_topology_and_apdc();
        idx
    }

    /// Connects `from_node` (or `PATCHER_HOST_INPUT_NODE`) to `to_node` (or `PATCHER_HOST_OUTPUT_NODE`).
    /// Rejects connections that would introduce a cycle.
    pub fn add_audio_connection(&mut self, from_node: usize, to_node: usize, gain: f32) -> bool {
        if from_node == to_node {
            return false;
        }
        self.connections
            .push(ModularAudioConnection::new(from_node, to_node, gain));
        if !self.rebuild_topology_and_apdc() {
            self.connections.pop();
            self.rebuild_topology_and_apdc();
            return false;
        }
        true
    }

    pub fn add_param_connection(&mut self, conn: ModularParamConnection) {
        self.param_connections.push(conn);
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Computes automatic Sugiyama 2D layout positions (`TSugiyamaLayout` `0x008294b8`) for all internal nodes.
    pub fn compute_sugiyama_layout(&self) -> Vec<NodeLayoutPosition> {
        let n = self.nodes.len();
        let edges: Vec<(usize, usize)> = self
            .connections
            .iter()
            .filter(|c| c.from_node < n && c.to_node < n)
            .map(|c| (c.from_node, c.to_node))
            .collect();
        SugiyamaLayout::compute_layout(n, &edges, 180.0, 96.0)
    }

    /// Rebuilds the topological sort (`[ModularMap + 0x60]`) and calculates internal `TMultiBufferDelay`
    /// APDC for every parallel audio cord inside the Patcher graph.
    pub fn rebuild_topology_and_apdc(&mut self) -> bool {
        let n = self.nodes.len();
        let mut in_deg = vec![0usize; n];
        let mut adj = vec![Vec::<usize>::new(); n];

        for conn in &self.connections {
            if conn.from_node < n && conn.to_node < n {
                adj[conn.from_node].push(conn.to_node);
                in_deg[conn.to_node] += 1;
            }
        }

        let mut queue: Vec<usize> = (0..n).filter(|&i| in_deg[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        let mut max_in_lat = vec![0usize; n];

        let mut head = 0usize;
        while head < queue.len() {
            let u = queue[head];
            head += 1;
            order.push(u);

            let out_lat_u = max_in_lat[u] + self.nodes[u].dsp.latency_samples();
            self.nodes[u].accumulated_latency = out_lat_u;

            for &v in &adj[u] {
                max_in_lat[v] = max_in_lat[v].max(out_lat_u);
                in_deg[v] -= 1;
                if in_deg[v] == 0 {
                    queue.push(v);
                }
            }
        }

        if order.len() != n {
            return false; // Cycle detected
        }

        // Compute critical-path latency at PATCHER_HOST_OUTPUT_NODE
        let mut host_out_lat = 0usize;
        for conn in &self.connections {
            if conn.to_node == PATCHER_HOST_OUTPUT_NODE {
                let src_lat = if conn.from_node == PATCHER_HOST_INPUT_NODE {
                    0
                } else if conn.from_node < n {
                    self.nodes[conn.from_node].accumulated_latency
                } else {
                    0
                };
                host_out_lat = host_out_lat.max(src_lat);
            }
        }
        self.total_latency_samples = host_out_lat;

        // Configure `TMultiBufferDelay` (`edge_apdc_delay`) on every internal edge
        for conn in &mut self.connections {
            let src_lat = if conn.from_node == PATCHER_HOST_INPUT_NODE {
                0
            } else if conn.from_node < n {
                self.nodes[conn.from_node].accumulated_latency
            } else {
                0
            };
            let dst_target_lat = if conn.to_node == PATCHER_HOST_OUTPUT_NODE {
                host_out_lat
            } else if conn.to_node < n {
                max_in_lat[conn.to_node]
            } else {
                0
            };
            let slack = dst_target_lat.saturating_sub(src_lat);
            conn.edge_apdc_delay
                .ensure_capacity(slack.max(DEFAULT_APDC_CAPACITY));
            conn.edge_apdc_delay.set_delay_samples(slack);
        }

        self.topo_order = order;
        true
    }
}

impl UnifiedDspNode for PatcherGraphNode {
    fn name(&self) -> &'static str {
        "Patcher Modular DAG"
    }

    fn latency_samples(&self) -> usize {
        self.total_latency_samples
    }

    fn prepare(&mut self, sample_rate: f64, max_block_size: usize) {
        self.sample_rate = sample_rate.max(8000.0);
        self.max_block_size = max_block_size.max(MAX_BLOCK_SIZE);
        if self.output_sum_buf.len() < self.max_block_size {
            self.output_sum_buf.resize(self.max_block_size, [0.0; 2]);
        }
        for node in &mut self.nodes {
            if node.in_buf.len() < self.max_block_size {
                node.in_buf.resize(self.max_block_size, [0.0; 2]);
            }
            node.dsp.prepare(self.sample_rate, self.max_block_size);
        }
        for conn in &mut self.connections {
            if conn.edge_scratch.len() < self.max_block_size {
                conn.edge_scratch.resize(self.max_block_size, [0.0; 2]);
            }
        }
        self.rebuild_topology_and_apdc();
    }

    fn reset_state(&mut self) {
        for node in &mut self.nodes {
            node.dsp.reset_state();
        }
        for conn in &mut self.connections {
            conn.edge_apdc_delay.reset();
        }
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let mut matched_cord = false;
        if param_id <= u16::MAX as u32 {
            let src_ctrl = param_id as u16;
            for i in 0..self.param_connections.len() {
                let conn = self.param_connections[i];
                if conn.source_controller == src_ctrl {
                    let mapped = (norm_val * conn.scale + conn.offset).clamp(0.0, 1.0);
                    if let Some(node) = self.nodes.get_mut(conn.target_node) {
                        node.dsp
                            .set_parameter_normalized(conn.target_param_id, mapped);
                        matched_cord = true;
                    }
                }
            }
        }
        if !matched_cord {
            let node_idx = (param_id >> 16) as usize;
            let local_param = param_id & 0xFFFF;
            if let Some(node) = self.nodes.get_mut(node_idx) {
                node.dsp.set_parameter_normalized(local_param, norm_val);
            }
        }
    }

    fn on_ppq_tick(&mut self, ctx: &ProcessContext) {
        for node in &mut self.nodes {
            node.dsp.on_ppq_tick(ctx);
        }
    }

    fn export_visual_telemetry(&self, frame: &mut crate::telemetry_shm::VisualTelemetryFrame) {
        let limit = self.nodes.len().min(510);
        for i in 0..limit {
            let lat_weight = (self.nodes[i].dsp.latency_samples() as f32) * 0.0005;
            frame.automation_spline.patcher_node_cpu_pct[i] = (0.15 + lat_weight).min(25.0);
            self.nodes[i].dsp.export_visual_telemetry(frame);
        }
    }

    /// Executes `sub_79cd80(ModularMap, Length)` in topological order with Zero Heap Allocation.
    fn process_slice(
        &mut self,
        ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        let len = main_io.len().min(self.max_block_size);
        if len == 0 || self.connections.is_empty() {
            return;
        }

        // Clear all node input buffers and output accumulator
        for node in &mut self.nodes {
            node.in_buf[..len].fill([0.0; 2]);
        }
        self.output_sum_buf[..len].fill([0.0; 2]);

        // 1. Route edges originating from PATCHER_HOST_INPUT_NODE
        let num_nodes = self.nodes.len();
        for conn in &mut self.connections {
            if conn.from_node == PATCHER_HOST_INPUT_NODE {
                let scratch = &mut conn.edge_scratch[..len];
                scratch.copy_from_slice(&main_io[..len]);
                conn.edge_apdc_delay.process_in_place(scratch);
                let g = conn.gain;
                if conn.to_node == PATCHER_HOST_OUTPUT_NODE {
                    for i in 0..len {
                        self.output_sum_buf[i][0] += scratch[i][0] * g;
                        self.output_sum_buf[i][1] += scratch[i][1] * g;
                    }
                } else if conn.to_node < num_nodes {
                    let dst = &mut self.nodes[conn.to_node].in_buf[..len];
                    for i in 0..len {
                        dst[i][0] += scratch[i][0] * g;
                        dst[i][1] += scratch[i][1] * g;
                    }
                }
            }
        }

        // 2. Execute internal nodes in topological order (`[ModularMap + 0x60]`)
        for ord_idx in 0..self.topo_order.len() {
            let u = self.topo_order[ord_idx];
            {
                let node_u = &mut self.nodes[u];
                node_u
                    .dsp
                    .process_slice(ctx, &mut node_u.in_buf[..len], sidechain_in, &mut []);
            }

            // Route outgoing edges from node `u`
            for conn in &mut self.connections {
                if conn.from_node == u {
                    let scratch = &mut conn.edge_scratch[..len];
                    scratch.copy_from_slice(&self.nodes[u].in_buf[..len]);
                    conn.edge_apdc_delay.process_in_place(scratch);
                    let g = conn.gain;
                    if conn.to_node == PATCHER_HOST_OUTPUT_NODE {
                        for i in 0..len {
                            self.output_sum_buf[i][0] += scratch[i][0] * g;
                            self.output_sum_buf[i][1] += scratch[i][1] * g;
                        }
                    } else if conn.to_node < num_nodes {
                        let dst = &mut self.nodes[conn.to_node].in_buf[..len];
                        for i in 0..len {
                            dst[i][0] += scratch[i][0] * g;
                            dst[i][1] += scratch[i][1] * g;
                        }
                    }
                }
            }
        }

        // 3. Copy final mixed output to `main_io`
        main_io[..len].copy_from_slice(&self.output_sum_buf[..len]);
    }
}
