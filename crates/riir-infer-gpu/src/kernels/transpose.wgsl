// Standalone matrix transpose: output[cols × rows] = input[rows × cols]^T.
//
// Used by Issue 402 Phase 9h: GPU-side transpose of the LM head weight matrix
// (640 MB at 0.40B scale). The CPU transpose of this matrix was ~674 ms/step
// (91% of the backward update phase) due to the extreme aspect ratio
// (163840 × 1024) causing scattered cache writes.
//
// Tiled with shared memory: each workgroup loads a 16×16 tile of the input,
// then writes the transposed tile to the output. This keeps both the read
// and write patterns coalesced within the workgroup.
//
// Workgroup size: 16×16 = 256 invocations.

var<workgroup> tile: array<f32, 256>; // 16×16 tile

@group(0) @binding(0) var<storage, read>         input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write>   output_data: array<f32>;
@group(0) @binding(2) var<uniform>               params: TransposeParams;

struct TransposeParams {
    rows: u32,  // input rows (output cols)
    cols: u32,  // input cols (output rows)
}

@compute @workgroup_size(16, 16, 1)
fn transpose_matrix(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let local_row = lid.x;
    let local_col = lid.y;

    // Input element position.
    let in_row = wid.x * 16u + local_row;
    let in_col = wid.y * 16u + local_col;

    // Load input tile into shared memory (guard on input dims).
    if (in_row < params.rows && in_col < params.cols) {
        tile[local_col * 16u + local_row] = input_data[in_row * params.cols + in_col];
    } else {
        tile[local_col * 16u + local_row] = 0.0;
    }

    workgroupBarrier();

    // Write transposed tile to output.
    // After transpose: output position is (in_col, in_row) in the [cols × rows] output.
    // The transposed local position is (local_col, local_row) within the tile.
    let out_row = wid.y * 16u + local_row; // maps to in_col dimension
    let out_col = wid.x * 16u + local_col; // maps to in_row dimension

    if (out_row < params.cols && out_col < params.rows) {
        // Read from transposed position in the tile.
        output_data[out_row * params.rows + out_col] = tile[local_row * 16u + local_col];
    }
}
