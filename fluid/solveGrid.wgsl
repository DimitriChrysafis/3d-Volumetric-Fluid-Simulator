struct GridAccum {
    momentum_x: atomic<i32>,
    momentum_y: atomic<i32>,
    momentum_z: atomic<i32>,
    mass: atomic<i32>,
}

struct GridState {
    velocity: vec3f,
    density: f32,
}

struct SimParams {
    grid_shape: vec4f,
    live_box: vec4f,
    forces: vec4f,
    transfer: vec4f,
    damping: vec4f,
}

@group(0) @binding(0) var<storage, read> accumulation: array<GridAccum>;
@group(0) @binding(1) var<storage, read_write> grid_state: array<GridState>;
@group(0) @binding(2) var<uniform> params: SimParams;

fn decodeFixedPoint(value: i32) -> f32 {
    return f32(value) / params.transfer.w;
}

fn gridShape() -> vec3i {
    return vec3i(i32(params.grid_shape.x), i32(params.grid_shape.y), i32(params.grid_shape.z));
}

fn insideGrid(cell: vec3i, dims: vec3i) -> bool {
    return all(cell >= vec3i(0, 0, 0)) && all(cell < dims);
}

fn flattenCell(cell: vec3i, dims: vec3i) -> u32 {
    return u32((cell.x * dims.y + cell.y) * dims.z + cell.z);
}

fn unflattenCell(index: u32, dims: vec3i) -> vec3i {
    let yz = dims.y * dims.z;
    let x = i32(index) / yz;
    let rem = i32(index) - x * yz;
    let y = rem / dims.z;
    let z = rem - y * dims.z;
    return vec3i(x, y, z);
}

fn cellToVec(cell: vec3i) -> vec3f {
    return vec3f(f32(cell.x), f32(cell.y), f32(cell.z));
}

fn sampleDensity(cell: vec3i, dims: vec3i) -> f32 {
    if (!insideGrid(cell, dims)) {
        return 0.0;
    }
    return decodeFixedPoint(atomicLoad(&accumulation[flattenCell(cell, dims)].mass));
}

fn sampleVelocity(cell: vec3i, dims: vec3i) -> vec3f {
    if (!insideGrid(cell, dims)) {
        return vec3f(0.0, 0.0, 0.0);
    }

    let flat = flattenCell(cell, dims);
    let mass = max(sampleDensity(cell, dims), 1e-6);
    return vec3f(
        decodeFixedPoint(atomicLoad(&accumulation[flat].momentum_x)),
        decodeFixedPoint(atomicLoad(&accumulation[flat].momentum_y)),
        decodeFixedPoint(atomicLoad(&accumulation[flat].momentum_z))
    ) / mass;
}

fn applyWallResponse(cell: vec3f, velocity: vec3f) -> vec3f {
    var corrected = velocity;
    let wallPadding = params.transfer.y;
    let drag = 1.0 - params.damping.x;
    let boxMax = params.live_box.xyz - vec3f(wallPadding + 1.0, wallPadding + 1.0, wallPadding + 1.0);

    if (cell.x < wallPadding && corrected.x < 0.0) {
        corrected.x = -corrected.x * params.transfer.z;
        corrected.y *= drag;
        corrected.z *= drag;
    }
    if (cell.x > boxMax.x && corrected.x > 0.0) {
        corrected.x = -corrected.x * params.transfer.z;
        corrected.y *= drag;
        corrected.z *= drag;
    }
    if (cell.y < wallPadding && corrected.y < 0.0) {
        corrected.y = -corrected.y * params.transfer.z;
        corrected.x *= drag;
        corrected.z *= drag;
    }
    if (cell.y > boxMax.y && corrected.y > 0.0) {
        corrected.y = -corrected.y * params.transfer.z;
        corrected.x *= drag;
        corrected.z *= drag;
    }
    if (cell.z < wallPadding && corrected.z < 0.0) {
        corrected.z = -corrected.z * params.transfer.z;
        corrected.x *= drag;
        corrected.y *= drag;
    }
    if (cell.z > boxMax.z && corrected.z > 0.0) {
        corrected.z = -corrected.z * params.transfer.z;
        corrected.x *= drag;
        corrected.y *= drag;
    }

    return corrected;
}

@compute @workgroup_size(128)
fn solveGrid(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&grid_state)) {
        return;
    }

    let dims = gridShape();
    let cell = unflattenCell(id.x, dims);
    let density = sampleDensity(cell, dims);

    if (density <= 1e-6) {
        grid_state[id.x].velocity = vec3f(0.0, 0.0, 0.0);
        grid_state[id.x].density = 0.0;
        return;
    }

    var velocity = sampleVelocity(cell, dims);
    let dt = params.live_box.w;

    let densityGradient = 0.5 * vec3f(
        sampleDensity(cell + vec3i(1, 0, 0), dims) - sampleDensity(cell - vec3i(1, 0, 0), dims),
        sampleDensity(cell + vec3i(0, 1, 0), dims) - sampleDensity(cell - vec3i(0, 1, 0), dims),
        sampleDensity(cell + vec3i(0, 0, 1), dims) - sampleDensity(cell - vec3i(0, 0, 1), dims)
    );

    let neighborAverage = (
        sampleVelocity(cell + vec3i(1, 0, 0), dims) +
        sampleVelocity(cell - vec3i(1, 0, 0), dims) +
        sampleVelocity(cell + vec3i(0, 1, 0), dims) +
        sampleVelocity(cell - vec3i(0, 1, 0), dims) +
        sampleVelocity(cell + vec3i(0, 0, 1), dims) +
        sampleVelocity(cell - vec3i(0, 0, 1), dims)
    ) / 6.0;

    let compression = max(density - params.forces.y, 0.0);
    velocity -= densityGradient * (params.forces.z * dt);
    velocity -= densityGradient * (compression * params.forces.z * 0.25 * dt);
    velocity += (neighborAverage - velocity) * clamp(params.forces.w * dt, 0.0, 0.45);
    velocity.y += params.forces.x * dt;
    velocity *= max(0.0, 1.0 - params.damping.y * dt);
    velocity = applyWallResponse(cellToVec(cell) + vec3f(0.5, 0.5, 0.5), velocity);

    grid_state[id.x].velocity = velocity;
    grid_state[id.x].density = density;
}
