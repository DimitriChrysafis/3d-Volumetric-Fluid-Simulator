struct Particle {
    position: vec3f,
    _pad0: f32,
    v: vec3f,
    _pad1: f32,
    C: mat3x3f,
}

struct GridAccum {
    momentum_x: atomic<i32>,
    momentum_y: atomic<i32>,
    momentum_z: atomic<i32>,
    mass: atomic<i32>,
}

struct SimParams {
    grid_shape: vec4f,
    live_box: vec4f,
    forces: vec4f,
    transfer: vec4f,
    damping: vec4f,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> accumulation: array<GridAccum>;
@group(0) @binding(2) var<uniform> params: SimParams;

fn encodeFixedPoint(value: f32) -> i32 {
    return i32(round(value * params.transfer.w));
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

fn cellToVec(cell: vec3i) -> vec3f {
    return vec3f(f32(cell.x), f32(cell.y), f32(cell.z));
}

fn floorToCell(position: vec3f) -> vec3i {
    return vec3i(i32(floor(position.x)), i32(floor(position.y)), i32(floor(position.z)));
}

fn squareVec(value: vec3f) -> vec3f {
    return value * value;
}

fn kernelWeights(offset: vec3f) -> array<vec3f, 3> {
    var weights: array<vec3f, 3>;
    weights[0] = 0.5 * squareVec(vec3f(0.5, 0.5, 0.5) - offset);
    weights[1] = vec3f(0.75, 0.75, 0.75) - offset * offset;
    weights[2] = 0.5 * squareVec(vec3f(0.5, 0.5, 0.5) + offset);
    return weights;
}

@compute @workgroup_size(128)
fn scatterParticles(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&particles)) {
        return;
    }

    let dims = gridShape();
    let particle = particles[id.x];
    let baseCell = floorToCell(particle.position);
    let offsetFromCenter = particle.position - (cellToVec(baseCell) + vec3f(0.5, 0.5, 0.5));
    let weights = kernelWeights(offsetFromCenter);

    for (var z = 0; z < 3; z++) {
        for (var y = 0; y < 3; y++) {
            for (var x = 0; x < 3; x++) {
                let cell = baseCell + vec3i(x - 1, y - 1, z - 1);
                if (!insideGrid(cell, dims)) {
                    continue;
                }

                let weight = weights[x].x * weights[y].y * weights[z].z;
                let relativeNodeOffset = (cellToVec(cell) + vec3f(0.5, 0.5, 0.5)) - particle.position;
                let apicContribution = particle.C * relativeNodeOffset;
                let nodeMomentum = (particle.v + apicContribution) * weight;
                let flat = flattenCell(cell, dims);

                atomicAdd(&accumulation[flat].mass, encodeFixedPoint(weight));
                atomicAdd(&accumulation[flat].momentum_x, encodeFixedPoint(nodeMomentum.x));
                atomicAdd(&accumulation[flat].momentum_y, encodeFixedPoint(nodeMomentum.y));
                atomicAdd(&accumulation[flat].momentum_z, encodeFixedPoint(nodeMomentum.z));
            }
        }
    }
}
