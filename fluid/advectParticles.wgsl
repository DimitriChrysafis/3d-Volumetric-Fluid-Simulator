struct Particle {
    position: vec3f,
    _pad0: f32,
    v: vec3f,
    _pad1: f32,
    C: mat3x3f,
}

struct GridState {
    velocity: vec3f,
    density: f32,
}

struct PosVel {
    position: vec3f,
    _pad0: f32,
    v: vec3f,
    _pad1: f32,
}

struct SimParams {
    grid_shape: vec4f,
    live_box: vec4f,
    forces: vec4f,
    transfer: vec4f,
    damping: vec4f,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> grid_state: array<GridState>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<storage, read_write> posvel: array<PosVel>;

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

fn outerProduct(lhs: vec3f, rhs: vec3f) -> mat3x3f {
    return mat3x3f(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z);
}

struct BoundaryResult {
    position: vec3f,
    velocity: vec3f,
}

fn resolveParticleBoundary(position: vec3f, velocity: vec3f) -> BoundaryResult {
    var nextPosition = position;
    var nextVelocity = velocity;
    let padding = params.transfer.y;
    let minBounds = vec3f(padding, padding, padding);
    let maxBounds = params.live_box.xyz - vec3f(padding + 1.0, padding + 1.0, padding + 1.0);
    let drag = 1.0 - params.damping.x;

    if (nextPosition.x < minBounds.x) {
        nextPosition.x = minBounds.x;
        if (nextVelocity.x < 0.0) { nextVelocity.x = -nextVelocity.x * params.transfer.z; }
        nextVelocity.y *= drag;
        nextVelocity.z *= drag;
    }
    if (nextPosition.x > maxBounds.x) {
        nextPosition.x = maxBounds.x;
        if (nextVelocity.x > 0.0) { nextVelocity.x = -nextVelocity.x * params.transfer.z; }
        nextVelocity.y *= drag;
        nextVelocity.z *= drag;
    }
    if (nextPosition.y < minBounds.y) {
        nextPosition.y = minBounds.y;
        if (nextVelocity.y < 0.0) { nextVelocity.y = -nextVelocity.y * params.transfer.z; }
        nextVelocity.x *= drag;
        nextVelocity.z *= drag;
    }
    if (nextPosition.y > maxBounds.y) {
        nextPosition.y = maxBounds.y;
        if (nextVelocity.y > 0.0) { nextVelocity.y = -nextVelocity.y * params.transfer.z; }
        nextVelocity.x *= drag;
        nextVelocity.z *= drag;
    }
    if (nextPosition.z < minBounds.z) {
        nextPosition.z = minBounds.z;
        if (nextVelocity.z < 0.0) { nextVelocity.z = -nextVelocity.z * params.transfer.z; }
        nextVelocity.x *= drag;
        nextVelocity.y *= drag;
    }
    if (nextPosition.z > maxBounds.z) {
        nextPosition.z = maxBounds.z;
        if (nextVelocity.z > 0.0) { nextVelocity.z = -nextVelocity.z * params.transfer.z; }
        nextVelocity.x *= drag;
        nextVelocity.y *= drag;
    }

    return BoundaryResult(nextPosition, nextVelocity);
}

@compute @workgroup_size(128)
fn advectParticles(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&particles)) {
        return;
    }

    let dims = gridShape();
    var particle = particles[id.x];
    let baseCell = floorToCell(particle.position);
    let offsetFromCenter = particle.position - (cellToVec(baseCell) + vec3f(0.5, 0.5, 0.5));
    let weights = kernelWeights(offsetFromCenter);
    let dt = params.live_box.w;

    var sampledVelocity = vec3f(0.0, 0.0, 0.0);
    var sampledDensity = 0.0;
    var affineMatrix = mat3x3f(vec3f(0.0, 0.0, 0.0), vec3f(0.0, 0.0, 0.0), vec3f(0.0, 0.0, 0.0));

    for (var z = 0; z < 3; z++) {
        for (var y = 0; y < 3; y++) {
            for (var x = 0; x < 3; x++) {
                let cell = baseCell + vec3i(x - 1, y - 1, z - 1);
                if (!insideGrid(cell, dims)) {
                    continue;
                }

                let weight = weights[x].x * weights[y].y * weights[z].z;
                let flat = flattenCell(cell, dims);
                let nodeVelocity = grid_state[flat].velocity;
                let relativeNodeOffset = (cellToVec(cell) + vec3f(0.5, 0.5, 0.5)) - particle.position;

                sampledVelocity += nodeVelocity * weight;
                sampledDensity += grid_state[flat].density * weight;
                affineMatrix += outerProduct(nodeVelocity, relativeNodeOffset) * (4.0 * weight);
            }
        }
    }

    particle.v += (sampledVelocity - particle.v) * params.transfer.x;
    if (sampledDensity < params.forces.y * 0.35) {
        particle.v.y += params.forces.x * dt * 0.5;
    }

    particle.C = affineMatrix;
    particle.position += particle.v * dt;
    let bounded = resolveParticleBoundary(particle.position, particle.v);
    particle.position = bounded.position;
    particle.v = bounded.velocity;

    particles[id.x] = particle;
    posvel[id.x].position = particle.position;
    posvel[id.x].v = particle.v;
}
