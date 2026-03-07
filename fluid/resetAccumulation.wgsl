struct GridAccum {
    momentum_x: atomic<i32>,
    momentum_y: atomic<i32>,
    momentum_z: atomic<i32>,
    mass: atomic<i32>,
}

@group(0) @binding(0) var<storage, read_write> accumulation: array<GridAccum>;

@compute @workgroup_size(128)
fn resetAccumulation(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&accumulation)) {
        return;
    }

    atomicStore(&accumulation[id.x].momentum_x, 0);
    atomicStore(&accumulation[id.x].momentum_y, 0);
    atomicStore(&accumulation[id.x].momentum_z, 0);
    atomicStore(&accumulation[id.x].mass, 0);
}
