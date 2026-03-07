import { numParticlesMax, renderUniformsViews } from '../common.js'

export const fluidParticleStructSize = 80

const PARAM_FLOAT_COUNT = 20
const SHADER_REVISION = '20260307b'

export class ParticleGridFluidSimulator {
    constructor(particleBuffer, posvelBuffer, renderDiameter, device, boxWidth, boxHeight, boxDepth) {
        this.max_x_grids = Math.ceil(boxWidth * 1.15)
        this.max_y_grids = Math.ceil(boxHeight * 1.15)
        this.max_z_grids = Math.ceil(boxDepth * 2.1)
        this.gridAccumStructSize = 16
        this.gridStateStructSize = 16
        this.numParticles = 0
        this.gridCount = this.max_x_grids * this.max_y_grids * this.max_z_grids
        this.renderDiameter = renderDiameter
        this.device = device
        this.particleBuffer = particleBuffer
        this.posvelBuffer = posvelBuffer
        this.substeps = 2
        this._liveBoxSize = new Float32Array(3)
        this._gridShape = new Float32Array([this.max_x_grids, this.max_y_grids, this.max_z_grids])
        this._params = new Float32Array(PARAM_FLOAT_COUNT)
        this.tuning = {
            dt: 0.16,
            gravity: -0.38,
            targetMass: 0.96,
            pressureGain: 0.12,
            viscosity: 0.22,
            transferBlend: 0.9,
            wallPadding: 3.0,
            wallBounce: 0.12,
            fixedPointScale: 1e6,
            wallDrag: 0.18,
            velocityDecay: 0.03,
        }
    }

    async initialize() {
        const [resetAccumulation, scatterParticles, solveGrid, advectParticles] = await Promise.all([
            fetch(`fluid/resetAccumulation.wgsl?v=${SHADER_REVISION}`).then((response) => response.text()),
            fetch(`fluid/scatterParticles.wgsl?v=${SHADER_REVISION}`).then((response) => response.text()),
            fetch(`fluid/solveGrid.wgsl?v=${SHADER_REVISION}`).then((response) => response.text()),
            fetch(`fluid/advectParticles.wgsl?v=${SHADER_REVISION}`).then((response) => response.text()),
        ])

        const resetModule = this.device.createShaderModule({ code: resetAccumulation })
        const scatterModule = this.device.createShaderModule({ code: scatterParticles })
        const solveModule = this.device.createShaderModule({ code: solveGrid })
        const advectModule = this.device.createShaderModule({ code: advectParticles })

        this.resetPipeline = this.device.createComputePipeline({
            label: 'grid reset pipeline',
            layout: 'auto',
            compute: { module: resetModule },
        })
        this.scatterPipeline = this.device.createComputePipeline({
            label: 'particle scatter pipeline',
            layout: 'auto',
            compute: { module: scatterModule },
        })
        this.solvePipeline = this.device.createComputePipeline({
            label: 'grid solve pipeline',
            layout: 'auto',
            compute: { module: solveModule },
        })
        this.advectPipeline = this.device.createComputePipeline({
            label: 'particle update pipeline',
            layout: 'auto',
            compute: { module: advectModule },
        })

        this.gridAccumulationBuffer = this.device.createBuffer({
            label: 'grid accumulation buffer',
            size: this.gridAccumStructSize * this.gridCount,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
        this.gridStateBuffer = this.device.createBuffer({
            label: 'grid state buffer',
            size: this.gridStateStructSize * this.gridCount,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
        this.simulationParamsBuffer = this.device.createBuffer({
            label: 'simulation params buffer',
            size: this._params.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })

        this._writeSimulationParams()

        this.resetBindGroup = this.device.createBindGroup({
            layout: this.resetPipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: this.gridAccumulationBuffer } }],
        })
        this.scatterBindGroup = this.device.createBindGroup({
            layout: this.scatterPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.gridAccumulationBuffer } },
                { binding: 2, resource: { buffer: this.simulationParamsBuffer } },
            ],
        })
        this.solveBindGroup = this.device.createBindGroup({
            layout: this.solvePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.gridAccumulationBuffer } },
                { binding: 1, resource: { buffer: this.gridStateBuffer } },
                { binding: 2, resource: { buffer: this.simulationParamsBuffer } },
            ],
        })
        this.advectBindGroup = this.device.createBindGroup({
            layout: this.advectPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.particleBuffer } },
                { binding: 1, resource: { buffer: this.gridStateBuffer } },
                { binding: 2, resource: { buffer: this.simulationParamsBuffer } },
                { binding: 3, resource: { buffer: this.posvelBuffer } },
            ],
        })
    }

    _writeSimulationParams() {
        this._params[0] = this._gridShape[0]
        this._params[1] = this._gridShape[1]
        this._params[2] = this._gridShape[2]
        this._params[3] = 0

        this._params[4] = this._liveBoxSize[0]
        this._params[5] = this._liveBoxSize[1]
        this._params[6] = this._liveBoxSize[2]
        this._params[7] = this.tuning.dt

        this._params[8] = this.tuning.gravity
        this._params[9] = this.tuning.targetMass
        this._params[10] = this.tuning.pressureGain
        this._params[11] = this.tuning.viscosity

        this._params[12] = this.tuning.transferBlend
        this._params[13] = this.tuning.wallPadding
        this._params[14] = this.tuning.wallBounce
        this._params[15] = this.tuning.fixedPointScale

        this._params[16] = this.tuning.wallDrag
        this._params[17] = this.tuning.velocityDecay
        this._params[18] = 0
        this._params[19] = 0

        this.device.queue.writeBuffer(this.simulationParamsBuffer, 0, this._params)
    }

    initDambreak(initBoxSize, numParticles) {
        const particlesBuf = new ArrayBuffer(fluidParticleStructSize * numParticles)
        const spacing = 0.95
        this.numParticles = 0

        for (let y = 0; y < initBoxSize[1] * 1.6 && this.numParticles < numParticles; y += spacing) {
            for (let x = 3; x < initBoxSize[0] - 4 && this.numParticles < numParticles; x += spacing) {
                for (let z = 3; z < initBoxSize[2] - 4 && this.numParticles < numParticles; z += spacing) {
                    const offset = fluidParticleStructSize * this.numParticles
                    const particleViews = {
                        position: new Float32Array(particlesBuf, offset + 0, 3),
                        velocity: new Float32Array(particlesBuf, offset + 16, 3),
                        affine: new Float32Array(particlesBuf, offset + 32, 12),
                    }
                    const jitter = Math.random()
                    particleViews.position.set([x + jitter, y + jitter, z + jitter])
                    particleViews.velocity.set([0, 0, 0])
                    particleViews.affine.fill(0)
                    this.numParticles++
                }
            }
        }

        this.device.queue.writeBuffer(this.particleBuffer, 0, particlesBuf, 0, this.numParticles * fluidParticleStructSize)
    }

    reset(numParticles, initBoxSize) {
        renderUniformsViews.sphere_size.set([this.renderDiameter, this.renderDiameter])
        this.initDambreak(initBoxSize, numParticles)
        this._liveBoxSize.set(initBoxSize)
        this._writeSimulationParams()
    }

    execute(commandEncoder) {
        const computePass = commandEncoder.beginComputePass()
        const substeps = this.substeps > 0 ? this.substeps | 0 : 1

        for (let step = 0; step < substeps; step++) {
            computePass.setBindGroup(0, this.resetBindGroup)
            computePass.setPipeline(this.resetPipeline)
            computePass.dispatchWorkgroups(Math.ceil(this.gridCount / 128))

            computePass.setBindGroup(0, this.scatterBindGroup)
            computePass.setPipeline(this.scatterPipeline)
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 128))

            computePass.setBindGroup(0, this.solveBindGroup)
            computePass.setPipeline(this.solvePipeline)
            computePass.dispatchWorkgroups(Math.ceil(this.gridCount / 128))

            computePass.setBindGroup(0, this.advectBindGroup)
            computePass.setPipeline(this.advectPipeline)
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 128))
        }

        computePass.end()
    }

    changeBoxSize(realBoxSize) {
        this._liveBoxSize[0] = Math.min(realBoxSize[0], this.max_x_grids - 2)
        this._liveBoxSize[1] = Math.min(realBoxSize[1], this.max_y_grids - 2)
        this._liveBoxSize[2] = Math.min(realBoxSize[2], this.max_z_grids - 2)
        this._writeSimulationParams()
    }

    setSubsteps(n) {
        this.substeps = n | 0
    }

    addSphere(centerX, centerY, centerZ, radius, numSphereParticles) {
        if (this.numParticles + numSphereParticles > numParticlesMax) {
            return
        }

        const sphereParticlesBuf = new ArrayBuffer(fluidParticleStructSize * numSphereParticles)
        const spacing = 0.35
        let sphereParticleCount = 0

        for (let x = -radius; x <= radius && sphereParticleCount < numSphereParticles; x += spacing) {
            for (let y = -radius; y <= radius && sphereParticleCount < numSphereParticles; y += spacing) {
                for (let z = -radius; z <= radius && sphereParticleCount < numSphereParticles; z += spacing) {
                    if (Math.sqrt(x * x + y * y + z * z) > radius) {
                        continue
                    }

                    const offset = fluidParticleStructSize * sphereParticleCount
                    const particleViews = {
                        position: new Float32Array(sphereParticlesBuf, offset + 0, 3),
                        velocity: new Float32Array(sphereParticlesBuf, offset + 16, 3),
                        affine: new Float32Array(sphereParticlesBuf, offset + 32, 12),
                    }

                    particleViews.position.set([
                        Math.min(Math.max(centerX + x, 2), this._liveBoxSize[0] - 3),
                        Math.min(Math.max(centerY + y, 2), this._liveBoxSize[1] - 3),
                        Math.min(Math.max(centerZ + z, 2), this._liveBoxSize[2] - 3),
                    ])
                    particleViews.velocity.set([0, 0, 0])
                    particleViews.affine.fill(0)
                    sphereParticleCount++
                }
            }
        }

        const offset = this.numParticles * fluidParticleStructSize
        const sphereData = new Uint8Array(sphereParticlesBuf, 0, sphereParticleCount * fluidParticleStructSize)
        this.device.queue.writeBuffer(this.particleBuffer, offset, sphereData)
        this.numParticles += sphereParticleCount
    }
}
