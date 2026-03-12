export class FluidRenderer {
    constructor(
        device, canvas, presentationFormat,
        radius, fov, posvelBuffer, 
        renderUniformBuffer,
        visibilityBuffer
    ) {
        this.device = device
        this.canvas = canvas
        this.presentationFormat = presentationFormat
        this.posvelBuffer = posvelBuffer
        this.renderUniformBuffer = renderUniformBuffer
        this.visibilityBuffer = visibilityBuffer
        this.wireframeEnabled = false
        this.cachedColorView = null
        this.lastTexture = null
    }

    async initialize() {
        const sphere = await fetch('render/sphere.wgsl?v=20260309i').then(r => r.text());
        const wireframe = await fetch('render/wireframe.wgsl?v=20260309i').then(r => r.text());
        const wall = await fetch('render/wall.wgsl?v=20260309i').then(r => r.text());
        const sphereModule = this.device.createShaderModule({ code: sphere })
        const wireframeModule = this.device.createShaderModule({ code: wireframe })
        const wallModule = this.device.createShaderModule({ code: wall })

        this.spherePipeline = this.device.createRenderPipeline({
            label: 'sphere pipeline', 
            layout: 'auto', 
            vertex: { module: sphereModule }, 
            fragment: {
                module: sphereModule, 
                targets: [{ format: this.presentationFormat }]
            }, 
            primitive: { topology: 'triangle-list' },
            depthStencil: {
                depthWriteEnabled: true, 
                depthCompare: 'less',
                format: 'depth32float'
            }
        })

        this.wireframePipeline = this.device.createRenderPipeline({
            label: 'wireframe pipeline', 
            layout: 'auto', 
            vertex: { module: wireframeModule }, 
            fragment: {
                module: wireframeModule, 
                targets: [{ format: this.presentationFormat }]
            }, 
            primitive: { 
                topology: 'line-list',
                stripIndexFormat: undefined
            },
            depthStencil: {
                depthWriteEnabled: true, 
                depthCompare: 'less',
                format: 'depth32float'
            }
        })

        this.wallPipeline = this.device.createRenderPipeline({
            label: 'wall pipeline',
            layout: 'auto',
            vertex: { module: wallModule },
            fragment: {
                module: wallModule,
                targets: [{ format: this.presentationFormat }]
            },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth32float'
            }
        })

        this._createDepthTexture(this.canvas.width, this.canvas.height);

        this.sphereBindGroup = this.device.createBindGroup({
            label: 'sphere bind group', 
            layout: this.spherePipeline.getBindGroupLayout(0),  
            entries: [
                { binding: 0, resource: { buffer: this.posvelBuffer }},
                { binding: 1, resource: { buffer: this.renderUniformBuffer }},
                { binding: 2, resource: { buffer: this.visibilityBuffer }},
            ]
        })

        this.wireframeBindGroup = this.device.createBindGroup({
            label: 'wireframe bind group', 
            layout: this.wireframePipeline.getBindGroupLayout(0),  
            entries: [
                { binding: 0, resource: { buffer: this.posvelBuffer }},
                { binding: 1, resource: { buffer: this.renderUniformBuffer }},
                { binding: 2, resource: { buffer: this.visibilityBuffer }},
            ]
        })

    }

    setWireframeMode(enabled) {
        this.wireframeEnabled = enabled;
    }

    resize(width, height) {
        // Recreate depth texture with new size and drop cached color view
        this._createDepthTexture(width, height);
        this.cachedColorView = null;
        this.lastTexture = null;
    }

    _createDepthTexture(width, height) {
        const depthTestTexture = this.device.createTexture({
            size: [Math.max(1, width), Math.max(1, height), 1],
            format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.depthTestTextureView = depthTestTexture.createView();
    }

    setVisibilityBuffer(buffer) {
        this.visibilityBuffer = buffer;
        // Recreate bind groups with new buffer
        this.sphereBindGroup = this.device.createBindGroup({
            label: 'sphere bind group', 
            layout: this.spherePipeline.getBindGroupLayout(0),  
            entries: [
                { binding: 0, resource: { buffer: this.posvelBuffer }},
                { binding: 1, resource: { buffer: this.renderUniformBuffer }},
                { binding: 2, resource: { buffer: this.visibilityBuffer }},
            ]
        })

        this.wireframeBindGroup = this.device.createBindGroup({
            label: 'wireframe bind group', 
            layout: this.wireframePipeline.getBindGroupLayout(0),  
            entries: [
                { binding: 0, resource: { buffer: this.posvelBuffer }},
                { binding: 1, resource: { buffer: this.renderUniformBuffer }},
                { binding: 2, resource: { buffer: this.visibilityBuffer }},
            ]
        })

        this.wallBindGroup = this.device.createBindGroup({
            label: 'wall bind group',
            layout: this.wallPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.renderUniformBuffer }},
            ]
        })
    }

    execute(context, commandEncoder, numParticles) {
        const currentTexture = context.getCurrentTexture();
        
        if (this.lastTexture !== currentTexture) {
            this.cachedColorView = currentTexture.createView();
            this.lastTexture = currentTexture;
        }
        
        const renderPassDescriptor = {
            colorAttachments: [
                {
                    view: this.cachedColorView,
                    clearValue: { r: 0.8, g: 0.8, b: 0.8, a: 1.0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
            depthStencilAttachment: {
                view: this.depthTestTextureView,
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        }
        
        const renderPassEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

        renderPassEncoder.setBindGroup(0, this.wallBindGroup);
        renderPassEncoder.setPipeline(this.wallPipeline);
        renderPassEncoder.draw(31 * 36);

        if (this.wireframeEnabled) {
            renderPassEncoder.setBindGroup(0, this.wireframeBindGroup);
            renderPassEncoder.setPipeline(this.wireframePipeline);
            renderPassEncoder.draw(96, numParticles);
        } else {
            renderPassEncoder.setBindGroup(0, this.sphereBindGroup);
            renderPassEncoder.setPipeline(this.spherePipeline);
            renderPassEncoder.draw(6, numParticles);
        }
        
        renderPassEncoder.end();
    }
}
