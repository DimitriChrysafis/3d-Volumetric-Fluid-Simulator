import { Camera } from './camera.js'
import { MLSMPMSimulator, mlsmpmParticleStructSize } from './mls-mpm/mls-mpm.js'
import { FluidRenderer } from './render/fluidRender.js'
import { renderUniformsValues, renderUniformsViews, numParticlesMax } from './common.js'
import { FrustumCuller } from './optimization/frustumCulling.js'
import { mat4 } from 'https://wgpu-matrix.org/dist/3.x/wgpu-matrix.module.min.js'

const BOX_WIDTH = 100;
const BOX_HEIGHT = 100;
const BOX_DEPTH = 100;

async function init() {
  const canvas = document.querySelector('canvas');

  if (!navigator.gpu) {
    alert("WebGPU is not supported on your browser.");
    throw new Error();
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    alert("Adapter is not available.");
    throw new Error();
  }

  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    }
  });
  const context = canvas.getContext('webgpu');

  if (!context) {
    throw new Error();
  }

  let devicePixelRatio = Math.min(2, (window.devicePixelRatio || 1));
  const resizeCanvas = () => {
    // Recompute DPR on resize to adapt to user changes
    devicePixelRatio = Math.min(2, (window.devicePixelRatio || 1));
    canvas.width = Math.max(1, Math.floor(devicePixelRatio * canvas.clientWidth));
    canvas.height = Math.max(1, Math.floor(devicePixelRatio * canvas.clientHeight));
  };
  resizeCanvas();

  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format: presentationFormat, alphaMode: 'opaque' });

  // Keep canvas/resolution in sync
  window.addEventListener('resize', () => {
    const prevW = canvas.width, prevH = canvas.height;
    resizeCanvas();
    if (canvas.width !== prevW || canvas.height !== prevH) {
      // Trigger renderer resize in main() after it is created
      if (window.__renderer && window.__renderer.resize) {
        window.__renderer.resize(canvas.width, canvas.height);
      }
      renderUniformsViews.texel_size.set([1.0 / canvas.width, 1.0 / canvas.height]);
    }
  });

  return { canvas, device, presentationFormat, context };
}

async function main() {
  while (!window.dat || !window.Stats) {
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  
  try {
    const { canvas, device, presentationFormat, context } = await init();
    const stats = new window.Stats();
    stats.showPanel(0);
    stats.dom.style.position = 'fixed';
    stats.dom.style.left = '0px';
    stats.dom.style.top = '0px';
    stats.dom.style.zIndex = '100';
    document.body.appendChild(stats.dom);


    const canvasElement = document.getElementById("fluidCanvas");
    const fov = 45 * Math.PI / 180;
    const radius = 0.75;
    const diameter = 2 * radius;
    const zoomRate = 10.0;

    renderUniformsViews.texel_size.set([1.0 / canvas.width, 1.0 / canvas.height]);
    renderUniformsViews.sphere_size.set([radius, radius]);

    const particleBuffer = device.createBuffer({
      label: 'particles buffer',
      size: mlsmpmParticleStructSize * numParticlesMax,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const posvelBuffer = device.createBuffer({
      label: 'position buffer',
      size: 32 * numParticlesMax,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const renderUniformBuffer = device.createBuffer({
      label: 'render uniform buffer',
      size: renderUniformsValues.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const simulator = new MLSMPMSimulator(particleBuffer, posvelBuffer, diameter, device, BOX_WIDTH, BOX_HEIGHT, BOX_DEPTH);
    await simulator.initialize();

    // Setup frustum culling and default visibility (all visible)
    const culler = new FrustumCuller(device);
    await culler.initialize(numParticlesMax);
    const ones = new Uint32Array(numParticlesMax);
    ones.fill(1);
    device.queue.writeBuffer(culler.visibilityBuffer, 0, ones);
    const cullBindGroup = culler.createBindGroup(posvelBuffer, renderUniformBuffer);

    const renderer = new FluidRenderer(device, canvas, presentationFormat, radius, fov, posvelBuffer, renderUniformBuffer, culler.visibilityBuffer);
    await renderer.initialize();
    // Expose to resize handler
    window.__renderer = renderer;
    
    const camera = new Camera(canvasElement);
    
    let isPaused = false;
    let waveEnabled = true;
    let waveTime = 0;
    let waveAmplitude = 0.40;
    // let boxDepth = 100; // unused
    
    let wireframeEnabled = true;
    let boundingBoxEnabled = true;
    
    renderer.setBoundingBoxMode(boundingBoxEnabled);
    
    renderer.setWireframeMode(wireframeEnabled);
    
    const gui = new dat.GUI();
    
    const simulationFolder = gui.addFolder('Simulation');
    simulationFolder.add({ isPaused: isPaused }, 'isPaused').name('Pause Simulation').onChange((value) => {
      isPaused = value;
    });
    const simState = { substeps: 2 };
    simulationFolder.add(simState, 'substeps', 1, 4, 1).name('Substeps').onChange((v) => {
      simulator.setSubsteps(v|0);
    });
    
    simulationFolder.add({ 
      addParticles: () => addMoreParticles() 
    }, 'addParticles').name('Add 10,000 Particles');
    
    simulationFolder.open();
    
    const waveFolder = gui.addFolder('Wave Controls');
    waveFolder.add({ waveEnabled: waveEnabled }, 'waveEnabled').name('Enable Moving Wall').onChange((value) => {
      waveEnabled = value;
      if (waveEnabled) {
        waveTime = 0;
      } else {
      }
    });
    waveFolder.add({ amplitude: waveAmplitude }, 'amplitude', 0.0, 1.0, 0.01).name('Wave Amplitude').onChange((v) => {
      waveAmplitude = v;
    });
    
    waveFolder.open();
    
    const cameraFolder = gui.addFolder('Camera');
    let cameraMode = 'orbit';
    cameraFolder.add({ cameraMode: cameraMode }, 'cameraMode', ['orbit', 'coolcal']).name('Camera Mode').onChange((value) => {
      cameraMode = value;
      camera.setCameraMode(value);
    });
    cameraFolder.open();
    
    const renderingFolder = gui.addFolder('Rendering');
    renderingFolder.add({ wireframeEnabled: wireframeEnabled }, 'wireframeEnabled').name('potato friendly mode').onChange((value) => {
      wireframeEnabled = value;
      if (renderer && renderer.setWireframeMode) {
        renderer.setWireframeMode(wireframeEnabled);
      }
    });

    renderingFolder.add({ boundingBoxEnabled: boundingBoxEnabled }, 'boundingBoxEnabled').name('Show Bounding Box').onChange((value) => {
      boundingBoxEnabled = value;
      if (renderer && renderer.setBoundingBoxMode) {
        renderer.setBoundingBoxMode(boundingBoxEnabled);
      }
    });
    const renderState = { resolutionScale: 1.0, cullingEnabled: true };
    renderingFolder.add(renderState, 'resolutionScale', 0.5, 2.0, 0.1).name('Resolution Scale').onChange((s) => {
      const w = Math.max(1, Math.floor(devicePixelRatio * canvas.clientWidth * s));
      const h = Math.max(1, Math.floor(devicePixelRatio * canvas.clientHeight * s));
      canvas.width = w;
      canvas.height = h;
      if (renderer && renderer.resize) {
        renderer.resize(w, h);
      }
      renderUniformsViews.texel_size.set([1.0 / canvas.width, 1.0 / canvas.height]);
    });
    renderingFolder.add(renderState, 'cullingEnabled').name('Enable Frustum Culling');
    
    renderingFolder.open();
    
    function addMoreParticles() {
      const centerX = BOX_WIDTH / 2;
      const centerY = BOX_HEIGHT / 2;
      const centerZ = BOX_DEPTH / 2;
      const radius = 5;
      const numSphereParticles = 10000;
      
      simulator.addSphere(centerX, centerY, centerZ, radius, numSphereParticles);
    }
    
    document.addEventListener('keydown', function(event) {
      if (event.key.toLowerCase() === 'p') {
        event.preventDefault();
        isPaused = !isPaused;
        gui.updateDisplay();
      }
    });
    
    document.addEventListener('keydown', function(event) {
      if (event.key.toLowerCase() === 'g') {
        addMoreParticles();
      }
    });

    let errorLog = document.getElementById('error-reason');
    errorLog.textContent = "";
    device.lost.then(info => {
      const reason = info.reason ? `reason: ${info.reason}` : 'unknown reason';
      errorLog.textContent = reason;
    });

    let currentParticleCount = 400000;
    let initBoxSize = [BOX_WIDTH, BOX_HEIGHT, BOX_DEPTH];
    let realBoxSize = [...initBoxSize];
    simulator.reset(currentParticleCount, initBoxSize);
    
    camera.reset(canvasElement, 150, [BOX_WIDTH / 2, BOX_HEIGHT / 4, BOX_DEPTH / 2], fov, zoomRate);

    let boxWidthRatio = 1.0;
    let prevBoxSize = [...realBoxSize];

    let lastTime = performance.now();
    async function frame(currentTime) {
      stats.begin();

      const deltaTime = (currentTime - lastTime) / 1000;
      lastTime = currentTime;

      // Always allow camera updates regardless of pause state
      camera.update(deltaTime);

      if (!isPaused) {
        if (waveEnabled) {
          waveTime += 0.02;
          boxWidthRatio = 1.0 + Math.sin(waveTime) * waveAmplitude;

          boxWidthRatio = Math.max(0.2, Math.min(2.0, boxWidthRatio));

          realBoxSize[2] = initBoxSize[2] * boxWidthRatio;
          simulator.changeBoxSize(realBoxSize);
        }
      }

      // Always update uniforms every frame to ensure camera works while paused
      renderUniformsViews.box_size.set(realBoxSize);
      device.queue.writeBuffer(renderUniformBuffer, 0, renderUniformsValues);
      
      // Update tracking variables
      if (realBoxSize[0] !== prevBoxSize[0] || realBoxSize[1] !== prevBoxSize[1] || realBoxSize[2] !== prevBoxSize[2]) {
        prevBoxSize = [...realBoxSize];
      }

      const commandEncoder = device.createCommandEncoder();

      if (!isPaused) {
        simulator.execute(commandEncoder);
      }
      // Frustum culling before rendering (optional)
      {
        const P = renderUniformsViews.projection_matrix;
        const V = renderUniformsViews.view_matrix;
        const VP = mat4.multiply(P, V);
        if (renderState.cullingEnabled) {
          culler.execute(commandEncoder, cullBindGroup, simulator.numParticles, VP);
        } else {
          // ensure first value visible (buffer is already ones initially)
          device.queue.writeBuffer(culler.visibilityBuffer, 0, new Uint32Array([1]));
        }
      }
      renderer.execute(context, commandEncoder, simulator.numParticles);

      device.queue.submit([commandEncoder.finish()]);
      
      stats.end();
      requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
    
    // No-op interval removed
  } catch (error) {
  }
}

main();
