(() => {
    'use strict';

    const root = document.getElementById('thesis-animation');
    if (!root) return;

    const canvas = root.querySelector('canvas');
    const context = canvas.getContext('2d');
    if (!context) {
        canvas.hidden = true;
        return;
    }

    const controls = root.querySelector('.thesis-controls');
    const scatter = root.querySelector('[data-scatter]');
    const status = root.querySelector('.thesis-status');
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    const background = document.createElement('canvas');
    const backgroundContext = background.getContext('2d');
    const surface = document.createElement('canvas');
    const surfaceContext = surface.getContext('2d');

    // Orthographic projection keeps the drawing lightweight: no 3D library.
    const yaw = -0.48;
    const pitch = 0.60;
    const cos = Math.cos(yaw);
    const sin = Math.sin(yaw);
    const cp = Math.cos(pitch);
    const sp = Math.sin(pitch);
    const radius = 0.305;
    const sigma = 0.026;
    const plane = 0.455;
    const heightScale = 0.076;
    const gridSize = 52;
    const stride = gridSize + 1;
    const bandwidth = 0.033;
    const extent = 3 * bandwidth;
    const stepDuration = 1 / 30;
    const loopDuration = 90;

    let running = !reducedMotion.matches;
    let visible = false;
    let seed = 1547;
    let time = 0;
    let lastFrame = null;
    let accumulator = 0;
    let request = null;
    let fitClock = 0;
    let surfaceClock = 0;
    let width = 0;
    let height = 0;
    let dpr = 1;
    let scale = 1;
    let colors = {};
    let particles = [];
    let fitted = new Float64Array(stride * stride);
    let target = new Float64Array(stride * stride);

    function random() {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        return (seed + 0.5) / 4294967296;
    }

    function normal() {
        return Math.sqrt(-2 * Math.log(random())) * Math.cos(2 * Math.PI * random());
    }

    function reflect(value) {
        if (value > 0.5) return 1 - value;
        if (value < -0.5) return -1 - value;
        return value;
    }

    function project(x, y, z = 0) {
        const a = x * cos - y * sin;
        const b = x * sin + y * cos;
        return {
            x: width / 2 + a * scale,
            y: height * 0.72 + (b * sp - z * cp) * scale,
            depth: b * cp + z * sp,
        };
    }

    function line(ctx, points, color, alpha, lineWidth = 1) {
        ctx.beginPath();
        points.forEach((point, index) => {
            if (index) ctx.lineTo(point.x, point.y);
            else ctx.moveTo(point.x, point.y);
        });
        ctx.strokeStyle = color;
        ctx.globalAlpha = alpha;
        ctx.lineWidth = lineWidth;
        ctx.stroke();
        ctx.globalAlpha = 1;
    }

    function polygon(ctx, points, color, alpha) {
        ctx.beginPath();
        points.forEach((point, index) => {
            if (index) ctx.lineTo(point.x, point.y);
            else ctx.moveTo(point.x, point.y);
        });
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.globalAlpha = alpha;
        ctx.fill();
        ctx.globalAlpha = 1;
    }

    // Reflect kernels at the square's edges to avoid losing density there.
    function kernelWeights(value) {
        const sources = [value];
        if (value < -0.5 + extent) sources.push(-1 - value);
        if (value > 0.5 - extent) sources.push(1 - value);
        const weights = new Map();
        for (const source of sources) {
            const low = Math.max(0, Math.ceil((source - extent + 0.5) * gridSize));
            const high = Math.min(gridSize, Math.floor((source + extent + 0.5) * gridSize));
            for (let i = low; i <= high; i++) {
                const distance = (-0.5 + i / gridSize - source) / bandwidth;
                weights.set(i, (weights.get(i) || 0) + Math.exp(-0.5 * distance * distance));
            }
        }
        return [...weights.entries()];
    }

    function estimateDensity() {
        const result = new Float64Array(stride * stride);
        for (const particle of particles) {
            const wx = kernelWeights(particle.x);
            const wy = kernelWeights(particle.y);
            for (const [i, a] of wx) {
                for (const [j, b] of wy) result[i * stride + j] += a * b;
            }
        }
        // Normalise using a trapezoidal integral on the square.
        let mass = 0;
        for (let i = 0; i <= gridSize; i++) {
            for (let j = 0; j <= gridSize; j++) {
                const edgeX = i === 0 || i === gridSize ? 0.5 : 1;
                const edgeY = j === 0 || j === gridSize ? 0.5 : 1;
                mass += result[i * stride + j] * edgeX * edgeY / (gridSize * gridSize);
            }
        }
        for (let k = 0; k < result.length; k++) result[k] /= mass;
        return result;
    }

    function drawFloor(z, alpha) {
        polygon(backgroundContext, [
            project(-0.5, -0.5, z), project(0.5, -0.5, z),
            project(0.5, 0.5, z), project(-0.5, 0.5, z),
        ], colors.floor, alpha);
        for (let i = 0; i <= 10; i++) {
            const value = -0.5 + i / 10;
            const opacity = i === 0 || i === 10 ? 0.19 : 0.055;
            line(backgroundContext, [project(value, -0.5, z), project(value, 0.5, z)], colors.ink, opacity, 0.7);
            line(backgroundContext, [project(-0.5, value, z), project(0.5, value, z)], colors.ink, opacity, 0.7);
        }
    }

    function buildBackground() {
        background.width = canvas.width;
        background.height = canvas.height;
        backgroundContext.setTransform(dpr, 0, 0, dpr, 0, 0);
        drawFloor(0, 0.16);
        for (const [x, y] of [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) {
            line(backgroundContext, [project(x, y), project(x, y, plane)], colors.ink, 0.075, 0.8);
        }
        drawFloor(plane, 0.10);
    }

    function buildSurface() {
        if (!width) return;
        surface.width = canvas.width;
        surface.height = canvas.height;
        surfaceContext.setTransform(dpr, 0, 0, dpr, 0, 0);
        const vertices = Array.from(fitted, (value, k) => project(
            -0.5 + Math.floor(k / stride) / gridSize,
            -0.5 + (k % stride) / gridSize,
            plane + heightScale * value,
        ));
        const cells = [];
        for (let i = 0; i < gridSize; i++) {
            for (let j = 0; j < gridSize; j++) {
                const indices = [i * stride + j, (i + 1) * stride + j, (i + 1) * stride + j + 1, i * stride + j + 1];
                const values = indices.map(k => fitted[k]);
                if (Math.max(...values) < 0.003) continue;
                const points = indices.map(k => vertices[k]);
                cells.push({
                    points,
                    value: values.reduce((a, b) => a + b, 0) / 4,
                    depth: points.reduce((a, b) => a + b.depth, 0) / 4,
                });
            }
        }
        cells.sort((a, b) => a.depth - b.depth);
        for (const cell of cells) {
            polygon(surfaceContext, cell.points, colors.series, 0.13 + 0.22 * Math.min(1, cell.value / 5.5));
        }
        for (let i = 0; i <= gridSize; i += 2) {
            const row = [];
            const column = [];
            for (let j = 0; j <= gridSize; j++) {
                row.push(vertices[i * stride + j]);
                column.push(vertices[j * stride + i]);
            }
            line(surfaceContext, row, colors.series, 0.24, 0.65);
            line(surfaceContext, column, colors.series, 0.24, 0.65);
        }
    }

    function draw() {
        if (!width) return;
        context.setTransform(dpr, 0, 0, dpr, 0, 0);
        context.clearRect(0, 0, width, height);
        context.drawImage(background, 0, 0, width, height);
        context.drawImage(surface, 0, 0, width, height);
        for (const particle of particles) {
            if (particle.trail && particle.trail.length > 2) {
                line(context, particle.trail.map(([x, y]) => project(x, y)), colors.series, 0.20);
            }
        }
        context.fillStyle = colors.series;
        context.globalAlpha = 0.73;
        for (const particle of particles) {
            const point = project(particle.x, particle.y);
            context.beginPath();
            context.arc(point.x, point.y, width < 400 ? 1.15 : 1.45, 0, Math.PI * 2);
            context.fill();
        }
        context.globalAlpha = 1;
    }

    function updateStatus() {
        const value = time < 36 ? 'Fitting density · converging' : 'Circulating · steady density estimate';
        if (status.textContent !== value) status.textContent = value;
    }

    function reset(settled = false) {
        time = settled ? 36 : 0;
        accumulator = 0;
        fitClock = 0;
        surfaceClock = 0;
        particles = Array.from({ length: 850 }, (_, i) => {
            let x = (random() - 0.5) * 0.98;
            let y = (random() - 0.5) * 0.98;
            if (settled) {
                const angle = random() * Math.PI * 2;
                const r = radius + sigma * normal();
                x = r * Math.cos(angle);
                y = r * Math.sin(angle);
            }
            return { x, y, trail: i < 38 ? [[x, y]] : null };
        });
        target = estimateDensity();
        fitted = target.slice();
        buildSurface();
        updateStatus();
        draw();
    }

    function step(dt) {
        time += dt;
        if (time >= loopDuration) {
            reset();
            return;
        }
        const pull = 0.105;
        const noise = Math.sqrt(2 * pull) * sigma * Math.sqrt(dt);
        for (const particle of particles) {
            const r = Math.hypot(particle.x, particle.y);
            const angle = Math.atan2(particle.y, particle.x) + 0.65 * dt + 0.065 * Math.sqrt(dt) * normal();
            const next = Math.abs(radius + (r - radius) * Math.exp(-pull * dt) + noise * normal());
            particle.x = reflect(next * Math.cos(angle));
            particle.y = reflect(next * Math.sin(angle));
            if (particle.trail) {
                particle.trail.push([particle.x, particle.y]);
                if (particle.trail.length > 26) particle.trail.shift();
            }
        }
        fitClock += dt;
        surfaceClock += dt;
        // Refit five times a second and smoothly track the latest estimate.
        if (fitClock >= 0.20) {
            target = estimateDensity();
            fitClock = 0;
        }
        const blend = 1 - Math.exp(-dt / 1.1);
        for (let k = 0; k < fitted.length; k++) fitted[k] += (target[k] - fitted[k]) * blend;
        if (surfaceClock >= 1 / 15) {
            buildSurface();
            surfaceClock = 0;
        }
        updateStatus();
    }

    function resize() {
        const rect = canvas.getBoundingClientRect();
        width = rect.width;
        height = rect.height;
        dpr = Math.min(window.devicePixelRatio || 1, 2);
        scale = Math.min(width * 0.61, height * 0.66);
        canvas.width = Math.round(width * dpr);
        canvas.height = Math.round(height * dpr);
        const style = getComputedStyle(root);
        colors = {
            series: style.getPropertyValue('--thesis-series').trim(),
            ink: style.getPropertyValue('--thesis-ink').trim(),
            floor: style.getPropertyValue('--thesis-floor').trim(),
        };
        buildBackground();
        buildSurface();
        draw();
    }

    function frame(now) {
        request = null;
        if (!running || !visible || document.hidden) {
            lastFrame = null;
            return;
        }
        let advanced = false;
        if (lastFrame !== null) {
            accumulator += Math.min((now - lastFrame) / 1000, 0.08);
            while (accumulator >= stepDuration) {
                accumulator -= stepDuration;
                step(stepDuration);
                advanced = true;
            }
        }
        lastFrame = now;
        if (advanced) draw();
        request = requestAnimationFrame(frame);
    }

    function schedule() {
        if (running && visible && !document.hidden && request === null) {
            request = requestAnimationFrame(frame);
        } else if ((!running || !visible || document.hidden) && request !== null) {
            cancelAnimationFrame(request);
            request = null;
        }
    }

    scatter.addEventListener('click', () => {
        reset(reducedMotion.matches);
        running = !reducedMotion.matches;
        lastFrame = null;
        updateStatus();
        schedule();
    });
    document.addEventListener('visibilitychange', () => {
        lastFrame = null;
        schedule();
    });
    new IntersectionObserver(entries => {
        visible = entries[0].isIntersecting;
        lastFrame = null;
        schedule();
    }).observe(canvas);
    new ResizeObserver(resize).observe(canvas);
    reducedMotion.addEventListener('change', () => {
        running = !reducedMotion.matches;
        lastFrame = null;
        updateStatus();
        schedule();
    });

    controls.hidden = false;
    reset(reducedMotion.matches);
    resize();
    schedule();
})();
