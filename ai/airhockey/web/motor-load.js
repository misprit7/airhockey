// Independent of the control socket: also usable during policies and probes.
(() => {
    const cards = document.getElementById('motor-load-cards');
    const badge = document.getElementById('motor-load-state');
    const summary = document.getElementById('motor-load-summary');
    let lastResponse = 0;
    for (let node = 0; node < 4; node++) {
        const card = document.createElement('div');
        card.className = 'motor-load-card';
        card.innerHTML = `<div class="motor-load-heading">Motor ${node}<small id="motor-${node}-state">Unknown</small></div>
            <div class="motor-load-electric"><strong id="motor-${node}-torque_amps">—</strong><span id="motor-${node}-bus_voltage_v">—</span></div>
            <div class="motor-load-rms"><span>Fast RMS</span><span id="motor-${node}-rms_pct">—</span></div>
            <div class="motor-load-rms"><span>Slow RMS</span><span id="motor-${node}-rms_slow_pct">—</span></div>`;
        cards.appendChild(card);
    }
    function render(data) {
        const live = data.state === 'live';
        badge.textContent = live ? 'Live' : data.state === 'stale' ? 'Stale' : 'Unavailable';
        badge.className = live ? 'live' : '';
        const context = data.context || {};
        summary.textContent = live
            ? (context.fault ? `Drive fault · motor ${context.fault_node}` : context.motors_enabled ? 'Drives enabled' : 'Drives disabled')
            : data.message || 'No fresh readings. Start cdpr_master or a hardware run.';
        summary.className = live && context.fault ? 'load-danger' : '';
        summary.title = data.source ? `Source: ${data.source}` : '';
        for (let node = 0; node < 4; node++) {
            const motor = (data.motors || []).find(m => m.node === node) || {};
            const status = document.getElementById(`motor-${node}-state`);
            const alerts = motor.alerts;
            const fault = live && alerts?.fresh && alerts.value.some(v => v !== 0);
            status.textContent = fault ? 'Alert' : live && motor.status?.fresh
                ? (motor.status.value[0] === 1 ? 'Enabled' : 'Disabled') : 'Unknown';
            status.className = fault ? 'load-danger' : '';
            status.title = fault ? `Alert bits: ${alerts.value.map(v => v.toString(16)).join(' ')}` : '';
            for (const key of ['torque_amps', 'bus_voltage_v', 'rms_pct', 'rms_slow_pct']) {
                const el = document.getElementById(`motor-${node}-${key}`);
                const field = motor[key];
                const valid = live && field?.fresh && Number.isFinite(field.value);
                const rms = key.startsWith('rms');
                const unit = rms ? '%' : key === 'torque_amps' ? ' A' : ' V';
                const value = field?.value;
                el.textContent = valid ? value.toFixed(rms ? 0 : key === 'torque_amps' ? 2 : 1) + unit : '—';
                el.title = valid ? `Sample age ${Math.round(field.age_s * 1000)} ms`
                    : Number.isFinite(value) ? `Last reading ${value.toFixed(2)}${unit}; stale` : 'No valid reading';
                el.className = valid && rms && value >= 85 ? 'load-danger'
                    : valid && rms && value >= 70 ? 'load-warn' : '';
                el.style.backgroundSize = valid && rms ? `${Math.max(0, Math.min(100, value))}% 100%` : '0% 100%';
            }
        }
    }
    async function poll() {
        try {
            const response = await fetch('/api/motor-load', {cache: 'no-store', signal: AbortSignal.timeout(1500)});
            if (!response.ok) throw Error(`HTTP ${response.status}`);
            const data = await response.json();
            lastResponse = performance.now();
            render(data);
        } catch (_) {
            render({state: 'unavailable', message: 'Telemetry connection unavailable.'});
        } finally {
            setTimeout(poll, 250);
        }
    }
    // Expire visible values even when a request hangs or the server stops.
    setInterval(() => {
        if (lastResponse && performance.now() - lastResponse > 1500)
            render({state: 'stale', message: 'Telemetry connection stale.'});
    }, 250);
    poll();
})();
