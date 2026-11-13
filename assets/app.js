(() => {
    'use strict';
    const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
    const { meshes, manifest } = window.__PTERIONAL_DATA__;
    const papers = manifest.papers;
    const escape = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const mobileQuery = matchMedia('(max-width:760px)');
    let isMobile = mobileQuery.matches, desktopLabelPreference = true, mobileLabelPreference = false, viewZoom = 1;
    let config = { clinoid: 'none', muscle: 'subfascial', fissure: 'yes' }, steps = [], stage = 0, mode = '3d', selected = -1, contextAlpha = 1, contextSide = 'right', procedure = true, prelabelsOn = !isMobile, opening = 1, animationToken = 0, separateLayers = false, instrumentsOn = true, capturing = false, beforeCurrent = false, regionalContext = true, contextLayerSnapshot = null;
    const enabled = { skin: true, galea: true, pericranium: true, bone: true, brain: true, muscle: true, fascia: true, deepfascia: true, fat: true, arteries: true, veins: true, nerves: true, dura: true, arachnoid: true };
    // Surgical palette: approximate unlit tissue albedo (sRGB) used by the material shader; Atlas keeps the manifest colours.
    const palettes = { atlas: { skin: '#d9a184', galea: '#d8dad3', pericranium: '#dcd6e4', bone: '#e2d5b8', brain: '#bca99d', muscle: '#b87367', fascia: '#a4c7ba', deepfascia: '#75a99a', fat: '#debd69', arteries: '#dd6268', veins: '#5f6fae', nerves: '#f2d776', dura: '#95b7bb', arachnoid: '#cfe3ea' }, surgical: { skin: '#cf9a7f', galea: '#dcdbd3', pericranium: '#d8d9e0', bone: '#dccdb0', brain: '#d4aa9c', muscle: '#82302a', fascia: '#e2ddd1', deepfascia: '#d9dcd8', fat: '#e3c677', arteries: '#c4423d', veins: '#5e3e66', nerves: '#ede3c6', dura: '#c4bcb5', arachnoid: '#e6eef1' } };
    // Distinct Surgical tones for the temporal fat pads and fascial laminae, so the separated planes stay
    // distinguishable (the Atlas palette keeps the manifest colours).
    const surgicalLayerTints = { 'ILL-subfat': '#d2a256', 'ILL-interfat': '#e3c677', 'ILL-suprafat': '#ece0bd', 'ILL-superfascia': '#e8e3d6', 'ILL-deepfascia': '#c3c8c4' };
    const materialIds = { bone: 0, brain: 1, muscle: 2, fascia: 3, deepfascia: 3, fat: 4, arteries: 5, nerves: 6, dura: 7, veins: 8, arachnoid: 9, skin: 10, galea: 11, pericranium: 12 };
    let palette = 'surgical';
    try {
        if (localStorage.getItem('pterional-palette') === 'atlas')
            palette = 'atlas';
    }
    catch (error) { }
    const hexRgb = c => c.match(/\w\w/g).map(x => parseInt(x, 16) / 255);
    const surgicalRgb = Object.fromEntries(Object.entries(palettes.surgical).map(([k, c]) => [k, hexRgb(c)]));
    const allEnabled = Object.fromEntries(Object.keys(enabled).map(k => [k, false]));
    const canvas = $('#canvas'), host = $('#viewport');
    // Camera state: viewQ/camFrame/target are described with the orientation helpers below; distance, pan (screen
    // space, after rotation) and viewZoom keep their original meaning.
    let gl = null, program = null, uniforms = {}, buffer = null, available = false, viewQ = qFromAngles(1.0, .25, 0), camFrame = 'head', distance = 15, pan = [0, 0], target = [-1, 0, .5], aspect = 1, scheduled = false, pickDirty = true, interactionActive = false, interactionTimer = 0;
    const scale = 2.05, LINE_MIN_HALF_PX = 1.3, LINE_NUDGE = .03;
    const configs = [];
    for (const c of ['none', 'extra', 'intra'])
        for (const m of ['subfascial', 'split'])
            for (const f of ['yes', 'no'])
                configs.push({ clinoid: c, muscle: m, fissure: f });
    const cname = c => ({ none: 'No clinoidectomy', extra: 'Extradural clinoidectomy', intra: 'Intradural clinoidectomy' }[c]);
    const mname = m => m === 'split' ? 'Muscle-splitting' : 'Subfascial';
    const configName = c => `${cname(c.clinoid)} · ${mname(c.muscle)} · ${c.fissure === 'yes' ? 'Split fissure' : 'No fissure split'}`;
    function operativeSide() { return contextSide === 'left' ? 'left' : 'right'; }
    function sideSign() { return operativeSide() === 'left' ? -1 : 1; }
    function canonicalPoint(p) { return [p[0] * sideSign(), p[1], p[2]]; }
    // ---- Free rotation: orientation helpers -----------------------------------------------------------------------
    // Rotations are row-major 3x3 arrays [m00, m01, m02, m10, ...]; quaternions are [x, y, z, w].
    // View space: +x right, +y up, +z toward the viewer. The eye sits at +z = `distance` from the orbit target and the
    // projection (w = distance - z, then screen-space pan and zoom) is unchanged from the original yaw/pitch renderer.
    // The camera orientation is the quaternion viewQ, which rotates the camera's reference frame (camFrame) into view
    // space:
    //   'head'  - patient model coordinates (+X patient left, +Y superior, +Z anterior). The camera moves with the head,
    //             so a head-pose change leaves the anatomy still on screen (only the room cues move). All anatomical
    //             presets are head-framed: Anterior/Right/Left/Top and step (recommended) views.
    //   'world' - the operating room (see the head-pose block below). The camera stays in the room while the head is
    //             posed; used by Surgeon's View, the 'position' step view and whenever the user changes the pose.
    // `target` (orbit centre) is expressed in camFrame as well. viewBasis() resolves everything to one head->view
    // rotation M plus the orbit centre in model coordinates; the shader (uView, uTarget), transform(), project(),
    // picking, label anchors, depth sorting and cameraPosition() all use that basis.
    // Presets may still be authored as yaw/pitch(/roll): M = Rz(roll) * Rx(pitch) * Ry(yaw). The old
    // screenOrientation = -1 (image turned 180 degrees so the skull base is at the top) is roll = PI.
    function qMul(a, b) { return [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0], a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]]; }
    function qNorm(q) { const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1; return q.map(x => x / n); }
    function qConj(q) { return [-q[0], -q[1], -q[2], q[3]]; }
    function qAxis(axis, angle) { const n = Math.hypot(axis[0], axis[1], axis[2]) || 1, s = Math.sin(angle / 2) / n; return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)]; }
    function qFromAngles(yaw, pitch, roll = 0) { return qMul(qAxis([0, 0, 1], roll), qMul(qAxis([1, 0, 0], pitch), qAxis([0, 1, 0], yaw))); }
    function qMat(q) { const [x, y, z, w] = q; return [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]; }
    function matQ(m) { const [a, b, c, d, e, f, g, h, i] = m, t = a + e + i; let s; if (t > 0) {
        s = Math.sqrt(t + 1) * 2;
        return qNorm([(h - f) / s, (c - g) / s, (d - b) / s, s / 4]);
    } if (a > e && a > i) {
        s = Math.sqrt(1 + a - e - i) * 2;
        return qNorm([s / 4, (b + d) / s, (c + g) / s, (h - f) / s]);
    } if (e > i) {
        s = Math.sqrt(1 + e - a - i) * 2;
        return qNorm([(b + d) / s, s / 4, (f + h) / s, (c - g) / s]);
    } s = Math.sqrt(1 + i - a - e) * 2; return qNorm([(c + g) / s, (f + h) / s, s / 4, (d - b) / s]); }
    function mMul(a, b) { const r = []; for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
            r.push(a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j]); return r; }
    function mT(a) { return [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]; }
    function mApply(m, v) { return [m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]]; }
    const vSub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], vAdd = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]], vScale = (a, k) => [a[0] * k, a[1] * k, a[2] * k], vDot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2], vCross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], vNorm = a => vScale(a, 1 / (Math.hypot(a[0], a[1], a[2]) || 1));
    // Look-at orientation: `eye` is the direction from the target to the eye, `up` the frame direction drawn upward.
    function qLook(eye, up) { const z = vNorm(eye), x = vNorm(vCross(up, z)), y = vCross(z, x); return matQ([...x, ...y, ...z]); }
    // Best-effort yaw/pitch/roll of a head->view rotation (inverse of qFromAngles with |pitch| <= PI/2, so a view past
    // the vertex reports yaw + PI and roll + PI).
    function anglesOf(M) { return { yaw: Math.atan2(-M[6], M[8]), pitch: Math.asin(Math.max(-1, Math.min(1, M[7]))), roll: Math.atan2(-M[1], M[4]) }; }
    // ---- Surgical head positioning --------------------------------------------------------------------------------
    // World (operating-room) frame: patient supine, +Z_w toward the ceiling (gravity is -Z_w), +Y_w toward the head of
    // the table (where the surgeon sits), +X_w toward the patient's left. In the anatomical (neutral supine) pose the
    // model and world frames coincide. A head pose is a rigid rotation of the whole model (every mesh, marker,
    // instrument, label anchor and pick target) about POSE_PIVOT that maps model directions to world directions:
    //   P = Rx(-extension) * Rz(-s * lateral) * Ry(s * rotation),  s = sideSign(): +1 right-sided, -1 left (mirror)
    //   rotation  - contralateral turn about the patient's long axis (the face turns away from the operative side)
    //   extension - the vertex drops toward the floor (rotation about the table's left-right axis)
    //   lateral   - lateral flexion; positive tilts the vertex toward the contralateral shoulder
    //   world point = POSE_PIVOT + P * (model point - POSE_PIVOT)
    // POSE_PIVOT estimates the craniocervical junction from the atlas: the mean of the Occipital bone vertices within
    // 0.35 units (7 mm) of its lowest point (the occipital condyles beside the foramen magnum) is (0, -4.01, -1.30).
    // Neck and body are not modeled; the pivot only decides how the head swings relative to the table and the room.
    const POSE_PIVOT = [0, -4.0, -1.3];
    // Default operative pose in degrees: Rodriguez Rubio et al. describe about 15-20 deg contralateral rotation and
    // 20 deg extension in their example, adjusted to the target (see the 'position' step note).
    const POSE_DEFAULT = { rotation: 20, extension: 20, lateral: 0 }, POSE_KEYS = ['rotation', 'extension', 'lateral'];
    // poseSettings: the user-adjustable operative pose. poseOverride: null (follow the step), 'anatomical' or
    // 'operative' (Position toggle; cleared on every step change). poseShown/poseR: the pose drawn this frame.
    let poseSettings = { ...POSE_DEFAULT }, poseOverride = null, poseShown = { rotation: 0, extension: 0, lateral: 0 }, poseR = [1, 0, 0, 0, 1, 0, 0, 0, 1], poseBlend = null, poseClock = 0, propsOn = true, viewCache = null;
    // 0 = anatomical (neutral supine) ... 1 = operative pose, as the current step requests it. A step requests the
    // operative pose through its state ('position' is a state change like 'bone' or 'dura'): steps before 'position'
    // are anatomical, 'position' animates with the stage progress (opening slider, Before/After), later steps keep it.
    function poseFraction() { if (poseOverride)
        return poseOverride === 'operative' ? 1 : 0; const s = steps[stage]; if (!s)
        return 0; if (beforeCurrent)
        return s.beforeState.position ? 1 : 0; if (s.change === 'position')
        return smoothstep01(Math.max(0, Math.min(1, opening))); return s.state.position ? 1 : 0; }
    function requestedPose() { const f = poseFraction(); return Object.fromEntries(POSE_KEYS.map(k => [k, poseSettings[k] * f])); }
    function poseMatrix(p) { const s = sideSign(), r = Math.PI / 180; return qMat(qMul(qAxis([1, 0, 0], -p.extension * r), qMul(qAxis([0, 0, 1], -s * p.lateral * r), qAxis([0, 1, 0], s * p.rotation * r)))); }
    // Advance the displayed pose; called at the start of every draw. Step changes and the Anatomical/Operative toggle
    // glide over 0.65 s (poseBlend, timed by poseClock = the render frame time); slider input and the 'position' step
    // progress apply directly. Study-plate captures always use the requested pose.
    function updatePose() { const want = requestedPose(); let shown = want; if (poseBlend && POSE_KEYS.every(k => Math.abs(poseBlend.from[k] - want[k]) < 1e-3))
        poseBlend = null; if (poseBlend && !capturing) {
        const t = Math.max(0, Math.min(1, (poseClock - poseBlend.start) / poseBlend.duration)), e = smoothstep01(t);
        shown = Object.fromEntries(POSE_KEYS.map(k => [k, poseBlend.from[k] + (want[k] - poseBlend.from[k]) * e]));
        if (t >= 1)
            poseBlend = null;
        else
            render();
    } poseShown = shown; poseR = poseMatrix(shown); viewCache = null; }
    function startPoseBlend() { poseBlend = matchMedia('(prefers-reduced-motion: reduce)').matches ? null : { from: { ...poseShown }, start: performance.now(), duration: 650 }; }
    function posePoint(p, P = poseR) { return vAdd(POSE_PIVOT, mApply(P, vSub(p, POSE_PIVOT))); } // model -> world
    function poseInverse(p, P = poseR) { return vAdd(POSE_PIVOT, mApply(mT(P), vSub(p, POSE_PIVOT))); } // world -> model
    // Unit gravity direction in raw model coordinates (x not mirrored; the canonical, operative-side-right direction
    // is [g[0] * sideSign(), g[1], g[2]]). Neutral supine: [0, 0, -1], toward the occiput. Also uploaded each frame as
    // the vertex-shader uniform uGravity (declared for later work such as brain relaxation; unused until then).
    function gravityInModel() { return [-poseR[6], -poseR[7], -poseR[8]]; }
    function viewChanged() { viewCache = null; }
    // Head(model) -> view rotation M and the orbit centre in model coordinates for the current camera and pose.
    function viewBasis() { if (!viewCache) {
        const V = qMat(viewQ);
        viewCache = camFrame === 'world' ? { M: mMul(V, poseR), target: poseInverse(target) } : { M: V, target: [...target] };
    } return viewCache; }
    // Re-express the camera in the other frame without changing the image at the displayed pose.
    let camGlide = null; // see glideCamera
    // A running camera glide is re-expressed in the new frame too (its endpoints), so e.g. a pose control used during
    // the glide out of the 'position' step still lands on the step view instead of mixing frames.
    function attachCamera(frame) { if (frame === camFrame)
        return; const Pq = matQ(poseR), toWorld = frame === 'world', q = x => qNorm(toWorld ? qMul(x, qConj(Pq)) : qMul(x, Pq)), t = x => toWorld ? posePoint(x) : poseInverse(x); viewQ = q(viewQ); target = t(target); if (camGlide)
        for (const e of [camGlide.from, camGlide.to]) {
            e.q = q(e.q);
            e.target = t(e.target);
        } camFrame = frame; viewChanged(); }
    // Apply a camera preset: v.frame 'head' (default) or 'world'; orientation from yaw/pitch(/roll) or eye/up (qLook);
    // optional target (in that frame), distance and zoom (a longer lens for room views). Pan is left to the caller.
    function applyCamera(v) { camFrame = v.frame || 'head'; viewQ = v.eye ? qLook(v.eye, v.up || [0, 1, 0]) : qFromAngles(v.yaw, v.pitch, v.roll || 0); if (v.target)
        target = [...v.target]; if (v.distance != null)
        distance = v.distance; if (v.zoom != null)
        viewZoom = v.zoom; viewChanged(); }
    // QA / scripting entry point: v = { frame?, q? | yaw/pitch/roll? | eye/up?, target?, distance?, zoom?, pan? }. A frame
    // change without a target keeps the current orbit centre (re-expressed in the new frame); omitted orientation keeps
    // the current one.
    function setCamera(v = {}) { stopGlide(); if (v.frame && v.frame !== camFrame)
        attachCamera(v.frame); if (v.q)
        viewQ = qNorm(v.q);
    else if (v.eye || v.yaw != null)
        viewQ = v.eye ? qLook(v.eye, v.up || [0, 1, 0]) : qFromAngles(v.yaw, v.pitch || 0, v.roll || 0); if (v.target)
        target = [...v.target]; if (v.distance != null)
        distance = v.distance; if (v.zoom != null)
        viewZoom = v.zoom; if (v.pan)
        pan = [...v.pan]; viewChanged(); render(); }
    // Orbit the view around the selected target with a fixed frame-up direction. Horizontal motion changes azimuth;
    // vertical motion changes elevation. Rebuilding the look-at quaternion prevents accumulated camera roll.
    function rotateByDrag(dx, dy, k = .008) { const len = Math.hypot(dx, dy); if (!len)
        return; viewQ = qNorm(qMul(qAxis([dy, dx, 0], len * k), viewQ)); viewChanged(); }
    function orbitByDrag(dx, dy, k = .008) { if (!dx && !dy)
        return; const eye = mApply(qMat(viewQ), [0, 0, 1]), worldFrame = camFrame === 'world', up = worldFrame ? [0, 0, 1] : [0, 1, 0], az = worldFrame ? Math.atan2(eye[0], eye[1]) : Math.atan2(eye[0], eye[2]), el = Math.asin(Math.max(-1, Math.min(1, worldFrame ? eye[2] : eye[1]))), nextAz = az + dx * k, nextEl = Math.max(-1.52, Math.min(1.52, el - dy * k)), c = Math.cos(nextEl), nextEye = worldFrame ? [c * Math.sin(nextAz), c * Math.cos(nextAz), Math.sin(nextEl)] : [c * Math.sin(nextAz), Math.sin(nextEl), c * Math.cos(nextAz)]; viewQ = qLook(nextEye, up); viewChanged(); }
    function markInteraction() { interactionActive = true; pickDirty = false; clearTimeout(interactionTimer); interactionTimer = setTimeout(() => { interactionActive = false; render(true); }, 140); }
    // Camera glide: slerp the orientation and ease target, distance, pan and zoom to a preset (same frame) over
    // `duration` ms, timed by poseClock like the pose blend. Used when the 'position' step frames the room; any user
    // camera input or preset cancels it (stopGlide). Study-plate captures ignore it.
    function qSlerp(a, b, t) { let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; if (d < 0) {
        b = b.map(x => -x);
        d = -d;
    } if (d > .9995)
        return qNorm(a.map((x, i) => x + (b[i] - x) * t)); const th = Math.acos(d), sa = Math.sin((1 - t) * th) / Math.sin(th), sb = Math.sin(t * th) / Math.sin(th); return a.map((x, i) => x * sa + b[i] * sb); }
    function glideCamera(v, duration = 1000) { attachCamera(v.frame || 'head'); const from = { q: [...viewQ], target: [...target], distance, pan: [...pan], zoom: viewZoom }; applyCamera(v); const to = { q: [...viewQ], target: [...target], distance, pan: [0, 0], zoom: v.zoom ?? 1 }; if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
        pan = [0, 0];
        viewZoom = to.zoom;
        camGlide = null;
    }
    else {
        viewQ = from.q;
        target = from.target;
        distance = from.distance;
        viewZoom = from.zoom;
        camGlide = { from, to, start: performance.now(), duration };
    } viewChanged(); render(); }
    // stopGlide(false) only halts the motion; any other call is user camera input or a preset, which also ends the
    // automatic room framing of the 'position' step (roomCameraAuto, see activate).
    // Leaving the 'position' step with a room-framed camera always glides back to the next step's view unless the user
    // chose Surgeon's View (surgeonChosen); clicks with pointer jitter (under 5 px) and wheel zoom do not count as
    // camera input for roomCameraAuto.
    let roomCameraAuto = false, surgeonChosen = false;
    function stopGlide(user = true) { camGlide = null; if (user)
        roomCameraAuto = false; }
    function updateGlide() { if (!camGlide || capturing)
        return; const { from, to } = camGlide, t = Math.max(0, Math.min(1, (poseClock - camGlide.start) / camGlide.duration)), e = smoothstep01(t), mix = (a, b) => a + (b - a) * e; viewQ = qSlerp(from.q, to.q, e); target = from.target.map((x, i) => mix(x, to.target[i])); distance = mix(from.distance, to.distance); pan = from.pan.map((x, i) => mix(x, to.pan[i])); viewZoom = mix(from.zoom, to.zoom); viewChanged(); if (t >= 1)
        camGlide = null;
    else
        render(); }
    // Roll about the line of sight; a positive angle turns the image counter-clockwise.
    function rollView(angle) { viewQ = qNorm(qMul(qAxis([0, 0, 1], angle), viewQ)); viewChanged(); }
    function meshSide(m) { if (m.procedureSide)
        return m.procedureSide; if (m.nativeName.endsWith('.l'))
        return 'left'; if (m.nativeName.endsWith('.r'))
        return 'right'; return 'both'; }
    function onOperativeSide(m) { const side = meshSide(m); return side === 'both' || side === operativeSide(); }
    function baseId(m) { return m.baseId || m.id; }
    // contextOnly meshes count as relevant only for groups a step lists in contextGroups (the 'position' step shows the
    // whole skull, face included, so the head's orientation in the room is legible).
    function relevantMesh(m) { const s = steps[stage]; if (m.contextOnly)
        return !!s?.contextGroups?.includes(m.group) && !m.nativeName.startsWith('Hyoid'); if (m.group === 'bone' && s?.isolateBone && !s.isolateBone.includes(m.nativeName))
        return false; // The isolated skull-base view omits cortex (and the arachnoid on it), as the About panel states.
    if (s?.isolateBone && ['brain', 'arachnoid'].includes(m.group))
        return false; return true; }
    // A step's recommended camera, mirrored for a left-sided procedure (yaw and roll change sign; eye/up/target x flip).
    function stepView(s) { const sign = sideSign(), v = s.view, mx = p => p && [p[0] * sign, p[1], p[2]]; return { ...v, frame: v.frame || 'head', yaw: v.yaw * sign, roll: (v.roll || 0) * sign, eye: mx(v.eye), up: mx(v.up), target: mx(v.target) }; }
    function setProcedureSide(side) { if (!['right', 'left', 'both'].includes(side))
        return; contextSide = side; clearSelection(); syncControls(); updateChangeControls(); if (mode === '2d')
        showDiagram();
    else
        render(); }
    function setPalette(next) { if (!palettes[next])
        return; palette = next; try {
        localStorage.setItem('pterional-palette', next);
    }
    catch (error) { } syncControls(); if (mode === '2d')
        showDiagram();
    else
        render(); }
    function toggleFullContext() { const expanded = Object.keys(enabled).filter(k => enabled[k]).every(k => allEnabled[k]); if (!expanded) {
        contextLayerSnapshot = { ...allEnabled };
        // A step's contextLayers (the reflected scalp and soft tissue around the bony and intradural steps) join the
        // expanded view even though the step's relevant view leaves them out.
        const extra = steps[stage]?.contextLayers || [];
        for (const k in allEnabled)
            allEnabled[k] = enabled[k] || extra.includes(k);
        regionalContext = false;
    }
    else {
        Object.assign(allEnabled, contextLayerSnapshot || Object.fromEntries(Object.keys(allEnabled).map(k => [k, false])));
        regionalContext = true;
    } syncControls(); clearSelection(); updateChangeControls(); if (mode === '2d')
        showDiagram();
    else
        render(); }
    function chooseIndication(key, value) { choose({ ...config, [key]: value }); const id = key === 'clinoid' ? (value === 'none' ? 'retain' : 'clinoid') : key === 'muscle' ? 'muscle' : value === 'yes' ? 'fissure' : 'cistern'; activate(Math.max(0, steps.findIndex(s => s.id === id))); if (mode !== '3d')
        setMode('3d'); }
    function titleCase(value) { return String(value).replace(/\b([a-z][a-z0-9]*)\b/gi, (word, _, offset) => { const low = word.toLowerCase(); if (/^(ii|iii|iv|vi|vii|viii|ix|xi|xii|ica|mca|aca|sta|csf|m1|m2)$/.test(low))
        return low.toUpperCase(); if (offset > 0 && ['a', 'an', 'and', 'as', 'at', 'by', 'for', 'in', 'of', 'on', 'or', 'the', 'to', 'with'].includes(low))
        return low; return low[0].toUpperCase() + low.slice(1); }); }
    function updateMobilePanelHeight() { if (isMobile)
        $('#mobileModelDock').style.setProperty('--mobile-panel-height', Math.max(0, host.clientHeight - 62) + 'px'); }
    function zoomBy(factor, refreshPicking = true) { if (!Number.isFinite(factor) || factor <= 0)
        return; viewZoom = Math.max(.25, Math.min(16, viewZoom * factor)); render(refreshPicking); }
    function alignActiveStep() { const card = $$('.step')[stage], panel = $('#steps-panel'); if (!card || panel.hidden)
        return; if (isMobile) {
        panel.scrollTop += card.getBoundingClientRect().top - panel.getBoundingClientRect().top;
    }
    else
        card.scrollIntoView({ block: 'nearest', inline: 'nearest' }); }
    function updateMobileReaderHeight() { if (!isMobile)
        return; const card = $$('.step')[stage], panel = $('#steps-panel'), layout = $('.layout'); if (!card || panel.hidden || !panel.clientWidth)
        return; const probe = document.createElement('article'); probe.className = 'step active mobile-step-measure'; probe.setAttribute('aria-hidden', 'true'); probe.style.width = panel.clientWidth + 'px'; probe.innerHTML = `<div class="step-heading"><span class="step-number">${String(stage + 1).padStart(2, '0')}</span><div class="step-heading-text"><h2>${escape(titleCase(steps[stage].title))}</h2></div></div><p>${escape(steps[stage].body.split(/(?<=[.!?])\s+/).slice(0, 2).join(' '))}</p>`; document.body.append(probe); const available = layout.getBoundingClientRect().height, desired = Math.ceil(probe.getBoundingClientRect().height + $('#guideTabs').getBoundingClientRect().height + 3), height = Math.min(Math.max(132, available * .42), Math.max(150, desired)); probe.remove(); layout.style.setProperty('--mobile-reader-height', Math.round(height) + 'px'); updateMobilePanelHeight(); alignActiveStep(); }
    function syncMobileLayout() { const next = mobileQuery.matches; if (next !== isMobile) {
        if (isMobile)
            mobileLabelPreference = prelabelsOn;
        else
            desktopLabelPreference = prelabelsOn;
        isMobile = next;
        prelabelsOn = isMobile ? mobileLabelPreference : desktopLabelPreference;
        clearSelection();
    } const layers = $('#layerControls'), reset = $('#resetView'), pager = $('#stepPager'); if (isMobile) {
        $('#mobileModelDock').append(layers, reset);
        $('#mobileStepNav').append(pager);
    }
    else {
        $('#viewport').append(layers);
        $('#viewTools').append(reset);
        $('#stepStatus').append(pager);
    } syncControls(); requestAnimationFrame(() => { updateMobileReaderHeight(); alignActiveStep(); render(); }); }
    function modelIndex() { return configs.findIndex(c => JSON.stringify(c) === JSON.stringify(config)); }
    function downloadBlob(blob, name) { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 10000); }
    function decode64(s) { const str = atob(s), bytes = new Uint8Array(str.length); for (let i = 0; i < str.length; i++)
        bytes[i] = str.charCodeAt(i); return bytes; }
    function setTab(t) { for (const k of ['steps', 'indications', 'papers']) {
        $('#tab-' + k).setAttribute('aria-selected', k === t);
        $('#' + k + '-panel').hidden = k !== t;
    } if (t === 'steps') {
        updateMobileReaderHeight();
        alignActiveStep();
    } }
    $$('[role=tab]').forEach(b => b.onclick = () => setTab(b.id.slice(4)));
    function setMode(next) { animationToken++; mode = next; $('#mode3d').setAttribute('aria-pressed', mode === '3d'); $('#mode2d').setAttribute('aria-pressed', mode === '2d'); for (const s of ['#canvas', '#viewinfo', '.floating', '#gesturehint', '.viewtools', '#mobileModelDock', '#stepLabels', '#labelLines', '#orientGizmo'])
        $(s).hidden = mode !== '3d'; $('#figurepanel').hidden = mode === '3d'; $('#popup').hidden = true; if (mode === '2d')
        showDiagram();
    else {
        playStepAnimation();
        render();
    } updateChangeControls(); }
    $('#mode3d').onclick = () => setMode('3d');
    $('#mode2d').onclick = () => setMode('2d');
    $('#creditsOpen').onclick = () => $('#credits').showModal();
    $('#creditsClose').onclick = () => $('#credits').close();
    const publicSummaries = { anatomy: 'Layer-by-layer cadaveric anatomy of the pterional approach, including temporal tissue planes, bone exposure and dural opening.', extra: 'Extradural anterior clinoid anatomy and technique, emphasizing the optic canal, carotid artery and adjacent dural relationships.', intra: 'Comparison of intradural, extradural and hybrid anterior clinoidectomy techniques, with discussion of exposure and complication avoidance.', subfascial: 'Subfascial temporalis dissection and reconstruction, with attention to preservation of the frontotemporal facial nerve branch.', muscles: 'Clinical comparison of muscle-splitting, interfascial and subfascial temporalis dissection techniques and their functional and cosmetic outcomes.', sylvian: 'Microsurgical anatomy of the Sylvian fissure, its opercular and insular compartments, and associated cortical and vascular relationships.', 'no-split': 'A limited low subfrontal dural opening directed toward the optico-carotid region, with reduced exposure of frontal, temporal and Sylvian surfaces.' };
    const hasArachnoid = () => meshes.some(m => m.group === 'arachnoid' && m.fissureLine?.length > 1);
    function hasStageChange() { const s = steps[stage]; return !!s && (!!s.change && (s.change !== 'fissure' || hasArachnoid()) || meshes.some(m => m.marker && m.markerStages.includes(s.id))); }
    function updateChangeControls() { const s = steps[stage]; if (!s)
        return; $('#regionalContext').hidden = false; $('#regionalContext').textContent = 'Full Head Context'; const keys = Object.keys(enabled).filter(k => enabled[k]), expanded = keys.length > 0 && keys.every(k => allEnabled[k]), partial = Object.values(allEnabled).some(Boolean); regionalContext = !expanded; $('#regionalContext').setAttribute('aria-pressed', expanded ? 'true' : partial ? 'mixed' : 'false'); $('#viewlabel').textContent = 'Isolated Skull Base'; $('#viewlabel').hidden = !(s.isolateBone && !allEnabled.bone); $('#sideCaption').textContent = contextSide === 'both' ? 'Both Sides · Right Procedure' : (operativeSide() === 'left' ? 'Left' : 'Right') + '-Sided Procedure'; $('#focusChange').hidden = mode !== '3d'; $('#compareChange').hidden = mode !== '3d' || !hasStageChange(); const tools = meshes.filter(m => m.instrument && visible(m)); $('#visualnote').textContent = (tools.length ? 'Cyan: instrument · ' : '') + 'Colored markings are schematic · ' + (operativeSide() === 'left' ? 'Left' : 'Right') + ' procedure'; $('#surgeonView').hidden = mode !== '3d'; }
    function changeKind(m) { if (!procedure || m.instrument || m.marker || !onOperativeSide(m))
        return 0; const s = steps[stage], id = s.change; if (id === 'clinoid' && m.nativeName === 'Sphenoid bone')
        return 2; if (id === 'burr' && m.group === 'bone')
        return 7; if (id === 'bone' && m.group === 'bone')
        return 1; if (id === 'dura' && m.group === 'dura')
        return config.fissure === 'no' ? 4 : 3; if (id === 'muscle' && ['muscle', 'fascia', 'deepfascia', 'fat'].includes(m.group) && !m.contextOnly && (config.muscle === 'split' ? isTemporalis(m) || FASCIA_IDS.has(baseId(m)) : FASCIA_IDS.has(baseId(m))))
        return 5; if (id === 'pericranium' && (m.flap?.key === 'pericranium' || isTemporalis(m) && config.muscle === 'subfascial'))
        return 5; if (id === 'fissure' && ['brain', 'arteries', 'veins', 'arachnoid'].includes(m.group) && !m.contextOnly)
        return 6; return 0; }
    const indicationGroups = [
        { title: 'Clinoidectomy', key: 'clinoid', items: [
                { value: 'none', title: 'No clinoidectomy', body: 'Appropriate when the selected target is accessible without removing the anterior clinoid or unroofing the optic canal. Examples include many MCA and anterior communicating artery exposures when the clinoid does not obstruct the required view.', sources: ['anatomy', 'general'] },
                { value: 'extra', title: 'Extradural', body: 'Considered when a paraclinoid vascular or anterior skull-base target requires optic-canal decompression, clinoid removal or a wider optic–carotid window. Bone is approached before durotomy, with the dura separating the extradural work from the intradural contents.', sources: ['extra'] },
                { value: 'intra', title: 'Intradural', body: 'Considered when direct observation of the lesion, optic nerve and carotid artery helps determine how much clinoid or optic-canal bone to remove. It is a tailored option for selected paraclinoid vascular and skull-base lesions; the conventional sequence represented here opens the dura and Sylvian corridor first.', sources: ['intra'] }
            ] },
        { title: 'Muscle elevation', key: 'muscle', items: [
                { value: 'subfascial', title: 'Subfascial', body: 'Chosen to develop a separate fascial flap while protecting temporal facial nerve branches in the superficial tissues. Both fascial laminae and the intervening fat remain together; the choice concerns the exposure and soft-tissue reconstruction rather than a unique intracranial diagnosis.', sources: ['anatomy', 'subfascial', 'muscles'] },
                { value: 'split', title: 'Muscle-splitting / myocutaneous', body: 'An alternative soft-tissue exposure when scalp and temporalis are mobilized together. The selected comparative technique describes en-bloc reflection; it does not establish an exact intramuscular split or a separate pathology-specific indication.', sources: ['muscles'] }
            ] },
        { title: 'Sylvian fissure', key: 'fissure', items: [
                { value: 'yes', title: 'Split the fissure', body: 'Used when the corridor needs to expose MCA branches, the insula or deeper basal cisternal relationships. The extent is tailored to the vascular or tumor target and the surrounding veins and arteries.', sources: ['sylvian', 'anatomy'] },
                { value: 'no', title: 'No routine wide split', body: 'An option for selected anterior skull-base targets reachable through a limited low subfrontal opening over the optico-carotid region. Adequate access must be available without broad Sylvian exposure; the choice does not imply that arachnoid or cisternal dissection is never needed.', sources: ['no-split'] }
            ] }
    ];
    function renderIndications() { const p = $('#indications'); p.replaceChildren(); const intro = document.createElement('p'); intro.className = 'indication-intro'; intro.textContent = 'Select an option below to update the procedure. The target and surrounding anatomy determine the appropriate exposure; muscle-plane choices concern soft-tissue handling rather than a separate diagnosis.'; p.append(intro); for (const group of indicationGroups) {
        const h = document.createElement('h2');
        h.className = 'indication-group';
        h.textContent = group.title;
        p.append(h);
        for (const item of group.items) {
            const card = document.createElement('article');
            card.className = 'indication-card' + (config[group.key] === item.value ? ' active' : '');
            card.tabIndex = 0;
            card.setAttribute('role', 'button');
            card.setAttribute('aria-pressed', config[group.key] === item.value);
            card.setAttribute('aria-label', 'Select ' + item.title);
            card.dataset.optionKey = group.key;
            card.dataset.optionValue = item.value;
            const title = document.createElement('h3');
            title.textContent = item.title;
            const body = document.createElement('p');
            body.textContent = item.body;
            card.append(title, body);
            card.onclick = () => chooseIndication(group.key, item.value);
            card.onkeydown = e => { if (['Enter', ' '].includes(e.key)) {
                e.preventDefault();
                chooseIndication(group.key, item.value);
            } };
            p.append(card);
        }
    } }
    function relevant(p) { if (['extra', 'extra-review', 'corridors'].includes(p.id))
        return config.clinoid === 'extra'; if (p.id === 'intra')
        return config.clinoid === 'intra'; if (p.id === 'subfascial')
        return config.muscle === 'subfascial'; if (p.id === 'muscles')
        return true; if (['sylvian', 'focused-sylvian'].includes(p.id))
        return config.fissure === 'yes'; if (p.id === 'no-split')
        return config.fissure === 'no'; return p.tags.includes('all'); }
    function renderPapers() { const q = $('#paperSearch').value.toLowerCase(); $('#papers').replaceChildren(); let count = 0; for (const p of papers) {
        if (q && !(JSON.stringify(p) + ' ' + (publicSummaries[p.id] || '')).toLowerCase().includes(q))
            continue;
        count++;
        const card = document.createElement('article');
        card.className = 'paper-card';
        card.id = 'paper-' + p.id;
        card.innerHTML = `<div class="meta">${escape(p.citation)}</div><h2>${escape(p.title)}</h2><div class="meta">${escape(p.kind)} ${relevant(p) ? '<span class="match">Current selection</span>' : ''}</div><p>${escape(publicSummaries[p.id] || p.summary)}</p><div class="paper-actions"><a href="${escape(p.url)}" target="_blank" rel="noopener">Read source ↗</a><button data-paper-model="${p.id}">◇ Related anatomy</button>${p.doi ? `<a href="https://doi.org/${p.doi}" target="_blank" rel="noopener">DOI ↗</a>` : ''}</div>`;
        card.querySelector('button').onclick = () => { const choice = { ...config }; if (['extra', 'extra-review', 'corridors'].includes(p.id))
            choice.clinoid = 'extra'; if (p.id === 'intra')
            choice.clinoid = 'intra'; if (p.id === 'subfascial')
            choice.muscle = 'subfascial'; if (p.id === 'muscles')
            choice.muscle = 'split'; if (['sylvian', 'focused-sylvian'].includes(p.id))
            choice.fissure = 'yes'; if (p.id === 'no-split')
            choice.fissure = 'no'; choose(choice); let k = steps.findIndex(s => s.sources.includes(p.id)); if (k < 0)
            k = steps.findIndex(s => s.id === (p.id === 'focused-sylvian' ? 'fissure' : ['extra-review', 'corridors'].includes(p.id) ? 'clinoid' : 'orient')); activate(Math.max(0, k)); setTab('steps'); setMode('3d'); };
        $('#papers').append(card);
    } $('#paperEmpty').hidden = count > 0; $('#paperCount').textContent = papers.length; }
    $('#paperSearch').oninput = renderPapers;
    function openPaper(id) { setTab('papers'); $('#paperSearch').value = ''; renderPapers(); $('#paper-' + id)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }
    function captureStepImages() { if (!available || typeof canvas.toDataURL !== 'function')
        return manifest.study_plates || []; const saved = { viewQ: [...viewQ], camFrame, poseOverride, distance, viewZoom, target: [...target], pan: [...pan], selected, opening, procedure, contextSide, separateLayers, enabled: { ...enabled }, allEnabled: { ...allEnabled }, beforeCurrent, regionalContext, hidden: meshes.map(m => m.hidden) }; const s = steps[stage], vside = stepView(s), images = []; capturing = true; try {
        viewZoom = 1;
        regionalContext = saved.regionalContext;
        selected = -1;
        opening = 1;
        procedure = true;
        for (const k in enabled)
            enabled[k] = persistentLayers.has(k) || s.groups.includes(k);
        meshes.forEach(m => m.hidden = false);
        poseOverride = null; // plates show the pose the step itself requests
        // Plate cameras: the step's recommended camera (head-framed, except the world-framed 'position' view) and,
        // without a stage change, a second head-framed oblique view.
        const views = hasStageChange() ? [{ ...vside, title: 'Before this step', before: true }, { ...vside, title: 'After this step', before: false }] : [{ ...vside, title: 'Approach view' }, { title: s.id === 'muscle' ? 'Fascial planes' : 'Superior oblique view', frame: 'head', yaw: (s.id === 'muscle' ? .72 : .8) * sideSign(), pitch: s.id === 'muscle' ? .12 : 1.02, distance: s.id === 'muscle' ? 15 : s.view.distance + 1, target: vside.target }];
        for (const v of views) {
            beforeCurrent = v.before ?? saved.beforeCurrent;
            updatePose(); // so a world-framed plate target resolves against the plate's own pose
            applyCamera(v);
            viewZoom = v.zoom ?? 1;
            pan = [0, 0];
            separateLayers = s.id === 'muscle' && v.title === 'Fascial planes';
            draw(false);
            images.push({ image: canvas.toDataURL('image/png'), alt: v.title + ' for ' + s.title, caption: v.title + ' · ' + operativeSide() + (s.isolateBone ? (!allEnabled.bone ? ' · isolated skull-base context' : ' · expanded bone context') : '') });
        }
    }
    catch (error) {
        console.warn('2D snapshot unavailable', error);
        return manifest.study_plates || [];
    }
    finally {
        viewQ = saved.viewQ;
        camFrame = saved.camFrame;
        poseOverride = saved.poseOverride;
        distance = saved.distance;
        viewZoom = saved.viewZoom;
        target = saved.target;
        viewChanged();
        pan = saved.pan;
        selected = saved.selected;
        opening = saved.opening;
        procedure = saved.procedure;
        contextSide = saved.contextSide;
        separateLayers = saved.separateLayers;
        beforeCurrent = saved.beforeCurrent;
        regionalContext = saved.regionalContext;
        Object.assign(enabled, saved.enabled);
        Object.assign(allEnabled, saved.allEnabled);
        meshes.forEach((m, i) => m.hidden = saved.hidden[i]);
        capturing = false;
        draw(false);
    } return images; }
    function showDiagram() { const p = $('#figurepanel'), images = captureStepImages(); p.innerHTML = `<div class="diagram"><h2>${escape(titleCase(steps[stage].title))}</h2><div class="reference-plates">${images.map((im, i) => `<figure><img src="${im.image}" alt="${escape(im.alt)}" data-expand="${i}"><figcaption>${escape(im.caption)}</figcaption><button class="image-save" data-save="${i}">Save PNG</button></figure>`).join('')}</div><details class="plane-details" ${steps[stage].id === 'muscle' ? 'open' : ''}><summary>Temporal Tissue Planes</summary><svg viewBox="0 0 670 500" role="img" aria-label="Superficial to deep temporal tissue layers"><g font-family="system-ui" font-size="14" fill="#dce9dd">${[['Skin / subcutaneous tissue', '#8e9f79'], ['Suprafascial fat pad', '#dfc990'], ['Superficial fascial lamina', '#85b3a6'], ['Interfascial fat pad', '#c4ad70'], ['Deep fascial lamina', '#77a397'], ['Subfascial fat pad', '#cbb76b'], ['Temporalis muscle', '#b87367'], ['Periosteum / skull', '#d0c3a5']].map(([name, color], i) => `<rect x="24" y="${24 + i * 53}" width="260" height="34" rx="6" fill="${color}"/><text x="308" y="${46 + i * 53}">${escape(titleCase(name))}</text>`).join('')}${config.muscle === 'subfascial' ? '<path d="M 10 281 H 288" stroke="#dff09d" stroke-width="3" stroke-dasharray="6 4"/>' : ''}</g></svg></details></div>`; p.querySelectorAll('[data-expand]').forEach(img => img.onclick = () => img.parentElement.classList.toggle('expanded', !img.parentElement.classList.contains('expanded'))); p.querySelectorAll('[data-save]').forEach(b => b.onclick = () => { const im = images[+b.dataset.save]; downloadBlob(new Blob([decode64(im.image.split(',')[1])], { type: 'image/png' }), 'pterional-' + steps[stage].id + '-' + (+b.dataset.save + 1) + '.png'); }); }
    function buildSteps() {
        // position: head in the operative pose (see poseFraction). Scalp states: skin / galea incised (incision parting),
        // scalp flap reflected, pericranial flap raised, temporalis reflected (subfascial branch; in the myocutaneous
        // branch the temporalis travels with the scalp under `muscle`).
        let state = { bone: false, dura: false, clinoid: false, muscle: false, fissure: false, burr: false, position: false, skin: false, galea: false, scalp: false, pericranium: false, temporalis: false };
        const list = [], split = config.muscle === 'split';
        const scalpLayers = ['skin', 'galea', 'pericranium'], surface = ['bone', 'muscle', 'fascia', 'deepfascia', 'fat'], soft = [...scalpLayers, ...surface], cranial = ['bone', 'brain', 'dura', 'arteries', 'veins', 'nerves', 'arachnoid'];
        // From the burr holes to bone replacement the reflected scalp, pericranium, temporalis and fascial flap are not
        // part of the relevant (bone / dura / cranial) view, but Full Head Context shows them in their current state.
        const softStages = new Set(['burr-holes', 'bone-edges', 'dural-separation', 'bone', 'epidural-hemostasis', 'sphenoid-ridge', 'extension', 'clinoid', 'tack-up', 'dura', 'fissure', 'cistern', 'retain', 'intradural-work', 'dura-close', 'bone-replace']), extracranialContext = id => softStages.has(id) ? soft : null;
        const add = (id, title, body, options = {}) => { const beforeState = { ...state }; if (options.change)
            Object.assign(state, options.change); const sources = options.sources || ['anatomy']; list.push({ id, title, body, note: options.note || '', sources, groups: options.groups || cranial, labels: options.labels || ['Frontal bone'], state: { ...state }, beforeState, change: options.change ? Object.keys(options.change)[0] : null, changes: Object.keys(options.change || {}), stagger: options.stagger || null, visualAction: options.visualAction || id, markings: meshes.filter(m => m.marker && m.markerStages.includes(id) && m.procedureSide === 'right').map(m => m.id), isolateBone: options.isolateBone || null, contextGroups: options.contextGroups || null, contextLayers: options.contextLayers || (options.isolateBone ? null : extracranialContext(id)), view: { yaw: 1.0, pitch: .28, distance: 14, target: [-1, 0, .7], alpha: 1, ...options.view }, evidence: sources.map(paper => ({ paper, section: options.sections?.[paper] || papers.find(p => p.id === paper).section, origin: options.origin || 'Technique synthesis; preparation and closure also follow the procedural outline supplied for this guide.' })), figure: null, page: 1 }); };
        const clinoidView = { distance: 7, target: [-.8, -1.1, 1.35], pitch: 1.25, yaw: .6 };
        add('orient', 'Identify the pterion and target', 'Identify the meeting point of the frontal, parietal, temporal and sphenoid bones and relate it to the planned intracranial target. This establishes the frontotemporal route and the amount of frontal, temporal or basal exposure that will be needed.', { groups: ['bone', 'brain', 'arteries', 'veins', 'nerves', 'arachnoid'], view: { distance: 17 }, note: 'The illustrative scalp is hidden in this step so the sutures meeting at the pterion stay visible; it covers the region again from positioning onward (Layers & View › Scalp Skin shows it here). On the surface the pterion lies roughly two finger-breadths above the zygomatic arch and a thumb-breadth behind the frontal process of the zygomatic bone, the usual bedside estimate.' });
        add('position', 'Position and immobilize the head', 'Position supine with the head elevated for venous drainage and secured in a Mayfield or Sugita head frame. Rotate away from the operative side and tailor extension to the target so gravity can relax the frontal lobe without allowing the orbital roof to obscure the optic–carotid corridor.', { sources: ['anatomy', 'general'], groups: ['bone', 'brain', 'skin'], labels: ['Right scalp of the vault (skin and subcutaneous tissue) — illustration'], note: 'Rodriguez Rubio et al. describe about 15–20° contralateral rotation and 20° extension in their example, with target-dependent adjustment. The model turns the atlas head rigidly by those default angles about an estimated craniocervical pivot; adjust them under Layers & View › Patient Position. The cyan arrow on the scalp draws in as the head turns and marks the direction of the contralateral rotation (seen best from the top of the head or in Surgeon\'s View). Neck, body and venous-drainage elevation are not modeled, and the head holder, pin sites and table edge are schematic.', change: { position: true }, contextGroups: ['bone', 'skin'], view: { frame: 'world', eye: [-1, -.4, .55], up: [0, 0, 1], target: [-.3, -5, -3], distance: 48, zoom: 1.9 }, sections: { anatomy: 'PDF pp.6–7, Positioning', general: 'Positioning' } });
        add('prepare', 'Prepare and mark the incision', 'Clip hair as needed, prepare the skin and mark a curving frontotemporal incision behind the hairline. Infiltrate the marked line with local anesthetic and epinephrine as appropriate to reduce incisional pain and scalp bleeding.', { sources: ['general', 'radiopaedia'], groups: soft, contextGroups: ['skin'], labels: ['Right scalp around the incision (skin and subcutaneous tissue) — illustration'], note: 'Hair clipping, skin preparation and infiltration are described, not modeled; the scalp is an illustrative layered shell.', sections: { general: 'Approach context only; preparation and infiltration details come from the supplied procedural outline' }, origin: 'Preparation details follow the procedural outline supplied for this guide; they are not attributed to a measured atlas or to a specific passage in the linked review.' });
        add('sterile', 'Drape the operative field', 'Apply sterile drapes around the prepared operative area while keeping the marked incision and required landmarks accessible. This isolates the field before the skin is opened.', { sources: ['general', 'radiopaedia'], groups: soft, contextGroups: ['skin'], labels: ['Right scalp around the incision (skin and subcutaneous tissue) — illustration'], note: 'Drapes and the sterile field are not modeled.', sections: { general: 'Approach context only; draping detail comes from the supplied procedural outline' }, origin: 'Draping step follows the procedural outline supplied for this guide; the linked review provides general approach context.' });
        add('incision', 'Make the curving scalp incision', 'Begin at the superior border of the zygomatic arch about 1 cm anterior to the tragus and curve toward the midline, preferably behind the hairline. Keep the superficial temporal artery and auriculotemporal region in view while opening the scalp so the flap retains its blood supply.', { groups: soft, contextGroups: ['skin'], change: { skin: true, galea: true }, note: 'The scalp is an illustrative layered shell (skin with subcutaneous tissue, galea, pericranium) fitted over the atlas skull and temporal layers, not a segmentation. The cut opens along an illustrative course from the zygomatic arch toward the midline: its upper part follows the atlas schematic marking, its lower part is redrawn to start on the arch about 1 cm in front of the estimated tragus. The parting of the edges is exaggerated slightly for visibility.', sections: { anatomy: 'PDF p.7, Incision of skin; p.8, Figure 3' } });
        if (config.muscle === 'subfascial')
            add('scalp-flap', 'Mobilize the scalp flap', 'Elevate the scalp in the loose tissue plane, exposing pericranium above the superior temporal line and temporal fascia below it. Keep the superficial temporal artery with the flap and protect the temporal facial nerve branches to preserve flap perfusion and forehead movement.', { groups: soft, change: { scalp: true }, labels: ['Right scalp flap (skin and subcutaneous tissue) — illustration'], note: 'The illustrative scalp flap (skin with the galea) is peeled from the incision toward its base, rolled over the orbital margin about a straight hinge and laid toward the face (about 9 mm above the facial skeleton, standing in for the rolled sponge and the facial soft tissue the model lacks); it is not a soft-tissue simulation. The superficial temporal vessels lie in its subcutaneous layer (Full Head Context) and travel with it; facial nerve branches are not shown.', sections: { anatomy: 'PDF pp.8–10, Dissection of skin flap; Figures 3–4' } });
        else
            add('scalp-flap', 'Preserve the myocutaneous layers', 'Preserve continuity between scalp and temporal fascia while defining the edges of the planned myocutaneous flap. Protect the superficial temporal vessels and temporal facial nerve branches so the scalp and muscle can subsequently be mobilized together.', { sources: ['muscles'], groups: ['pericranium', ...surface], labels: ['Right temporalis muscle'], sections: { muscles: 'PDF p.2, Methods: scalp and temporalis reflected together without separating scalp from temporal fascia' }, note: 'No preliminary separate scalp–fascial dissection is included in this branch. The illustrative scalp and galea are hidden in this step so the planned flap margin on the temporal fascia and pericranium stays visible; they stay attached and are reflected together with the temporalis in the next step (Layers & View › Scalp Skin shows them here).' });
        if (config.muscle === 'subfascial')
            add('muscle', 'Develop the subfascial plane', 'Incise both temporal fascial laminae and elevate them together, keeping the interfascial fat within the superficial flap. Separate this fascial envelope from temporalis so the facial nerve branches remain protected in the superficial tissues.', { sources: ['anatomy', 'subfascial'], groups: soft, labels: ['Right superficial fascial lamina — illustration'], change: { muscle: true }, note: 'Fascial laminae and fat pads are original illustrations. The fascial flap is cut along the scalp incision and peeled forward about the scalp flap\'s hinge, folding outside the reflected scalp; the posterior temporal fascia stays.' });
        else
            add('muscle', 'Mobilize the myocutaneous flap', 'Mobilize the scalp and temporalis together as a myocutaneous flap while protecting the superficial temporal vessels and temporal facial nerve branches. Keeping these tissues together exposes the pterion without opening a separate scalp–fascial plane.', { sources: ['muscles'], groups: soft, labels: ['Right temporalis muscle'], change: { muscle: true, scalp: true }, note: 'Muscle-splitting terminology is retained from the comparative paper; an exact intramuscular fiber split is not specified or simulated. The illustrative scalp, temporal fascia and temporalis anterior to the incision fold forward together about one hinge; the muscle below the zygomatic arch stays.' });
        add('retract-flap', 'Reflect and secure the flap', 'Reflect the mobilized tissues anteriorly over the orbital margin and secure them with scalp hooks or an equivalent retraction system. Distribute tension to keep the bony field accessible while avoiding excessive traction on the superficial tissues.', { sources: ['anatomy', 'general'], groups: soft, labels: [split ? 'Right temporalis muscle' : 'Right scalp flap (skin and subcutaneous tissue) — illustration'], note: 'The flap is shown rolled over the orbital margin about a straight illustrative hinge; scalp hooks and sponges are not modeled.', sections: { anatomy: 'PDF pp.7–10, Anterior flap reflection', general: 'Skin incision and soft tissue dissection' } });
        add('periosteum', 'Raise pericranium and expose bone', 'Elevate the pericranium and detach the temporalis attachment from the bone as required, preserving tissue for reconstruction. Expose the superior temporal line, frontozygomatic region and temporal squama so burr holes and the bone flap can be planned from visible landmarks.', { sources: ['anatomy', 'general'], groups: ['bone', 'muscle', ...scalpLayers], labels: ['Right pericranial flap — illustration'], change: split ? { pericranium: true } : { pericranium: true, temporalis: true }, stagger: { pericranium: [0, .62], temporalis: [.38, 1] }, note: split ? 'The illustrative pericranium is raised as an anteriorly based flap and folded forward over the reflected scalp.' : 'The illustrative pericranium is raised as an anteriorly based flap and folded forward over the reflected scalp; the temporalis anterior to the incision is then peeled from the superior temporal line and reflected inferiorly over the zygomatic arch (its hinge is illustrative). Fish hooks and the musculofascial cuff are not modeled.', sections: { anatomy: 'PDF pp.11–13, Temporal fascia and temporalis mobilization', general: 'Bony landmarks' } });
        add('burr-holes', 'Place the frontal and temporal burr holes', 'Use the frontozygomatic sutures to locate the MacCarty keyhole region and place a second hole above the zygomatic arch in the temporal squama. Hole number and position are tailored to the required flap and dural adherence; the purpose is to obtain controlled access to the epidural plane before connecting the cuts.', { sources: ['anatomy', 'general'], groups: ['bone', 'dura'], view: { yaw: 1.25, pitch: .18, distance: 10.5, target: [-2.5, -.65, 1.5] }, change: { burr: true }, labels: ['Right temporal bone'], sections: { anatomy: 'PDF pp.13–18, Craniotomy; MacCarty keyhole and burr-hole placement on p.17', general: 'Bony landmarks' }, note: 'Burr holes and drill directions are schematic; millimetric keyhole positioning is not inferred from this atlas.' });
        add('bone-edges', 'Control bleeding at the burr holes', 'Control bleeding from the exposed bony edges, using bone wax where indicated, and remove loose bony spicules. This clears the entry points and reduces the chance that residual sharp edges will catch the dura.', { sources: ['general'], groups: ['bone', 'dura'], sections: { general: 'Bony landmarks for context; bone-wax and edge-cleaning details follow the supplied procedural outline' }, origin: 'Bone-edge hemostasis and cleaning follow the supplied procedural outline; the linked review provides craniotomy context.' });
        add('dural-separation', 'Separate dura from the inner table', 'Use a Penfield dissector to free the periosteal dura from the inner calvarial surface along the intended cuts. Creating this epidural plane protects the dura as the craniotome footplate passes beneath the bone.', { sources: ['general', 'anatomy'], groups: ['bone', 'dura'], labels: ['Right local dura — original illustration'], sections: { general: 'Bony landmarks', anatomy: 'PDF pp.17–18, Craniotomy and bone-flap elevation' } });
        add('bone', 'Raise the frontotemporal bone flap', 'Connect the burr holes with a craniotome and gently elevate the bone flap while completing dural separation. Tailor the frontal and temporal extent to the target so the opening provides the needed line of sight without unnecessary bone removal.', { sources: ['anatomy', 'general'], groups: cranial, labels: ['Right temporal bone'], change: { bone: true }, note: 'The flap follows the illustrative frontotemporal cut and lifts from its inferior edge before moving laterally.', sections: { anatomy: 'PDF pp.17–18, Craniotomy and bone-flap elevation', general: 'Bony landmarks' } });
        add('epidural-hemostasis', 'Inspect the dura and obtain hemostasis', 'Inspect the exposed dura and the margins of the craniotomy for tears and epidural bleeding. Control bleeding before proceeding so the dural surface and basal landmarks remain visible.', { sources: ['anatomy', 'general'], groups: ['bone', 'dura'], labels: ['Right local dura — original illustration'], sections: { anatomy: 'PDF pp.17–19, Craniotomy, basal drilling and dural suspension', general: 'Bony landmarks' } });
        add('sphenoid-ridge', 'Flatten the sphenoid wing', 'Reduce the lateral sphenoid wing to lower the basal edge of the frontotemporal corridor. This improves the view along the anterior skull base and reduces the amount of frontal or temporal displacement needed for access.', { sources: ['anatomy', 'general'], groups: ['bone', 'dura', 'nerves', 'arteries'], labels: ['Sphenoid bone'], sections: { anatomy: 'PDF pp.18–19, Drilling of basal structures; Figure 10', general: 'Bony landmarks' }, note: 'The atlas window summarizes basal bone work; incremental sphenoid drilling is not separately segmented.' });
        add('extension', 'Extend basal bone removal only if needed', 'If the target requires additional exposure, extend bone removal along the orbital roof, sphenoid ridge or temporal base. These extensions lower or widen the working corridor and are selected for the lesion rather than added routinely.', { sources: ['anatomy', 'general'], groups: ['bone', 'dura'], labels: ['Sphenoid bone'], sections: { anatomy: 'PDF pp.18–19 and 22–23, Basal drilling and modifications', general: 'Tailoring the pterional approach' }, note: 'Optional extension is described; its precise osteotomy is not simulated.' });
        if (config.clinoid === 'extra')
            add('clinoid', 'Remove the anterior clinoid extradurally', 'With the dura still closed, identify the meningo-orbital fold and develop the extradural exposure of the anterior clinoid before unroofing the optic canal and addressing the optic-strut attachment. Removing the required clinoid bone expands the optic–carotid corridor while preserving the dural boundary around the intradural contents.', { sources: ['extra'], groups: ['bone', 'nerves', 'arteries'], isolateBone: ['Sphenoid bone'], labels: ['Sphenoid bone'], change: { clinoid: true }, view: clinoidView, note: 'The clinoid mask and cyan instrument direction are approximate; meningo-orbital fold division, dural peeling, optic canal and optic strut removal are not separately reconstructed.' });
        add('tack-up', 'Place dural tack-up sutures', 'Place dural hitch sutures at suitable craniotomy margins when required and tent the dura toward the bone edge. Apposing the dura to the inner table reduces the epidural space in which postoperative blood can collect.', { sources: ['anatomy', 'general'], groups: ['bone', 'dura'], labels: ['Right local dura — original illustration'], sections: { anatomy: 'PDF p.19, Dural opening: dural suspension', general: 'Dura opening' }, note: 'Tack-up sutures are described, not modeled.' });
        add('dura', config.fissure === 'yes' ? 'Open and reflect the dura' : 'Create the low subfrontal dural opening', config.fissure === 'yes' ? 'Open the dura in a tailored curving flap and reflect it anteriorly toward the sphenoid base. Protect the underlying cortical vessels and Sylvian veins so the frontal and temporal surfaces can be exposed without injuring the structures beneath the incision.' : 'Make a limited dural opening low over the optico-carotid region, keeping the remaining frontal and temporal surfaces covered. This directs the corridor toward an anterior skull-base target while reducing broad Sylvian exposure.', { sources: config.fissure === 'yes' ? ['anatomy', 'general'] : ['no-split'], groups: cranial, labels: ['Right local dura — original illustration'], change: { dura: true }, view: { distance: 11 }, sections: config.fissure === 'yes' ? { anatomy: 'PDF pp.19–20, Dural opening; Figure 11', general: 'Dura opening' } : { 'no-split': 'PDF pp.2–3, Surgical Technique; Figure 1' }, note: 'A smooth illustrative window represents the dural opening; reflected sutured leaflets are not reconstructed.' });
        if (config.fissure === 'yes')
            add('fissure', 'Open the Sylvian arachnoid corridor', 'Identify the superficial Sylvian veins and open the arachnoid interval between the frontal and temporal opercula while preserving veins and arterial branches. Develop the interval toward the insular and basal cisterns, releasing cerebrospinal fluid as appropriate to expose the middle cerebral artery and gain working space with less fixed cortical retraction.', { sources: ['sylvian'], groups: ['brain', 'arteries', 'veins', 'nerves', 'arachnoid', 'bone'], labels: ['Right middle cerebral artery (m1-segment)'], change: { fissure: true }, view: { distance: 9, target: [-1.9, -.5, 1], pitch: .4 }, note: hasArachnoid() ? 'The translucent arachnoid and superficial Sylvian veins are original illustrations; the arachnoid opens along a schematic fissure line on native cortex. Opercular separation, small veins and the depth of the split are not reconstructed.' : 'The corridor is highlighted on native cortex; arachnoid, small veins and exact fissure opening are not reconstructed.' });
        else
            add('cistern', 'Use the limited subfrontal corridor', 'Work through the low subfrontal opening toward the optico-carotid cistern without a routine wide Sylvian split. This concentrates exposure around the skull-base target while preserving the covered frontal and temporal surfaces.', { sources: ['no-split'], groups: cranial, labels: ['Right optic nerve (ii)'], view: clinoidView });
        if (config.clinoid === 'intra')
            add('clinoid', 'Remove the anterior clinoid intradurally', 'After dural opening, identify the optic nerve, carotid artery and their relationship to the anterior clinoid before removing the required bone under direct intradural observation. Tailor optic-canal unroofing and clinoid removal to the exposure needed while protecting the adjacent neurovascular structures.', { sources: ['intra'], groups: ['bone', 'nerves', 'arteries'], isolateBone: ['Sphenoid bone'], labels: ['Sphenoid bone'], change: { clinoid: true }, view: clinoidView, note: 'The conventional description includes Sylvian splitting; the no-split combination retains its evidence note in Papers. Bone removal is approximate.' });
        if (config.clinoid === 'none')
            add('retain', 'Keep the anterior clinoid intact', 'Leave the anterior clinoid in place when the target can be reached through the available corridor without additional optic-canal or paraclinoid exposure. This preserves the bony boundary beside the optic nerve and carotid artery.', { groups: ['bone', 'nerves', 'arteries'], isolateBone: ['Sphenoid bone'], labels: ['Sphenoid bone'], view: clinoidView });
        add('intradural-work', 'Inspect the target corridor', 'Identify the optic nerve, carotid artery and relevant arterial branches at the end of the selected approach. The lesion-specific treatment proceeds from this exposure; its maneuvers depend on the pathology and are outside this approach sequence.', { sources: ['general'], groups: cranial, labels: ['Right internal carotid artery'], sections: { general: 'Intradural corridors' }, note: 'No lesion, aneurysm clipping or tumor resection is simulated.' });
        add('dura-close', 'Reapproximate and close the dura', 'After the intradural work and hemostasis, reapproximate the dural margins and obtain a watertight closure, adding a graft if necessary. Restoring this barrier helps contain cerebrospinal fluid and separates the intradural compartment from the wound.', { sources: ['general'], groups: cranial, labels: ['Right local dura — original illustration'], change: { dura: false, fissure: false }, visualAction: 'dura-close', sections: { general: 'Closure' }, note: 'Restoration of the illustrative dural surface represents closure; suture lines and grafts are not modeled.' });
        add('bone-replace', 'Replace and secure the bone flap', 'Return the bone flap to its original position and secure it with suitable plates and screws, adding a central dural hitch if required. Stable fixation restores the cranial contour and protects the underlying dural closure.', { sources: ['general'], groups: ['bone', 'dura'], labels: ['Right temporal bone'], change: { bone: false, burr: false }, sections: { general: 'Closure' }, note: 'The flap returns along the reverse path and fixation sites are marked schematically. Any removed clinoid remains removed.' });
        add('muscle-close', 'Reattach temporalis and fascia', 'Reapproximate temporalis to its preserved attachment or reconstructive cuff, then restore the fascial layers as appropriate. Re-establishing the muscle position and tissue planes supports mastication and reduces contour deformity.', { sources: ['general', 'anatomy'], groups: soft, labels: ['Right temporalis muscle'], change: split ? { muscle: false, scalp: false, pericranium: false } : { muscle: false, temporalis: false, pericranium: false }, stagger: split ? { pericranium: [0, .45], muscle: [.35, 1], scalp: [.35, 1] } : { pericranium: [0, .45], temporalis: [.3, .75], muscle: [.55, 1] }, sections: { general: 'Closure', anatomy: 'PDF pp.11–12, Preserved musculo-fascial cuff for temporalis reattachment' }, note: split ? 'The pericranial flap returns, then the myocutaneous flap (scalp, fascia and temporalis together) folds back into place; muscle sutures are marked schematically.' : 'The pericranial flap returns, then the temporalis and the fascial flap fold back into place; muscle sutures are marked schematically.' });
        add('drain', 'Consider a subgaleal drain', 'If persistent wound oozing suggests a risk of extracranial blood collection, consider a subgaleal drain before the soft tissues are closed. Drain use is conditional on hemostasis and the operative situation rather than an obligatory part of every exposure.', { sources: ['general', 'radiopaedia'], groups: soft, labels: ['Right galea aponeurotica around the incision — illustration'], sections: { general: 'Closure discusses drainage; conditional use follows the supplied procedural outline' }, origin: 'Conditional drain use follows the supplied procedural outline; the review describes drainage without establishing a universal indication.', note: 'The drain is described, not modeled.' });
        add('galea-close', 'Close the galea', 'Reapproximate the galea with a layered closure after confirming hemostasis and the position of any drain. This restores the supportive scalp layer and reduces tension on the skin edges.', { sources: ['general', 'radiopaedia'], groups: soft, contextGroups: ['skin'], labels: ['Galeal closure — schematic'], change: split ? { galea: false } : { scalp: false, galea: false }, stagger: split ? null : { scalp: [0, .75], galea: [.6, 1] }, sections: { general: 'Closure context; separate galeal-layer step follows the supplied procedural outline' }, origin: 'The separate galeal closure step follows the supplied procedural outline; it is not inferred from the article discussion of galeal grafts.', note: (split ? 'The galeal edges of the illustrative scalp close.' : 'The illustrative scalp flap folds back and its galeal edges close.') + ' Buried galeal stitches are marked schematically on the galea between the skin edges, which stay slightly apart until the skin is closed.' });
        add('skin-close', 'Close the skin', 'Align and close the skin edges with the selected sutures or staples. This restores the external barrier and completes the wound closure with attention to scalp perfusion and contour.', { sources: ['general', 'radiopaedia'], groups: soft, contextGroups: ['skin'], labels: ['Skin closure — schematic'], change: { skin: false }, sections: { general: 'Closure' }, note: 'The illustrative skin edges close; sutures or staples are marked schematically, not modeled.' });
        return list;
    }
    function rebuild({ reset = false } = {}) { animationToken++; const old = steps[stage]?.id; steps = buildSteps(); $('#steps').replaceChildren(); steps.forEach((s, i) => { const card = document.createElement('article'); card.className = 'step'; card.tabIndex = 0; card.innerHTML = `<div class="step-heading"><span class="step-number">${String(i + 1).padStart(2, '0')}</span><div class="step-heading-text"><h2>${escape(titleCase(s.title))}</h2></div></div><p>${escape(s.body)}</p>`; card.onclick = () => activate(i); card.onkeydown = e => { if (e.target === card && ['Enter', ' '].includes(e.key)) {
        e.preventDefault();
        activate(i);
    } }; $('#steps').append(card); }); const index = modelIndex(); $('#conflict').hidden = !(config.clinoid === 'intra' && config.fissure === 'no'); $('#conflict').textContent = 'Evidence conflict: this intradural clinoidectomy source describes wide Sylvian splitting. This selectable combination is shown for comparison, but is not validated by the cited sequence.'; renderPapers(); renderIndications(); activate(reset ? 0 : Math.max(0, steps.findIndex(s => s.id === old))); $('#planLive').textContent = steps.length + ' guide steps. Configuration updated.'; }
    function choose(c) { config = { ...c }; for (const [k, v] of Object.entries(config))
        $(`input[name="${k}"][value="${v}"]`).checked = true; rebuild(); }
    // Static snapshot of the guide steps for all 12 configurations, as stored in data/manifest.json `variants`
    // (regenerate with JSON.stringify(guideQA.variantSnapshot()) in the browser console; the build keeps the field).
    const VARIANT_KEYS = ['id', 'title', 'body', 'note', 'sources', 'groups', 'labels', 'state', 'beforeState', 'change', 'changes', 'stagger', 'visualAction', 'markings', 'isolateBone', 'contextGroups', 'contextLayers', 'view', 'evidence', 'figure', 'page'];
    function variantSnapshot() { const keep = { ...config }, at = stage, out = configs.map((c, i) => { choose(c); return { number: i + 1, config: { ...c }, steps: steps.map(s => Object.fromEntries(VARIANT_KEYS.map(k => [k, s[k] ?? null]))), evidence_conflict: c.clinoid === 'intra' && c.fissure === 'no' }; }); choose(keep); activate(at); return out; }
    $$('.config input').forEach(i => i.onchange = e => { config[e.target.name] = e.target.value; rebuild(); });
    function syncControls() { $$('[data-layer]').forEach(e => e.checked = enabled[e.dataset.layer]); $$('[data-swatch]').forEach(e => e.style.background = palettes[palette][e.dataset.swatch]); if ($('#palette'))
        $('#palette').value = palette; $$('[data-all-layer]').forEach(e => e.checked = allEnabled[e.dataset.allLayer]); $('#side').value = contextSide; $('#procedure').checked = procedure; $('#prelabels').checked = prelabelsOn; $('#instruments').checked = instrumentsOn; }
    const persistentLayers = new Set(['brain', 'arteries', 'veins', 'nerves', 'arachnoid']);
    function applyRecommendedView({ preserveCamera = false } = {}) { const s = steps[stage]; for (const k in enabled)
        enabled[k] = persistentLayers.has(k) || s.groups.includes(k); contextAlpha = 1; procedure = true; meshes.forEach(m => m.hidden = false); if (!preserveCamera) {
        viewZoom = 1;
        regionalContext = true;
        for (const k in allEnabled)
            allEnabled[k] = false;
        stopGlide();
        applyCamera(stepView(s)); // head-framed, or world-framed for the 'position' step
        pan = [0, 0];
    } $('#presetname').textContent = titleCase(s.title); syncControls(); clearSelection(); playStepAnimation(); updateChangeControls(); render(); }
    let activeStepId = null;
    function activate(i) { animationToken++; stopGlide(false); startPoseBlend(); poseOverride = null; beforeCurrent = false; $('#compareChange').setAttribute('aria-pressed', false); $('#compareChange').textContent = 'Before'; stage = Math.max(0, Math.min(steps.length - 1, i)); opening = 1; $$('.step').forEach((c, j) => { c.classList.toggle('active', j === stage); c.setAttribute('aria-current', j === stage ? 'step' : 'false'); }); $('#stagecaption').textContent = String(stage + 1).padStart(2, '0') + ' · ' + titleCase(steps[stage].title); $('#stagecount').textContent = String(stage + 1).padStart(2, '0') + ' / ' + steps.length; $('#previous').disabled = stage === 0; $('#next').disabled = stage === steps.length - 1; // Entering the 'position' step glides the camera to the step's room-fixed view while the head turns into the
    // operative pose (the camera is pinned to the room, so the head moves rather than the room around it). Leaving it
    // with that framing untouched glides on to the new step's recommended (head-framed) view, so the walkthrough
    // returns to the anatomy, which then looks as it did before positioning existed.
    const changed = activeStepId !== steps[stage].id; if (changed && steps[stage].change === 'position' && $('#positionControls'))
        $('#positionControls').open = true; if (changed && steps[stage].change === 'position') {
        glideCamera(stepView(steps[stage]), 1100);
        roomCameraAuto = true;
    }
    else if (changed && (roomCameraAuto || activeStepId === 'position' && camFrame === 'world' && !surgeonChosen)) {
        glideCamera(stepView(steps[stage]), 900);
        roomCameraAuto = false;
    } activeStepId = steps[stage].id; applyRecommendedView({ preserveCamera: true }); updateChangeControls(); updateMobileReaderHeight(); alignActiveStep(); if (mode === '2d')
        showDiagram(); }
    $('#previous').onclick = () => { setTab('steps'); activate(stage - 1); };
    $('#next').onclick = () => { setTab('steps'); activate(stage + 1); };
    $('#recommendedView').onclick = () => { surgeonChosen = false; applyRecommendedView(); };
    $$('[data-layer]').forEach(e => e.onchange = () => { enabled[e.dataset.layer] = e.checked; clearSelection(); updateChangeControls(); render(); });
    $$('[data-all-layer]').forEach(e => e.onchange = () => { allEnabled[e.dataset.allLayer] = e.checked; regionalContext = !Object.values(allEnabled).some(Boolean); clearSelection(); updateChangeControls(); render(); });
    $('#side').onchange = e => setProcedureSide(e.target.value);
    $('#palette').onchange = e => setPalette(e.target.value);
    $('#procedure').onchange = e => { procedure = e.target.checked; playStepAnimation(); render(); };
    $('#prelabels').onchange = e => { prelabelsOn = e.target.checked; render(); };
    $('#instruments').onchange = e => { instrumentsOn = e.target.checked; syncControls(); updateChangeControls(); if (mode === '2d')
        showDiagram();
    else
        render(); };
    $('#focusChange').onclick = () => applyRecommendedView();
    $('#regionalContext').onclick = toggleFullContext;
    $('#compareChange').onclick = () => { startPoseBlend(); beforeCurrent = !beforeCurrent; animationToken++; opening = 1; $('#compareChange').setAttribute('aria-pressed', beforeCurrent); $('#compareChange').textContent = beforeCurrent ? 'After' : 'Before'; if (mode === '2d')
        showDiagram();
    else
        render(); };
    function playStepAnimation() { const token = ++animationToken; const s = steps[stage]; opening = 1; if (mode !== '3d' || !procedure || !hasStageChange() || matchMedia('(prefers-reduced-motion: reduce)').matches) {
        $('#opening').value = 100;
        $('#openingValue').textContent = '100%';
        render();
        return;
    } const start = performance.now(); opening = 0; function tick(now) { if (token !== animationToken || mode !== '3d' || !procedure)
        return; opening = Math.min(1, (now - start) / 1500); $('#opening').value = Math.round(opening * 100); $('#openingValue').textContent = Math.round(opening * 100) + '%'; render(); if (opening < 1)
        requestAnimationFrame(tick); } requestAnimationFrame(tick); }
    $('#replay').onclick = () => { beforeCurrent = false; $('#compareChange').setAttribute('aria-pressed', false); $('#compareChange').textContent = 'Before'; playStepAnimation(); };
    $('#opening').oninput = e => { animationToken++; opening = +e.target.value / 100; $('#openingValue').textContent = Math.round(opening * 100) + '%'; render(); };
    function visible(m) { if (m.hidden || hiddenWhileSeparated(m))
        return false; const side = meshSide(m); if (contextSide !== 'both' && side !== 'both' && side !== contextSide && !wholeScalp(m))
        return false; if (m.marker)
        return procedure && side === operativeSide() && m.markerStages.includes(steps[stage].id) && (!m.markerFissure || m.markerFissure === config.fissure); if (m.instrument) {
        if (beforeCurrent || side !== operativeSide() || !instrumentsOn)
            return false;
        const match = m.instrumentStage === steps[stage].visualAction && (m.instrumentKind !== 'dura-wide' || config.fissure === 'yes') && (m.instrumentKind !== 'dura-low' || config.fissure === 'no');
        if (!match)
            return false;
        if (m.instrumentKind === 'burr-1')
            return opening <= .5 || opening === 1;
        if (m.instrumentKind === 'burr-2')
            return opening >= .5;
        return true;
    } if (m.group === 'arachnoid' && arachnoidSealed(m))
        return false; if (SCALP_OVERLAP.has(m.id) && scalpShownOn(meshSide(m)))
        return false; return !!allEnabled[m.group] || !!enabled[m.group] && relevantMesh(m); }
    // The scalp is one covering: when a step shows the whole-head scalp (contextGroups) or its layer is expanded (All),
    // the contralateral half is drawn too (intact), so a one-sided procedure does not show a half-scalped head.
    function wholeScalp(m) { return ['skin', 'galea', 'pericranium'].includes(m.group) && !m.marker && (!!allEnabled[m.group] || !!steps[stage]?.contextGroups?.includes(m.group)); }
    // Atlas context sheets that coincide with the illustrative scalp (epicranial aponeurosis, frontalis, occipitalis,
    // temporoparietalis, atlas superficial temporal fascia): hidden while skin or galea of their side is drawn.
    const SCALP_OVERLAP = new Set(['CTX-44', 'CTX-45', 'CTX-46', 'CTX-47', 'CTX-78', 'CTX-79', 'CTX-100', 'CTX-101', 'ZA-29', 'ZA-30']);
    function scalpShownOn(side) { return meshes.some(x => (x.group === 'skin' || x.group === 'galea') && (side === 'both' || meshSide(x) === side) && visible(x)); }
    // The translucent arachnoid is hidden while a closed dura of its side is drawn, so it never veils or floats
    // over a closed dura. On the operative side the dura counts as open from the dural opening until it has
    // fully closed again (the closing animation keeps the before state until it ends, as in surgicalState).
    function duraOpenOnSide(m) { if (!procedure || !onOperativeSide(m))
        return false; const s = steps[stage], state = { ...(beforeCurrent ? s.beforeState : s.state) }; if (!beforeCurrent && s.change === 'dura' && opening < 1)
        state.dura = s.beforeState.dura || s.state.dura; return !!state.dura; }
    function arachnoidSealed(m) { const side = meshSide(m); if (duraOpenOnSide(m))
        return false; return meshes.some(d => d.group === 'dura' && meshSide(d) === side && visible(d)); }
    function opacity(m) { return Math.max(0, Math.min(1, m.opacity ?? 1)); }
    function surgicalState(m) { let cut = 0, motion = 0, action = 1, windowAction = 1, burrAction = 0; // The isolated skull-base view keeps only the part of a vein near its sinus (record isolateKeep).
    if (m.isolateKeep && procedure && steps[stage]?.isolateBone)
        return { cut: 9, motion, action, windowAction, burrAction, keep: m.isolateKeep }; if (!procedure || m.marker || m.instrument || !onOperativeSide(m))
        return { cut, motion, action, windowAction, burrAction }; const s = steps[stage], state = { ...(beforeCurrent ? s.beforeState : s.state) }; const a = id => { if (beforeCurrent || !s.changes.includes(id))
        return 1; return s.state[id] ? opening : 1 - opening; }; if (!beforeCurrent && opening < 1)
        for (const id of s.changes)
            state[id] = s.beforeState[id] || s.state[id]; if (m.group === 'bone') {
        if (state.burr)
            burrAction = a('burr');
        if (!beforeCurrent && s.id === 'bone-replace' && s.beforeState.burr)
            burrAction = 1 - opening;
        if (state.bone) {
            cut = m.nativeName === 'Sphenoid bone' && state.clinoid ? 4 : 1;
            action = a('bone');
            windowAction = 1;
        }
    } if (m.nativeName === 'Sphenoid bone' && state.clinoid) {
        cut = state.bone ? 4 : 6;
        action = a('clinoid');
    } if (m.group === 'dura' && state.dura && stateProgress('dura') > DURA_CUT) {
        // The opening is cut along the C first (groove), then the leaf folds back (second pass, see deformation()):
        // the stationary dura loses the whole opening once the leaf starts to move.
        cut = config.fissure === 'no' ? 5 : 3;
        action = 1;
    } if (m.group === 'arachnoid' && m.fissureLine?.length > 1 && state.fissure && config.fissure === 'yes') {
        cut = 8;
        action = a('fissure');
    } // motion (reported for QA): 1 fascial flap, 2 temporalis, 3 scalp, 4 pericranium; the movement itself is the
    // curl/peel returned by deformation().
    if (FASCIA_IDS.has(baseId(m)) && state.muscle)
        motion = config.muscle === 'subfascial' ? 1 : 2;
    if (isTemporalis(m) && (config.muscle === 'split' ? state.muscle : state.temporalis))
        motion = 2;
    if (m.flap?.key === 'scalp' && state.scalp)
        motion = 3;
    if (m.flap?.key === 'pericranium' && state.pericranium)
        motion = 4;
    return { cut, motion, action, windowAction, burrAction }; }
    // Model point -> view space (JS mirror of the vertex shader: p = uView * (q - uTarget)).
    function transform(p) { const { M, target: t } = viewBasis(); return mApply(M, [p[0] - t[0], p[1] - t[1], p[2] - t[2]]); }
    // Eye position in the canonical (operative-side = right) model frame used by the fragment shader (vOriginal).
    function cameraPosition() { const { M, target: t } = viewBasis(); return canonicalPoint(vAdd(t, mApply(mT(M), [-pan[0], -pan[1], distance]))); }
    function project(p) { const t = transform(p), w = distance - t[2]; return [(t[0] + pan[0]) * scale * viewZoom / aspect / w * host.clientWidth / 2 + host.clientWidth / 2, host.clientHeight / 2 - (t[1] + pan[1]) * scale * viewZoom / w * host.clientHeight / 2]; }
    const smoothstep01 = t => t * t * (3 - 2 * t);
    // ---- Soft-tissue curl / peel ------------------------------------------------------------------------------------
    // Flaps fold by the FLAP CONTRACT shared with the data build (record.flap = {key, pivot, axis, away, lift, radius,
    // angle}; reference: tools/anatomy_build/scalp.py flap_deform). With v = p - pivot, d = v.away and h = v.lift, a
    // point with d > 0 wraps a cylinder of radius R about the hinge for an arc of R*T and the rest of the flap continues
    // straight, folded by T. Normals turn with it, the baked AO (normal length) is kept, and only the away/lift plane
    // is used, so x-mirrored fields work for left-sided records. The renderer refines the contract into a peel:
    //  1. Hinge travel: the hinge starts at the flap's free edge (d = dmax) and moves to the base (d = 0) while the
    //     freed part is held folded by thetaS (dissection from the incision toward the base); then the fold completes
    //     to the final angle Tf. At progress 1 the result is the contract with the hinge at the record pivot.
    //  2. Hinge height: the pivot moves along `lift` to the highest point of the motion's outer surface near the
    //     current hinge line, tabulated over the axial coordinate a = v.axis (CURL_NA bins) and the hinge travel
    //     (CURL_ND rows). The roll therefore stays tangent to the tissue, and deeper layers folding about the same hinge
    //     roll outside shallower ones (row 0 plays the role of the record's optional hingeProfile).
    // The same mapping runs in the vertex shader and in curlPoint (labels, picking anchors, depth sort). Points are raw
    // model coordinates of the mesh's own side; the head pose lives only in uView.
    const SHELL_C = [0, -.4, -.4], CURL_NA = 16, CURL_ND = 24, INC_N = 24, PART_FALLOFF = .3;
    // Peel tuning per motion: R bend radius, thetaS fold held during the hinge travel, P1 progress at which the hinge
    // reaches the base, Tf final angle (record angle when omitted). Chosen with numerical checks on the build data at
    // progress 0.25-1: scalp, galea, pericranium, fascial flap and dural leaf stay outside the skull, the stationary
    // scalp and layers reflected earlier (scalp self-contact below 0.2% of sampled vertices). The reflected temporalis
    // clears the zygoma; its anterior edge still touches the rolled scalp near the lateral orbital rim (about 7% of
    // sampled vertices mid-animation, 0.5% at the end).
    const CURL_TUNING = { scalp: { R: .32, thetaS: .8 * Math.PI, P1: .8 }, pericranium: { R: .32, thetaS: .6 * Math.PI, P1: .75 }, fascia: { R: .32, thetaS: .6 * Math.PI, P1: .75 }, temporalis: { R: .3, thetaS: .75 * Math.PI, P1: .8, Tf: .95 * Math.PI }, dura: { R: .12, thetaS: .7 * Math.PI, P1: .85, Tf: .9 * Math.PI } };
    // Drape. Beyond its roll the contract folds a flap rigidly; on the curved head the far part of the large scalp flap
    // then stands off the face like a sail (its curvature flips when it is turned over). Toward the end of the fold the
    // straight part is therefore moved along the flap's outer side (li, which faces the head once folded) by a table
    // D(a, e) over the axial coordinate a and the straight length e = d - R*Tf (DRAPE_NA x DRAPE_NE, bilinear), so the
    // skin side comes to rest DRAPE_ALLOW (9 mm: rolled sponge plus the facial soft tissue the model lacks) above the
    // support: the facial skeleton and the stationary scalp, pits such as the orbit bridged (at most DRAPE_DEEP below a
    // 5-cell envelope). The table is built per side from the skin flap; the galea, fascial, myocutaneous and pericranial
    // motions in the same frame use the same table, so stacked layers keep their order and thickness. The shift fades
    // in over DRAPE_BLEND of the straight part beyond the roll and with the last part of the fold (weight wD).
    // The dural leaf uses the same mechanism with its own reference (the dura inside the opening) and support (bone
    // outside the window cylinder, DRAPE_ALLOW_DURA = 0.4 mm), and no pull where no bone lies beneath (the opening),
    // so beyond the window rim the reflected leaf lies on the outer table instead of floating above it.
    const DRAPE_NA = 12, DRAPE_NE = 8, DRAPE_ALLOW = .45, DRAPE_ALLOW_DURA = .02, DRAPE_BLEND = .4, DRAPE_DEEP = .25, DRAPE_KEYS = new Set(['scalp', 'myo', 'fascia', 'pericranium', 'dura-wide', 'dura-low']), SMALL_BONE = /incus|malleus|stapes|concha|ethmoid|hyoid|vomer|palatine|sinus/i;
    const drapeCache = {};
    function drapeTable(side, mo) { const dura = mo.key.startsWith('dura'), skinFlap = meshesWhere(side, x => x.group === 'skin' && x.flap?.key === 'scalp')[0], o = dura && DURA_OPENINGS[mo.key === 'dura-low' ? 'low' : 'wide']; if (!dura && !skinFlap)
        return null; const allow = dura ? DRAPE_ALLOW_DURA : DRAPE_ALLOW, ck = side + '|' + mo.key + '|' + [...mo.pivot, ...mo.axis, ...mo.away, ...mo.lift, mo.R, mo.Tf].map(x => x.toFixed(5)).join(','); if (ck in drapeCache)
        return drapeCache[ck]; const NA = DRAPE_NA, NE = DRAPE_NE, R = mo.R, T = mo.Tf, RT = R * T, c = Math.cos(T), sn = Math.sin(T), awf = vAdd(vScale(mo.away, c), vScale(mo.lift, sn)), lif = vSub(vScale(mo.lift, c), vScale(mo.away, sn)), row0 = mo.prof.rows.subarray(0, CURL_NA), hpAt = a => { const fi = Math.max(0, Math.min(CURL_NA - 1, (a - mo.prof.a0) * mo.prof.ia)), i0 = Math.floor(fi), i1 = Math.min(i0 + 1, CURL_NA - 1); return row0[i0] + (row0[i1] - row0[i0]) * (fi - i0); };
        // skin-top samples of the flap in (a, e, t): t is the height above the rigidly folded plane along li
        const ref = dura ? gatherPoints(meshesWhere(side, x => x.group === 'dura'), (x, y, z) => insideSplit({ kind: o.kind }, [x, y, z], side)) : gatherPoints([skinFlap], outerOnShell), A = [], E = [], H = []; let a0 = Infinity, a1 = -Infinity, em = .1; for (let i = 0; i < ref.length; i += 3) {
            const v = [ref[i] - mo.pivot[0], ref[i + 1] - mo.pivot[1], ref[i + 2] - mo.pivot[2]], a = vDot(v, mo.axis), e = vDot(v, mo.away) - RT;
            if (e <= -.2) continue;
            A.push(a); E.push(e); H.push(vDot(v, mo.lift) - hpAt(a)); a0 = Math.min(a0, a); a1 = Math.max(a1, a); em = Math.max(em, e);
        } if (!A.length)
        return drapeCache[ck] = null; const da = Math.max(1e-3, (a1 - a0) / (NA - 1)), de = em / (NE - 1), sum = new Float64Array(NA * NE), cnt = new Float64Array(NA * NE); for (let k = 0; k < A.length; k++) {
            const i = Math.round((A[k] - a0) / da), j = Math.round(E[k] / de);
            if (i >= 0 && i < NA && j >= 0 && j < NE) { sum[j * NA + i] += H[k]; cnt[j * NA + i]++; }
        } const top = Array.from(sum, (x, k) => cnt[k] ? x / cnt[k] : NaN), gf = new Float64Array(NA * NE).fill(Infinity), gc = new Float64Array(NA * NE).fill(Infinity);
        // support: bone (without the small internal bones) and every other skin record, every 2nd vertex
        for (const m of meshes) {
            if (m.marker || m.instrument || !(m.group === 'bone' && !SMALL_BONE.test(m.name || m.nativeName || '') || !dura && m.group === 'skin' && m !== skinFlap)) continue;
            ensureGeometry(m);
            const f = m.vertexData;
            for (let p = 0; p < f.length; p += dura ? 6 : 12) {
                if (dura && Math.abs(f[p]) > 1.9 && ((f[p + 1] - .3) / 1.55) ** 2 + ((f[p + 2] - 1.25) / 1.65) ** 2 < 1) continue; // removed bone flap
                const w = [f[p] - mo.pivot[0], f[p + 1] - mo.pivot[1], f[p + 2] - mo.pivot[2]], a = vDot(w, mo.axis), fa = (a - a0) / da;
                if (fa < -2 || fa > NA + 1) continue;
                const hb = hpAt(a) + R * (1 - c), fe = (vDot(w, awf) - R * sn * c - hb * sn) / de;
                if (fe < -2 || fe > NE + 1) continue;
                const t = vDot(w, lif) + R * sn * sn - hb * c;
                for (let j = Math.max(0, Math.ceil(fe - 2)); j <= Math.min(NE - 1, Math.floor(fe + 2)); j++)
                    for (let i = Math.max(0, Math.ceil(fa - 2)); i <= Math.min(NA - 1, Math.floor(fa + 2)); i++) {
                        const k = j * NA + i;
                        if (!(t > top[k] - (dura ? .1 : .5))) continue;
                        if (t < gc[k]) gc[k] = t;
                        if (Math.abs(fa - i) <= .75 && Math.abs(fe - j) <= .75 && t < gf[k]) gf[k] = t;
                    }
            }
        } const raw = top.map((x, k) => { const g = Math.min(gf[k], gc[k] + DRAPE_DEEP); return Number.isFinite(x) && Number.isFinite(g) ? g - allow - x : dura ? 0 : NaN; }), ok = []; raw.forEach((x, k) => { if (Number.isFinite(x)) ok.push(k); }); if (!ok.length)
        return drapeCache[ck] = null; const full = raw.map((x, k) => { if (Number.isFinite(x)) return x; const j = Math.floor(k / NA), i = k % NA; let best = Infinity, acc = 0, n = 0; for (const q of ok) { const dd = (Math.floor(q / NA) - j) ** 2 + (q % NA - i) ** 2; if (dd < best) { best = dd; acc = 0; n = 0; } if (dd === best) { acc += raw[q]; n++; } } return acc / n; }), tab = new Float32Array(NA * NE), K = [.25, .5, .25];
        for (let j = 0; j < NE; j++)
            for (let i = 0; i < NA; i++) {
                let v = 0;
                for (let y = -1; y <= 1; y++) for (let x = -1; x <= 1; x++) v += K[y + 1] * K[x + 1] * full[Math.max(0, Math.min(NE - 1, j + y)) * NA + Math.max(0, Math.min(NA - 1, i + x))];
                tab[j * NA + i] = v;
            }
        return drapeCache[ck] = { a0, ia: 1 / da, ie: 1 / de, RT, tab }; }
    function drapeAt(dr, a, d) { const fa = Math.max(0, Math.min(DRAPE_NA - 1, (a - dr.a0) * dr.ia)), fe = Math.max(0, Math.min(DRAPE_NE - 1, (d - dr.RT) * dr.ie)), i0 = Math.floor(fa), j0 = Math.floor(fe), i1 = Math.min(i0 + 1, DRAPE_NA - 1), j1 = Math.min(j0 + 1, DRAPE_NE - 1), u = fa - i0, w = fe - j0, t = dr.tab; return (t[j0 * DRAPE_NA + i0] * (1 - u) + t[j0 * DRAPE_NA + i1] * u) * (1 - w) + (t[j1 * DRAPE_NA + i0] * (1 - u) + t[j1 * DRAPE_NA + i1] * u) * w; }
    // Canonical (right-sided) hinge frames for tissues without record fields, mirrored in x for the left side.
    // Temporalis: hinge along the outer surface of the atlas muscle from just above the posterior root of the zygomatic
    // arch (z -0.7) to the fossa behind the frontal process of the zygomatic bone (z 2.3), kept above the arch and the
    // zygoma so the reflected muscle clears them; the muscle anterior to the incision is peeled from the superior
    // temporal line downward and reflected inferiorly over the arch (subfascial branch).
    // Scalp fallback (data without scalp records): the scalp build's frame, so the fascial flap still folds sensibly.
    const TEMPORALIS_FRAME = { pivot: [-3.4102, -1.8227, -.7344], axis: [.2809, .0352, .9591], away: [-.1871, .9822, .0188], lift: [-.9413, -.1848, .2825] };
    const SCALP_FRAME_FALLBACK = { pivot: [-1.954321, .956362, 4.080434], axis: [-.406128, -.913032, -.037852], away: [-.802196, .376053, -.463752], lift: [-.437655, .157978, .885156], radius: .25, angle: 3.769911 };
    // Dural openings in canonical y/z (the same ellipses as the fragment-shader cuts) and the base of the reflected
    // leaf: a chord at `chord` of the radius toward `base` (unit y/z direction, anteroinferior toward the sphenoid
    // ridge). The C-shaped cut runs along the rest of the rim.
    const DURA_OPENINGS = { wide: { kind: 1, c: [.3, 1.25], r: [1.08, 1.08], base: [-.8, .6], chord: .8 }, low: { kind: 2, c: [-.62, 1.65], r: [.38, .55], base: [-.6, .8], chord: .72 } };
    const DURA_CUT = .3; // share of the dural step spent cutting along the C before the leaf folds
    const FASCIA_IDS = new Set(['ILL-superfascia', 'ILL-deepfascia', 'ILL-interfat', 'ILL-suprafat']);
    const clamp01 = x => Math.max(0, Math.min(1, x)), mirrorX = (v, s) => [v[0] * s, v[1], v[2]];
    function frameOf(spec, s = 1) { return { pivot: mirrorX(spec.pivot, s), axis: mirrorX(spec.axis, s), away: mirrorX(spec.away, s), lift: mirrorX(spec.lift, s) }; }
    function sideOf(m) { return meshSide(m) === 'left' ? 'left' : 'right'; }
    function meshesWhere(side, pred) { return meshes.filter(x => !x.marker && !x.instrument && sideOf(x) === side && meshSide(x) !== 'both' && pred(x)); }
    // Vertex positions (flat xyz) of meshes passing `keep(x, y, z, nx, ny, nz)`.
    function gatherPoints(list, keep = () => true) { const out = []; for (const m of list) {
        ensureGeometry(m);
        const f = m.vertexData;
        for (let i = 0; i < f.length; i += 6)
            if (keep(f[i], f[i + 1], f[i + 2], f[i + 3], f[i + 4], f[i + 5]))
                out.push(f[i], f[i + 1], f[i + 2]);
    } return out; }
    const outerOnShell = (x, y, z, nx, ny, nz) => { const r = [x - SHELL_C[0], y - SHELL_C[1], z - SHELL_C[2]]; return (nx * r[0] + ny * r[1] + nz * r[2]) / ((Math.hypot(...r) * Math.hypot(nx, ny, nz)) || 1) > .5; };
    // Hinge-height table (see 2. above): rows over hinge travel d in [0, dmax], CURL_NA bins over a; each entry is the
    // highest h of the source points within a band around that row, gaps interpolated along a, 3-tap smoothed.
    function curlProfile(fr, pts) { const n = pts.length / 3, A = [], D = [], H = []; let a0 = Infinity, a1 = -Infinity, dmax = 0; for (let i = 0; i < n; i++) {
        const v = [pts[3 * i] - fr.pivot[0], pts[3 * i + 1] - fr.pivot[1], pts[3 * i + 2] - fr.pivot[2]], d = vDot(v, fr.away);
        if (d <= -.05)
            continue;
        const a = vDot(v, fr.axis);
        A.push(a); D.push(d); H.push(vDot(v, fr.lift)); a0 = Math.min(a0, a); a1 = Math.max(a1, a); dmax = Math.max(dmax, d);
    } if (!A.length)
        return null; const da = Math.max(1e-3, (a1 - a0) / (CURL_NA - 1)), dd = Math.max(1e-3, dmax / (CURL_ND - 1)), band = Math.max(dd, .1), rows = new Float32Array(CURL_ND * CURL_NA).fill(-Infinity); for (let i = 0; i < A.length; i++) {
        const ai = Math.max(0, Math.min(CURL_NA - 1, Math.round((A[i] - a0) / da)));
        for (let j = Math.max(0, Math.ceil((D[i] - band) / dd)); j <= Math.min(CURL_ND - 1, Math.floor((D[i] + band) / dd)); j++)
            rows[j * CURL_NA + ai] = Math.max(rows[j * CURL_NA + ai], H[i]);
    } for (let j = 0; j < CURL_ND; j++) {
        const r = rows.subarray(j * CURL_NA, (j + 1) * CURL_NA), ok = [];
        r.forEach((x, i) => { if (Number.isFinite(x)) ok.push(i); });
        if (!ok.length) {
            if (j) r.set(rows.subarray((j - 1) * CURL_NA, j * CURL_NA)); else r.fill(0);
            continue;
        }
        for (let i = 0; i < CURL_NA; i++)
            if (!Number.isFinite(r[i])) {
                const lo = ok.filter(x => x < i).pop(), hi = ok.find(x => x > i);
                r[i] = lo === undefined ? r[hi] : hi === undefined ? r[lo] : r[lo] + (r[hi] - r[lo]) * (i - lo) / (hi - lo);
            }
        const c = Array.from(r);
        for (let i = 1; i < CURL_NA - 1; i++) r[i] = (c[i - 1] + 2 * c[i] + c[i + 1]) / 4;
    } return { a0, ia: 1 / da, dmax, rows }; }
    function maxAway(fr, pts, floor = 0) { let d = floor; for (let i = 0; i < pts.length; i += 3) d = Math.max(d, (pts[i] - fr.pivot[0]) * fr.away[0] + (pts[i + 1] - fr.pivot[1]) * fr.away[1] + (pts[i + 2] - fr.pivot[2]) * fr.away[2]); return d; }
    // Incision polyline of a side (from the scalp flap's incisionLine, resampled to INC_N points by arc length) and
    // the sign that turns the polyline's perpendicular (tangent x shell radial) toward the flap.
    const incisionCache = {};
    function incisionFor(side) { if (side in incisionCache)
        return incisionCache[side]; const flap = meshesWhere(side, x => x.group === 'skin' && x.flap && x.incisionLine?.length > 1)[0]; if (!flap)
        return incisionCache[side] = null; const L = flap.incisionLine, acc = [0]; for (let i = 1; i < L.length; i++)
        acc.push(acc[i - 1] + Math.hypot(...vSub(L[i], L[i - 1]))); const pts = []; for (let k = 0, i = 1; k < INC_N; k++) {
        const s = acc[acc.length - 1] * k / (INC_N - 1);
        while (i < L.length - 1 && acc[i] < s) i++;
        const f = clamp01((s - acc[i - 1]) / ((acc[i] - acc[i - 1]) || 1));
        pts.push(L[i - 1].map((x, j) => x + (L[i][j] - x) * f));
    } const inc = { pts, flat: new Float32Array(pts.flat()), sign: 1 }; ensureGeometry(flap); const f = flap.vertexData, sides = []; for (let i = 0; i < f.length && sides.length < 4000; i += 6 * 7) {
        const c = incisionCoords(inc, [f[i], f[i + 1], f[i + 2]]);
        if (c.dist < .3) sides.push(c.side);
    } sides.sort((a, b) => a - b); inc.sign = sides.length && sides[sides.length >> 1] < 0 ? -1 : 1; return incisionCache[side] = inc; }
    // Nearest incision segment measured in the plane tangent to the shell: signed offset (along tangent x radial),
    // arc parameter (0 at the zygomatic end E0, 1 at the midline end E1) and distance. Mirrors the vertex shader.
    function incisionCoords(inc, p) { const rad = vNorm(vSub(p, SHELL_C)); let best = Infinity, bt = 0, tan = [0, 1, 0], be = [0, 0, 0]; for (let i = 0; i < INC_N - 1; i++) {
        const a = inc.pts[i], ab = vSub(inc.pts[i + 1], a), t = clamp01(vDot(vSub(p, a), ab) / Math.max(vDot(ab, ab), 1e-8));
        let e = vSub(p, vAdd(a, vScale(ab, t)));
        e = vSub(e, vScale(rad, vDot(e, rad)));
        const d = Math.hypot(e[0], e[1], e[2]);
        if (d < best) { best = d; bt = (i + t) / (INC_N - 1); tan = ab; be = e; }
    } const perp = vNorm(vCross(vNorm(tan), rad)); return { side: vDot(be, perp), t: bt, dist: best, perp }; }
    // Motions per side and key, built lazily from the data: frame, tuning, hinge-height table and hinge travel.
    const motionCache = {};
    function motionFor(side, key) { const ck = side + '|' + key; if (ck in motionCache)
        return motionCache[ck]; const s = side === 'left' ? -1 : 1, skinFlap = meshesWhere(side, x => x.group === 'skin' && x.flap?.key === 'scalp')[0], tune = CURL_TUNING[key.startsWith('dura') ? 'dura' : key === 'myo' ? 'scalp' : key]; let fr = null, prof = null, dmax = 0, R = tune.R, Tf = tune.Tf; const scalpFrame = () => skinFlap ? frameOf(skinFlap.flap) : frameOf(SCALP_FRAME_FALLBACK, s), scalpProfile = fr0 => {
        const src = skinFlap ? gatherPoints(meshesWhere(side, x => x.group === 'skin' && !x.contextOnly), outerOnShell) : gatherPoints(meshesWhere(side, x => baseId(x) === 'ILL-superfascia'), (x, y, z, nx, ny, nz) => vDot([nx, ny, nz], fr0.lift) > .3);
        return curlProfile(fr0, src);
    }, inRegion = (list, sp) => gatherPoints(list, (x, y, z) => insideSplit(sp, [x, y, z], side)); if (key === 'scalp' || key === 'myo') {
        fr = scalpFrame();
        prof = scalpProfile(fr);
        dmax = skinFlap ? maxAway(fr, gatherPoints([skinFlap])) : prof?.dmax;
        Tf = Tf ?? (skinFlap?.flap.angle || SCALP_FRAME_FALLBACK.angle);
    }
    else if (key === 'pericranium') {
        const peri = meshesWhere(side, x => x.group === 'pericranium' && x.flap)[0];
        fr = peri ? frameOf(peri.flap) : scalpFrame();
        prof = scalpProfile(fr);
        dmax = peri ? maxAway(fr, gatherPoints([peri])) : 0;
        Tf = Tf ?? (peri?.flap.angle || SCALP_FRAME_FALLBACK.angle);
    }
    else if (key === 'fascia') {
        fr = scalpFrame();
        prof = scalpProfile(fr);
        dmax = maxAway(fr, inRegion(meshesWhere(side, x => FASCIA_IDS.has(baseId(x))), splitFor('fascia', side)));
        Tf = Tf ?? (skinFlap?.flap.angle || SCALP_FRAME_FALLBACK.angle);
    }
    else if (key === 'temporalis') {
        fr = frameOf(TEMPORALIS_FRAME, s);
        const muscle = meshesWhere(side, isTemporalis), region = inRegion(muscle, splitFor('temporalis', side));
        prof = curlProfile(fr, gatherPoints(muscle, (x, y, z, nx, ny, nz) => vDot([nx, ny, nz], fr.lift) > .3 && insideSplit(splitFor('temporalis', side), [x, y, z], side)));
        dmax = maxAway(fr, region);
    }
    else if (key.startsWith('dura')) {
        const o = DURA_OPENINGS[key === 'dura-low' ? 'low' : 'wide'], dura = meshesWhere(side, x => x.group === 'dura'), sp = { kind: o.kind }, pts = inRegion(dura, sp);
        if (pts.length >= 9) {
            // lift: mean outward normal inside the opening; pivot: the dura vertex nearest the middle of the base chord.
            let nrm = [0, 0, 0], best = Infinity, pivot = null;
            const chordYZ = [o.c[0] + o.base[0] * o.chord * o.r[0], o.c[1] + o.base[1] * o.chord * o.r[1]];
            for (const m of dura) {
                const f = m.vertexData;
                for (let i = 0; i < f.length; i += 6) {
                    const p = [f[i], f[i + 1], f[i + 2]];
                    if (insideSplit(sp, p, side)) nrm = vAdd(nrm, vNorm([f[i + 3], f[i + 4], f[i + 5]]));
                    const dd = (f[i + 1] - chordYZ[0]) ** 2 + (f[i + 2] - chordYZ[1]) ** 2;
                    if (dd < best) { best = dd; pivot = p; }
                }
            }
            // away: perpendicular to the chord (direction (-base_z r_y, base_y r_z) in y/z), toward the centre.
            const lift = vNorm(nrm), inward = [0, -o.base[0] * o.r[1], -o.base[1] * o.r[0]], away = vNorm(vSub(inward, vScale(lift, vDot(inward, lift))));
            fr = { pivot, lift, away, axis: vNorm(vCross(lift, away)) };
            prof = curlProfile(fr, pts);
            dmax = maxAway(fr, pts);
        }
    } if (!fr || !prof || !(dmax > 0))
        return motionCache[ck] = null; const mo = { key, ...fr, R, Tf, thetaS: Math.min(tune.thetaS, Tf), P1: tune.P1, dmax, prof }; mo.drape = DRAPE_KEYS.has(key) ? drapeTable(side, mo) : null; return motionCache[ck] = mo; }
    // Hinge travel dh and fold angle T for progress P (see 1. above).
    function curlSchedule(mo, P) { const dh = mo.dmax * (1 - smoothstep01(clamp01(P / mo.P1))), T = mo.thetaS * smoothstep01(clamp01(P / .15)) + (mo.Tf - mo.thetaS) * smoothstep01(clamp01((P - mo.P1 + .05) / (1.05 - mo.P1))); return { dh, T }; }
    // Per-draw curl state: schedule plus the hinge-height row interpolated at the current hinge travel.
    function curlState(mo, P) { if (!mo || !(P > 0))
        return null; const { dh, T } = curlSchedule(mo, Math.min(1, P)); if (T <= 1e-4)
        return null; const rows = mo.prof.rows, f = Math.max(0, Math.min(CURL_ND - 1, dh / mo.prof.dmax * (CURL_ND - 1))), j0 = Math.floor(f), j1 = Math.min(j0 + 1, CURL_ND - 1), w = f - j0, row = new Float32Array(CURL_NA); for (let i = 0; i < CURL_NA; i++)
        row[i] = rows[j0 * CURL_NA + i] * (1 - w) + rows[j1 * CURL_NA + i] * w; const wD = mo.drape ? (mo.Tf - mo.thetaS > 1e-3 ? smoothstep01(clamp01((T - mo.thetaS) / (mo.Tf - mo.thetaS))) : +(T >= mo.Tf - 1e-3)) : 0; return { mo, P, dh, T, row, wD }; }
    // Curl one point (and optionally its normal, rotated with the roll only: the vertex shader instead differentiates the
    // whole map, including the hinge-height shear and the drape). Mirrors the vertex shader's curlF.
    function curlPoint(cs, p, n) { const { mo, dh, T, row } = cs, v = vSub(p, mo.pivot), d = vDot(v, mo.away), dd = d - dh; if (dd <= 0)
        return n ? [p, n] : p; const fi = Math.max(0, Math.min(CURL_NA - 1, (vDot(v, mo.axis) - mo.prof.a0) * mo.prof.ia)), i0 = Math.floor(fi), i1 = Math.min(i0 + 1, CURL_NA - 1), hp = row[i0] + (row[i1] - row[i0]) * (fi - i0), hh = vDot(v, mo.lift), R = mo.R, s = Math.min(dd, R * T), ph = s / R, c = Math.cos(ph), sn = Math.sin(ph), aw = vAdd(vScale(mo.away, c), vScale(mo.lift, sn)), li = vSub(vScale(mo.lift, c), vScale(mo.away, sn)), rest = vSub(v, vAdd(vScale(mo.away, d), vScale(mo.lift, hh))); let q = [mo.pivot, vScale(mo.away, dh + R * sn), vScale(mo.lift, hp + R * (1 - c)), rest, vScale(aw, dd - s), vScale(li, hh - hp)].reduce(vAdd); if (cs.wD > 0 && dd > s)
        q = vAdd(q, vScale(li, cs.wD * smoothstep01(clamp01((dd - s) / DRAPE_BLEND)) * drapeAt(mo.drape, vDot(v, mo.axis), d))); if (!n)
        return q; const na = vDot(n, mo.away), nl = vDot(n, mo.lift); return [q, vAdd(n, vAdd(vScale(vSub(aw, mo.away), na), vScale(vSub(li, mo.lift), nl)))]; }
    // Incision parting: skin and galea edges near the cut move apart (perpendicular to the cut in the tangent plane)
    // behind a knife point that travels from E0 to E1 with the incision progress; the gap tapers out at E1.
    function partPoint(pt, p) { const c = incisionCoords(pt.inc, p), open = smoothstep01(clamp01((pt.reveal * 1.05 - c.t) / .05)) * (1 - smoothstep01(clamp01((c.t - .94) / .06))), fall = 1 - smoothstep01(clamp01(c.dist / PART_FALLOFF)); return vAdd(p, vScale(c.perp, pt.sign * pt.gap * fall * open)); }
    // Two-pass regions: 1/2 = dural opening ellipses (y/z only, so mirror-free), 3 = flap side of the incision (beyond `margin`)
    // and, optionally, above a plane (n.p > w; temporalis above the zygomatic arch).
    // The temporalis moves only anterior to the incision; in the myocutaneous branch (scalp hinge) also only above
    // y = -2.3, so its insertion deep to the zygomatic arch stays (the arch's upper border is at y -1.8 to -2.75).
    function splitFor(key, side) { if (key === 'fascia' || key === 'temporalis')
        return { kind: 3, inc: incisionFor(side), margin: .08 }; if (key === 'myo')
        return { kind: 3, inc: incisionFor(side), margin: .08, plane: [0, 1, 0, -2.3] }; return null; }
    function insideSplit(sp, p, side) { if (!sp)
        return true; if (sp.kind === 1 || sp.kind === 2) {
        const o = DURA_OPENINGS[sp.kind === 1 ? 'wide' : 'low'], u = (p[1] - o.c[0]) / o.r[0], w = (p[2] - o.c[1]) / o.r[1];
        return u * u + w * w < 1;
    } let v = Infinity; if (sp.inc)
        v = incisionCoords(sp.inc, p).side * sp.inc.sign - sp.margin; if (sp.plane)
        v = Math.min(v, vDot(p, sp.plane) - sp.plane[3]); return v > 0; }
    // Marking reveal (uMarkReveal). Markings draw in with the stage progress: kinds 1/2 sweep by angle about
    // markerCenter or along markerAxis. The incision and closure stitches use kind 3 instead, the arc parameter along
    // the scalp incision (the same parameter as the incision parting, E0 at the zygomatic arch to E1 at the midline),
    // so the incision line draws in exactly behind the knife. While a returning scalp flap is still travelling (galea
    // closure) these skin markings wait until it is home. The dural opening line draws in with the C-shaped dural cut
    // (groove) and stays on the rim, as the closure line, while the dura is closed. Galeal stitches (on the galea, galea
    // closure) draw in as the galea closes; skin stitches and the skin incision line belong to the skin closure.
    const INCISION_MARKS = new Set(['MARK-incision', 'MARK-scalp-sutures', 'MARK-galea-sutures']), SKIN_GAP = .065, SKIN_GAP_WIDE = .12;
    function incisionMarker(m) { return INCISION_MARKS.has(baseId(m).replace(/-l$/, '')) ? incisionFor(sideOf(m)) : null; }
    function duraMarkReveal(m, gr) { if (!(gr > 0))
        return 0; const o = DURA_OPENINGS[/^MARK-dural-low/.test(baseId(m)) ? 'low' : 'wide'], a0 = Math.atan2(o.base[1], o.base[0]) + Math.acos(o.chord), th = a0 + gr * (2 * Math.PI - 2 * Math.acos(o.chord)), c = m.markerCenter || [0, o.c[0], o.c[1]], y = o.c[0] + o.r[0] * Math.cos(th), z = o.c[1] + o.r[1] * Math.sin(th); let t = (Math.atan2(z - c[2], y - c[1]) + Math.PI) / (2 * Math.PI); if (th < 1.5 * Math.PI && t > .5)
        t = 0; return t; }
    function markReveal(m) { if (beforeCurrent)
        return 0; const s = steps[stage], id = baseId(m).replace(/-l$/, ''); if (/^MARK-dural-(cut|low)$/.test(id) && s.changes.includes('dura'))
        return duraMarkReveal(m, s.state.dura ? clamp01(stateProgress('dura') / DURA_CUT) : 1); if (id === 'MARK-galea-sutures' && s.changes.includes('galea') && !s.state.galea)
        return (1 - stateProgress('galea')) * 1.05; // buried galeal stitches draw in along the cut as the galea closes
    if (incisionMarker(m)) {
        if (s.changes.includes('skin') && s.state.skin)
            return stateProgress('skin') * 1.05; // knife tip (partPoint opens where 1.05 * progress exceeds t)
        if (s.changes.includes('scalp') && !s.state.scalp) {
            const home = s.stagger?.scalp?.[1] ?? 1;
            return home < 1 ? clamp01((opening - home) / (1 - home)) * 1.05 : opening >= 1 ? 1.05 : 0;
        }
        return opening * 1.05;
    } return opening; }
    function superficialVessel(m) { return /^(Superficial temporal artery|Frontal branch of superficial temporal artery)\.[lr]$/.test(m.nativeName || '') || /^ILL-vein-stv(-l)?$/.test(m.id); }
    function isTemporalis(m) { return /^Temporalis muscle\.[lr]$/.test(m.nativeName) || baseId(m) === 'ILL-subfat'; }
    // Progress (0..1) of a state flag in the current step: 1/0 when static, animated by the stage progress (opening)
    // when the step changes it, optionally within a sub-range of the step (step.stagger[id] = [from, to]).
    function stateProgress(id) { const s = steps[stage]; if (!s)
        return 0; if (beforeCurrent || !s.changes.includes(id))
        return (beforeCurrent ? s.beforeState : s.state)[id] ? 1 : 0; const r = s.stagger?.[id] || [0, 1], t = clamp01((opening - r[0]) / (r[1] - r[0])); return s.state[id] ? t : 1 - t; }
    // Deformation of a mesh in the current state: { curl, split, part, groove, leafCut } or null. Memoised per state.
    let deformKey = '', deformCache = new Map();
    function deformation(m) { const key = [stage, opening, beforeCurrent, procedure, contextSide, config.muscle, config.fissure, steps.length, separateLayers].join('|'); if (key !== deformKey) {
        deformKey = key;
        deformCache = new Map();
    } if (!deformCache.has(m.index ?? m.id))
        deformCache.set(m.index ?? m.id, computeDeformation(m)); return deformCache.get(m.index ?? m.id); }
    function computeDeformation(m) { if (!procedure || m.instrument || !steps[stage])
        return null; if (m.marker) {
        // Markers that follow a tissue ride its curl (without its two-pass region or incision parting). Exceptions:
        // the dural opening line stays on the rim as the cut / closure line, and in the subfascial technique the
        // flap-reflection arrow stays over the temporalis as a direction cue instead of disappearing into the thin
        // fascial fold (in the myocutaneous branch it rides the thick flap, so it never floats over bare bone).
        const f = m.follow && followMesh(m), still = f?.group === 'dura' || baseId(m).replace(/-l$/, '') === 'MARK-flap-arrow' && config.muscle === 'subfascial', d = f && !still && deformation(f);
        return d?.curl ? { curl: d.curl } : null;
    } if (!onOperativeSide(m) || meshSide(m) === 'both' || m.contextOnly && !superficialVessel(m) || separateLayers && separationIndex(m))
        return null; const side = sideOf(m), id = baseId(m), split = config.muscle === 'split', out = {}; if ((m.group === 'skin' || m.group === 'galea') && (m.flap || id === 'ILL-scalp' || id === 'ILL-galea')) {
        const g = stateProgress(m.group), inc = incisionFor(side), st = steps[stage], closing = !beforeCurrent && st.changes.includes(m.group) && !st.state[m.group];
        // Skin edges are held a little wider while the galea is closed (its stitches show between them) and close from
        // that width in the skin step.
        const galeaClosing = !beforeCurrent && st.changes.includes('galea') && !st.state.galea, gap = m.group === 'skin' ? (closing ? SKIN_GAP_WIDE * g : galeaClosing ? SKIN_GAP + (SKIN_GAP_WIDE - SKIN_GAP) * smoothstep01(1 - stateProgress('galea')) : SKIN_GAP) : .045 * (closing ? g : 1);
        if (g > 0 && inc) out.part = { inc, reveal: closing ? 1 : g, gap, sign: (m.flap ? 1 : -1) * inc.sign };
    } if (m.flap?.key === 'scalp')
        out.curl = curlState(motionFor(side, 'scalp'), stateProgress('scalp'));
    else if (m.flap?.key === 'pericranium')
        out.curl = curlState(motionFor(side, 'pericranium'), stateProgress('pericranium'));
    else if (FASCIA_IDS.has(id)) {
        out.curl = curlState(motionFor(side, split ? 'myo' : 'fascia'), stateProgress(split ? 'scalp' : 'muscle'));
        out.split = splitFor('fascia', side);
    }
    else if (superficialVessel(m)) {
        // Superficial temporal vessels lie in the subcutaneous layer (re-seated by the scalp build step): the part
        // anterior to the incision travels with the scalp flap; the rest stays with the stationary scalp.
        out.curl = curlState(motionFor(side, 'scalp'), stateProgress('scalp'));
        out.split = { kind: 3, inc: incisionFor(side), margin: .065 };
    }
    else if (isTemporalis(m)) {
        out.curl = split ? curlState(motionFor(side, 'myo'), stateProgress('scalp')) : curlState(motionFor(side, 'temporalis'), stateProgress('temporalis'));
        out.split = splitFor(split ? 'myo' : 'temporalis', side);
    }
    else if (m.group === 'dura') {
        const p = stateProgress('dura'), o = DURA_OPENINGS[config.fissure === 'no' ? 'low' : 'wide'];
        // groove from the first frame of a step that changes the dura (at 0 nothing is cut yet, so the amber outline
        // and the dural marking draw in with the cut); Before shows the planned opening as a full outline.
        if (p > 0 || !beforeCurrent && steps[stage].changes.includes('dura')) {
            const a0 = Math.atan2(o.base[1], o.base[0]) + Math.acos(o.chord);
            out.groove = [Math.min(1, p / DURA_CUT), a0, 2 * Math.PI - 2 * Math.acos(o.chord), o.kind];
            if (p > DURA_CUT) {
                out.curl = curlState(motionFor(side, config.fissure === 'no' ? 'dura-low' : 'dura-wide'), (p - DURA_CUT) / (1 - DURA_CUT));
                out.split = { kind: o.kind };
                out.leafCut = true;
            }
        }
    } if (out.split && !out.split.inc && out.split.kind === 3)
        out.split = null; if (out.split && !out.curl && !out.leafCut)
        out.split = null; return out.curl || out.part || out.groove || out.leafCut ? out : null; }
    function followMesh(m) { if (!('followRef' in m))
        m.followRef = meshes.find(x => x.id === m.follow || x.nativeName === m.follow) || null; return m.followRef; }
    // Rest-pose point -> displayed point: incision parting, curl (inside the moving region of two-pass meshes), then
    // the translation (separated layers; instruments follow their tissue's translation only).
    function movePoint(m, p) { const d = meshOffset(m), def = m.instrument ? null : deformation(m); let q = p; if (def?.part)
        q = partPoint(def.part, q); if (def?.curl && (!def.split || insideSplit(def.split, p, sideOf(m))))
        q = curlPoint(def.curl, q); return vAdd(q, d); }
    function separationIndex(m) { return ({ 'ILL-subfat': 1, 'ILL-deepfascia': 2, 'ILL-interfat': 3, 'ILL-superfascia': 4, 'ILL-suprafat': 5 })[baseId(m)] || 0; }
    function hiddenWhileSeparated(m) { return separateLayers && (['skin', 'galea', 'pericranium'].includes(m.group) && !m.marker || m.marker && m.markerSurface === 'skin'); }
    function meshOffset(m) { if ((m.instrument || m.marker) && m.follow) {
        const follow = followMesh(m);
        return follow ? meshOffset(follow) : [0, 0, 0];
    } const d = [0, 0, 0]; if (separateLayers) {
        // Exploded fascial planes: the temporal laminae and fat pads, drawn in their rest pose (no curl, see
        // computeDeformation); the scalp layers and skin markings are hidden meanwhile (see visible).
        d[0] += (meshSide(m) === 'left' ? 1 : -1) * separationIndex(m) * .35;
    } return d; }
    // Rodrigues rotation (bone flap travel uses the same rotation in the shader through uFlapAngle/uFlapAxis/uFlapPivot).
    function rotateAbout(p, h) { if (!h || !h.angle)
        return p; const c = Math.cos(h.angle), sn = Math.sin(h.angle), k = h.axis, v = p.map((x, i) => x - h.pivot[i]), kv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2], cr = [k[1] * v[2] - k[2] * v[1], k[2] * v[0] - k[0] * v[2], k[0] * v[1] - k[1] * v[0]]; return v.map((x, i) => h.pivot[i] + x * c + cr[i] * sn + k[i] * kv * (1 - c)); }
    function labelPoint(m) { return movePoint(m, m.labelAnchor || m.anchor); }
    $('#separateLayers').onclick = () => { separateLayers = !separateLayers; $('#separateLayers').setAttribute('aria-pressed', separateLayers); $('#separateLayers').textContent = separateLayers ? 'Join Fascial Layers' : 'Separate Fascial Layers'; if (separateLayers) {
        enabled.fascia = enabled.deepfascia = enabled.fat = enabled.muscle = true;
        syncControls();
    } clearSelection(); render(); };
    const labelVisibility = new Set(), surfaceAnchors = new Map();
    let pickPixels = null, pickWidth = 0, pickHeight = 0;
    function scanPickAnchors(pixels, w, h, cssw, cssh) { const stats = new Map(), out = new Map(); for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
            const k = (y * w + x) * 4, id = pixels[k] + (pixels[k + 1] << 8) + (pixels[k + 2] << 16) - 1;
            if (id < 0 || id >= meshes.length)
                continue;
            let s = stats.get(id);
            if (!s) {
                s = { x: 0, y: 0, n: 0, best: Infinity };
                stats.set(id, s);
            }
            s.x += x;
            s.y += y;
            s.n++;
        } for (const s of stats.values()) {
        s.x /= s.n;
        s.y /= s.n;
    } for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
            const k = (y * w + x) * 4, id = pixels[k] + (pixels[k + 1] << 8) + (pixels[k + 2] << 16) - 1, s = stats.get(id);
            if (!s)
                continue;
            const d = (x - s.x) ** 2 + (y - s.y) ** 2;
            if (d < s.best) {
                s.best = d;
                out.set(id, { x: (x + .5) / w * cssw, y: (h - y - .5) / h * cssh });
            }
        } return out; }
    function cacheLabelVisibility() { labelVisibility.clear(); surfaceAnchors.clear(); if (!available || !pickTarget)
        return; pickWidth = pickTarget.w; pickHeight = pickTarget.h; if (!pickPixels || pickPixels.length !== pickWidth * pickHeight * 4)
        pickPixels = new Uint8Array(pickWidth * pickHeight * 4); gl.readPixels(0, 0, pickWidth, pickHeight, gl.RGBA, gl.UNSIGNED_BYTE, pickPixels); }
    function pickIdAt(cssx, cssy) { if (!pickPixels || cssx < 0 || cssy < 0 || cssx >= host.clientWidth || cssy >= host.clientHeight || !host.clientWidth || !host.clientHeight)
        return -1; const x = Math.min(pickWidth - 1, Math.floor(cssx / host.clientWidth * pickWidth)), y = Math.min(pickHeight - 1, Math.floor((host.clientHeight - cssy) / host.clientHeight * pickHeight)); let best = -1, bestD = Infinity; for (let dy = -3; dy <= 3; dy++)
        for (let dx = -3; dx <= 3; dx++) {
            const xx = x + dx, yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= pickWidth || yy >= pickHeight)
                continue;
            const at = (yy * pickWidth + xx) * 4, id = pickPixels[at] + (pickPixels[at + 1] << 8) + (pickPixels[at + 2] << 16) - 1, d = dx * dx + dy * dy;
            if (id >= 0 && id < meshes.length && d < bestD) {
                best = id;
                bestD = d;
            }
        } return best; }
    function procedureLabelRegion(m) { const sign = m.nativeName.endsWith('.l') ? 1 : -1; let target = [...m.anchor], radius = 2.2, cranial = false; if (m.group === 'dura') {
        target = [-3.1 * sideSign(), config.fissure === 'no' ? -.62 : .3, config.fissure === 'no' ? 1.65 : 1.25];
        radius = config.fissure === 'no' ? .9 : 1.7;
    }
    else if (/^Internal carotid artery/.test(m.nativeName)) {
        target = [sign * .8, -1.7, 1.1];
        radius = .85;
        cranial = true;
    }
    else if (m.nativeName === 'Sphenoid bone') {
        target = [-.8 * sideSign(), -1., 1.25];
        radius = 1.1;
    }
    else if (/^Optic nerve/.test(m.nativeName)) {
        target = [sign * .65, -1.45, 1.5];
        radius = .9;
    }
    else if (/^Middle cerebral artery/.test(m.nativeName)) {
        target = [sign * 1.2, -1.2, 1.1];
        radius = .9;
    }
    else if (/^Insular branches/.test(m.nativeName)) {
        target = [sign * 2.1, -.6, .9];
        radius = .8;
    }
    else if (m.nativeName === 'Frontal bone') {
        target = [-2.5 * sideSign(), 1.2, 2];
        radius = 1.6;
    }
    else if (m.nativeName.startsWith('Temporal bone.')) {
        target = [(m.nativeName.endsWith('.l') ? 3 : -3), -.5, 1.0];
        radius = 1.8;
    }
    else if (['skin', 'galea', 'pericranium'].includes(m.group))
        radius = 3.4; // anchors are bounding-box centres inside the head; reach the shell
    if (m.instrument) {
        target = [...m.anchor];
        radius = .4;
    } return { target, radius, cranial }; }
    function pointRetained(m, p) { const s = surgicalState(m), [x, y, z] = canonicalPoint(p), a = s.action; if ((s.cut === 1 || s.cut === 4) && x < -1.9 && ((y - .3) / 1.55) ** 2 + ((z - 1.25) / 1.65) ** 2 < s.windowAction * s.windowAction)
        return false; if ((s.cut === 4 || s.cut === 6) && ((x + .8) / .46) ** 2 + ((y + 1) / .55) ** 2 + ((z - 1.25) / .67) ** 2 < a * a)
        return false; if (s.cut === 3 && ((y - .3) / 1.08) ** 2 + ((z - 1.25) / 1.08) ** 2 < a * a)
        return false; if (s.cut === 5 && ((y + .62) / .38) ** 2 + ((z - 1.65) / .55) ** 2 < a * a)
        return false; if (s.burrAction > 0 && x < -1.9) {
        const first = Math.min(1, s.burrAction * 2) * .255, second = Math.max(0, s.burrAction * 2 - 1) * .255;
        if ((y + .7) ** 2 + (z - 2.45) ** 2 < first * first || (y + 1) ** 2 + (z - .8) ** 2 < second * second)
            return false;
    } return true; }
    function candidatesForLabel(m) { const spec = procedureLabelRegion(m), key = spec.target.join(',') + '|' + spec.radius; if (m.procedureCandidates?.key === key)
        return m.procedureCandidates.points; const a = m.vertexData, points = [], seen = new Set(); if (a)
        for (let i = 0; i < a.length; i += 6) {
            const p = [a[i], a[i + 1], a[i + 2]];
            if (spec.cranial && p[1] < -2.25)
                continue;
            const d = p.reduce((sum, x, j) => sum + (x - spec.target[j]) ** 2, 0);
            if (d > spec.radius ** 2)
                continue;
            const k = p.map(x => x.toFixed(5)).join(',');
            if (seen.has(k))
                continue;
            seen.add(k);
            points.push({ p, d });
        } points.sort((a, b) => a.d - b.d); // large shells (scalp layers): keep an even subsample so a hidden region costs at most ~1200 moved points per frame
    const keep = points.length > 1200 ? points.filter((_, i) => i % Math.ceil(points.length / 1200) === 0) : points; m.procedureCandidates = { key, points: keep.map(x => x.p) }; return m.procedureCandidates.points; }
    function regionSurfaceAnchor(m) { const points = candidatesForLabel(m), spec = procedureLabelRegion(m); for (const p of points) {
        if (!pointRetained(m, p))
            continue;
        const world = movePoint(m, p), q = project(world), x = Math.floor(q[0] / host.clientWidth * pickWidth), y = Math.floor((host.clientHeight - q[1]) / host.clientHeight * pickHeight);
        if (!pickPixels || q[0] < 0 || q[1] < 0 || q[0] >= host.clientWidth || q[1] >= host.clientHeight)
            continue;
        let found = false;
        for (let dy = -1; dy <= 1 && !found; dy++)
            for (let dx = -1; dx <= 1; dx++) {
                const xx = x + dx, yy = y + dy;
                if (xx < 0 || yy < 0 || xx >= pickWidth || yy >= pickHeight)
                    continue;
                const at = (yy * pickWidth + xx) * 4;
                if (pickPixels[at] + (pickPixels[at + 1] << 8) + (pickPixels[at + 2] << 16) - 1 === m.index) {
                    found = true;
                    break;
                }
            }
        if (found) {
            labelVisibility.add(m.index);
            return { x: q[0], y: q[1], visible: true, world };
        }
    } const q = project(movePoint(m, spec.target)); return { x: q[0], y: q[1], visible: false }; }
    function activeLabelMesh() { if (selected >= 0)
        return visible(meshes[selected]) ? meshes[selected] : null; if (!prelabelsOn)
        return null; const s = steps[stage]; if (s.id === 'orient')
        return meshes.find(m => m.marker && baseId(m) === 'MARK-pterion' && visible(m)); if (s.id === 'incision' || s.id === 'prepare') {
        const mark = meshes.find(m => m.marker && baseId(m) === 'MARK-incision' && visible(m));
        if (mark || s.id === 'incision')
            return mark || null;
    } if (s.id === 'burr-holes')
        return meshes.find(m => m.marker && baseId(m) === (opening < .5 ? 'MARK-burr-1' : 'MARK-burr-2') && visible(m)); if (s.id === 'clinoid' || s.id === 'retain')
        return meshes.find(m => m.nativeName === 'Sphenoid bone' && visible(m)); if (s.change === 'dura')
        return meshes.find(m => baseId(m) === 'ILL-dura' && onOperativeSide(m) && visible(m)); return s.labels.map(name => operativeSide() === 'left' ? name.replace(/^Right /, 'Left ') : name).map(name => meshes.find(m => m.name === name && visible(m))).find(Boolean) || null; }
    function labelName(m) { let name = m.name; if (selected < 0 && m.nativeName === 'Sphenoid bone' && ['clinoid', 'retain'].includes(steps[stage].id))
        name = 'Anterior Clinoid Region';
    else if (selected < 0 && m.group === 'dura' && steps[stage].change === 'dura')
        name = 'Dural Opening'; return titleCase(name.replace(/^(Right|Left) /, '').replace(/ — (original illustration|illustration|schematic)$/i, '')); }
    function fixedLabelRect(w, h) { return { x: Math.max(8, Math.min(14, host.clientWidth - w - 8)), y: Math.max(8, Math.min(isMobile ? 36 : 86, host.clientHeight - h - (isMobile ? 64 : 100))), w, h }; }
    let labelKey = '';
    function clearSelection() { selected = -1; labelKey = ''; render(); }
    function select(i) { if (i < 0 || i >= meshes.length || !visible(meshes[i])) {
        clearSelection();
        return;
    } selected = i; labelKey = ''; render(); }
    function positionPopup() { return null; }
    function updateStepLabels() { const popup = $('#popup'), svg = $('#labelLines'); $('#stepLabels').replaceChildren(); svg.replaceChildren(); if (mode !== '3d') {
        popup.hidden = true;
        return;
    } const m = activeLabelMesh(); if (!m) {
        popup.hidden = true;
        labelKey = '';
        return;
    } const key = m.id + '|' + stage + '|' + (selected >= 0 ? 'selected' : 'auto'); if (labelKey !== key) {
        popup.replaceChildren();
        const close = document.createElement('button');
        close.className = 'close';
        close.textContent = '×';
        close.setAttribute('aria-label', 'Hide anatomy label');
        close.onclick = () => { selected = -1; prelabelsOn = false; labelKey = ''; syncControls(); render(); };
        const title = document.createElement('strong');
        title.textContent = labelName(m);
        title.title = m.description;
        const status = document.createElement('div');
        status.className = 'label-status';
        status.id = 'labelStatus';
        const row = document.createElement('div');
        row.className = 'popup-actions';
        for (const [label, fn] of [['Hide', () => { m.hidden = true; clearSelection(); }], ['Isolate', () => { meshes.forEach(x => x.hidden = x !== m); for (const k in enabled)
                    enabled[k] = k === m.group; if (m.instrument)
                    instrumentsOn = true; selected = m.index; syncControls(); render(); }]]) {
            const b = document.createElement('button');
            b.textContent = label;
            b.style.width = 'auto';
            b.onclick = fn;
            row.append(b);
        }
        popup.append(close, title, status, row);
        labelKey = key;
    } popup.hidden = isMobile && !!$('#layerControls').open; // on phones the open Layers & View dock takes precedence
    const a = regionSurfaceAnchor(m), r = fixedLabelRect(popup.offsetWidth || 190, popup.offsetHeight || 75); popup.style.left = r.x + 'px'; popup.style.top = r.y + 'px'; $('#labelStatus').textContent = a.visible ? '' : procedureLabelRegion(m).cranial ? 'Cranial segment hidden in this view' : 'Region hidden in this view'; if (a.visible) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', a.x);
        line.setAttribute('y1', a.y);
        line.setAttribute('x2', Math.max(r.x, Math.min(r.x + r.w, a.x)));
        line.setAttribute('y2', Math.max(r.y, Math.min(r.y + r.h, a.y)));
        svg.append(line);
    } }
    // Head-framed anatomical presets; like before they keep the orbit target and distance (re-expressed in model
    // coordinates if the camera was room-fixed).
    function setView(v) { const views = { anterior: [0, 0], right: [Math.PI / 2, 0], left: [-Math.PI / 2, 0], superior: [0, 1.5] }; if (!views[v])
        return; stopGlide(); surgeonChosen = false; attachCamera('head'); applyCamera({ frame: 'head', yaw: views[v][0], pitch: views[v][1] }); pan = [0, 0]; render(); }
    // Surgeon's View (room-framed): from the head of the table, above and behind the vertex, looking down at the
    // operative field as the seated surgeon sees it (screen up = toward the patient's feet). It is aimed at the bone
    // window centre in the pose the step requests, so it also frames the field while the pose is still gliding.
    function setSurgeonView() { stopGlide(); surgeonChosen = true; const sign = sideSign(), P = poseMatrix(requestedPose()); applyCamera({ frame: 'world', eye: [-.3 * sign, .62, .72], up: [0, 0, 1], target: posePoint(canonicalPoint([-2.7, .3, 1.25]), P), distance: 14 }); viewZoom = 1; pan = [0, 0]; render(); }
    // ---- Patient Position controls (Layers & View panel) -------------------------------------------------------------
    // Changing the pose re-attaches the camera to the room first (same image), so the head visibly turns while the
    // table, holder and gizmo stay put. Slider input applies directly; the toggle and the defaults glide.
    function headFixed() { if (poseOverride)
        return poseOverride === 'operative'; const s = steps[stage]; return !!s && !!(beforeCurrent ? s.beforeState.position : s.state.position); }
    function poseState() { return { shown: { ...poseShown }, requested: requestedPose(), fraction: poseFraction(), settings: { ...poseSettings }, override: poseOverride, defaults: { ...POSE_DEFAULT }, pivot: [...POSE_PIVOT], matrix: [...poseR], headFixed: headFixed(), props: propsOn && propsVisible() }; }
    // Reported by guideQA.getState(): the orientation plus best-effort legacy yaw/pitch/roll of the effective
    // head-relative view (screenOrientation -1 when the image is turned by more than 90 degrees).
    function cameraState() { const M = viewBasis().M, a = anglesOf(M); return { orientation: { q: [...viewQ], frame: camFrame, matrix: [...M] }, yaw: a.yaw, pitch: a.pitch, roll: a.roll, screenOrientation: Math.abs(a.roll) > Math.PI / 2 ? -1 : 1, pose: poseState(), gravity: gravityInModel() }; }
    function poseInput(k) { return $(`[data-pose="${k}"]`); }
    // Set operative-pose angles (degrees, clamped to the slider ranges). If the step shows the anatomical pose the
    // operative pose is switched on, otherwise the change would be invisible.
    function setPose(partial = {}) { attachCamera('world'); if (poseFraction() < 1) {
        startPoseBlend();
        poseOverride = 'operative';
    } for (const k of POSE_KEYS)
        if (Number.isFinite(+partial[k]) && partial[k] !== null && partial[k] !== '') {
            const el = poseInput(k);
            poseSettings[k] = el ? Math.max(+el.min, Math.min(+el.max, +partial[k])) : +partial[k];
        } syncPoseControls(); render(); }
    // 'anatomical' | 'operative' override the step until the next step change; null follows the step again.
    function setPoseMode(next) { if (!['anatomical', 'operative', null].includes(next))
        return; attachCamera('world'); startPoseBlend(); poseOverride = next; syncPoseControls(); render(); }
    function resetPose() { attachCamera('world'); startPoseBlend(); poseSettings = { ...POSE_DEFAULT }; syncPoseControls(); render(); }
    let poseUiKey = '';
    function syncPoseControls() { if (!$('#positionControls'))
        return; const operative = poseFraction() >= .5, key = POSE_KEYS.map(k => poseSettings[k]).join() + operative + propsOn + camFrame; if (key === poseUiKey)
        return; poseUiKey = key; for (const k of POSE_KEYS) {
        const el = poseInput(k);
        if (+el.value !== poseSettings[k])
            el.value = poseSettings[k];
        el.closest('.pose-slider').querySelector('output').textContent = Math.round(poseSettings[k]) + '°';
    } $('#poseAnatomical').setAttribute('aria-pressed', !operative); $('#poseOperative').setAttribute('aria-pressed', operative); $('#poseProps').checked = propsOn; }
    $$('[data-pose]').forEach(el => el.oninput = () => setPose({ [el.dataset.pose]: +el.value }));
    $('#poseAnatomical').onclick = () => setPoseMode('anatomical');
    $('#poseOperative').onclick = () => setPoseMode('operative');
    $('#poseReset').onclick = resetPose;
    $('#poseProps').onchange = e => { propsOn = e.target.checked; poseUiKey = ''; render(); };
    $('#surgeonView').onclick = setSurgeonView;
    $('#surgeonViewPanel').onclick = setSurgeonView;
    // ---- Schematic room cues: head holder and table edge ----------------------------------------------------------
    // Thin-tube line drawings in the style of the coloured markings: a Mayfield-type three-pin skull clamp fixed to the
    // head (single pin behind the operative-side ear, two-pin rocker on the contralateral side, C-frame behind the
    // occiput, swivel arm to the table) and the head end of the table fixed in the room. Pin sites are illustrative,
    // chosen away from the incision: points ray-cast onto the atlas calvaria (canonical right-sided procedure) with the
    // outward surface normals listed. Shown only in room-fixed views (camFrame 'world') once the head is fixed
    // (step 'position' onward or the Operative toggle), so head-framed anatomical views are unchanged.
    const HOLDER_PINS = { single: { at: [-3.32, -.77, -2.66], n: [-.95, -.09, -.3] }, pair: [{ at: [3.24, .31, 1.18], n: [.98, .05, .18] }, { at: [3.09, -.29, -3.23], n: [.89, .08, -.44] }] };
    // Room coordinates: table top 2.4 cm below the lowest occipital point in the neutral pose, head edge at the
    // shoulders about 12 cm below the craniocervical pivot, 44 cm wide; only the head edge and 10 cm of the side rails
    // are drawn. Illustrative.
    const TABLE = { topZ: -6.4, headY: -10, halfWidth: 11, footY: -15, mount: [0, -10, -7.2] };
    const PROP_COLORS = { holder: hexRgb('#7fd6e8'), table: hexRgb('#8aa4ae') };
    let propBuffer = null, propCache = null;
    function propsVisible() { return propsOn && (mode === '3d' || capturing) && camFrame === 'world' && headFixed(); }
    // Append an unindexed tube (position + unit normal per vertex) along a polyline; parallel-transport frames.
    function tubeInto(out, pts, r, caps = false, sides = 10) { const n = pts.length; if (n < 2)
        return; const T = pts.map((p, i) => vNorm(vSub(pts[Math.min(n - 1, i + 1)], pts[Math.max(0, i - 1)]))), rings = []; let u = vNorm(vCross(T[0], Math.abs(T[0][1]) < .9 ? [0, 1, 0] : [1, 0, 0])); for (let i = 0; i < n; i++) {
        if (i)
            u = vNorm(vSub(u, vScale(T[i], vDot(u, T[i]))));
        const v = vCross(T[i], u);
        rings.push(Array.from({ length: sides }, (_, k) => { const a = 2 * Math.PI * k / sides, d = vAdd(vScale(u, Math.cos(a)), vScale(v, Math.sin(a))); return [vAdd(pts[i], vScale(d, r)), d]; }));
    } const put = ([p, nn]) => out.push(p[0], p[1], p[2], nn[0], nn[1], nn[2]); for (let i = 0; i + 1 < n; i++)
        for (let k = 0; k < sides; k++) {
            const a = rings[i][k], b = rings[i][(k + 1) % sides], c = rings[i + 1][k], d = rings[i + 1][(k + 1) % sides];
            [a, c, b, b, c, d].forEach(put);
        } if (caps)
        for (const [i, dir] of [[0, -1], [n - 1, 1]]) {
            const nn = vScale(T[i], dir);
            for (let k = 0; k < sides; k++)
                [[pts[i], nn], [rings[i][k][0], nn], [rings[i][(k + 1) % sides][0], nn]].forEach(put);
        } }
    // Prop vertices in model coordinates for the displayed pose (room-fixed parts go through poseInverse).
    function propGeometry() { const sign = sideSign(), key = sign + '|' + poseR.map(x => x.toFixed(4)).join(); if (propCache?.key === key)
        return propCache; const mx = p => [p[0] * sign, p[1], p[2]], along = (pin, d) => mx(vAdd(pin.at, vScale(pin.n, d))), holder = [], table = [], single = HOLDER_PINS.single, [pa, pb] = HOLDER_PINS.pair; tubeInto(holder, [along(single, 0), along(single, .9)], .035); tubeInto(holder, [along(single, .9), along(single, 1.7)], .11, true); for (const pin of [pa, pb]) {
        tubeInto(holder, [along(pin, 0), along(pin, .7)], .035);
        tubeInto(holder, [along(pin, .7), along(pin, 1.1)], .09, true);
    } // Rocker: a bar bowed outward between the two pin barrels, with a stem from its middle.
    const ra = along(pa, 1.1), rb = along(pb, 1.1), outward = vNorm(mx(vAdd(pa.n, pb.n))), rmid = vScale(vAdd(ra, rb), .5); tubeInto(holder, Array.from({ length: 13 }, (_, i) => vAdd(vAdd(vScale(ra, 1 - i / 12), vScale(rb, i / 12)), vScale(outward, .45 * Math.sin(Math.PI * i / 12)))), .08); const stem = vAdd(rmid, vScale(outward, 1.25)); tubeInto(holder, [vAdd(rmid, vScale(outward, .45)), stem], .1, true); // C-frame: half ellipse from the single-pin barrel to the rocker stem, passing behind the occiput.
    const s0 = along(single, 1.7), half = vScale(vSub(stem, s0), .5), mid = vAdd(s0, half), back = [0, -.25, -1], w = vScale(vNorm(vSub(back, vScale(half, vDot(back, half) / vDot(half, half)))), 5.4), arc = Array.from({ length: 41 }, (_, i) => vAdd(vSub(mid, vScale(half, Math.cos(Math.PI * i / 40))), vScale(w, Math.sin(Math.PI * i / 40)))); tubeInto(holder, arc, .1); tubeInto(holder, [arc[20], poseInverse(TABLE.mount)], .09); const room = p => poseInverse(p), { topZ: z, headY: y0, halfWidth: hw, footY: y1 } = TABLE; for (const [a, b] of [[[-hw, y0, z], [hw, y0, z]], [[-hw, y0, z], [-hw, y1, z]], [[hw, y0, z], [hw, y1, z]], [[-hw, y0, z - .9], [hw, y0, z - .9]], [[-hw, y0, z], [-hw, y0, z - .9]], [[hw, y0, z], [hw, y0, z - .9]]])
        tubeInto(table, [room(a), room(b)], .07); /* top edge, side rails and the end face of the table top */ propCache = { key, data: new Float32Array([...holder, ...table]), parts: [{ first: 0, count: holder.length / 6, rgb: PROP_COLORS.holder }, { first: holder.length / 6, count: table.length / 6, rgb: PROP_COLORS.table }], uploaded: false }; return propCache; }
    function bindVertexBuffer(buf) { gl.bindBuffer(gl.ARRAY_BUFFER, buf); for (const [name, start] of [['aPosition', 0], ['aNormal', 12]])
        gl.vertexAttribPointer(gl.getAttribLocation(program, name), 3, gl.FLOAT, false, 24, start); }
    // Drawn opaque before the anatomy; in the picking pass as background (id 0) so they hide what they cover.
    function drawProps(picking) { if (!propsVisible())
        return; const geo = propGeometry(); if (!propBuffer)
        propBuffer = gl.createBuffer(); bindVertexBuffer(propBuffer); if (!geo.uploaded) {
        gl.bufferData(gl.ARRAY_BUFFER, geo.data, gl.DYNAMIC_DRAW);
        geo.uploaded = true;
    } for (const [n, v] of [['Material', -1], ['IsMarker', 1], ['Line', 0], ['MarkKind', 0], ['MarkReveal', 1], ['Cut', 0], ['Burrs', 0], ['Change', 0], ['FlapAngle', 0], ['Selected', 0], ['Region', 0], ['FisVeil', 0], ['WindowAction', 1], ['Action', 1], ['Motion', 0], ['Side', sideSign()], ['CurlOn', 0], ['PartOn', 0], ['IncReveal', 0], ['SplitKind', 0], ['SplitPass', 0]])
        gl.uniform1f(uniforms[n], v); gl.uniform4fv(uniforms.Groove, [0, 0, 0, 1]); gl.uniform3fv(uniforms.Offset, [0, 0, 0]); for (const part of geo.parts) {
        gl.uniform4fv(uniforms.Color, picking ? [0, 0, 0, 1] : [...part.rgb, 1]);
        gl.drawArrays(gl.TRIANGLES, part.first, part.count);
    } bindVertexBuffer(buffer); }
    // ---- Orientation gizmo -----------------------------------------------------------------------------------------
    // Viewport corner: the posed head's patient axes (S superior, A anterior, R/L) and an amber gravity arrow onto a
    // small floor tile, seen through the current camera, so pose changes stay legible during free rotation. The
    // caption names the pose and whether the view follows the head or is fixed to the room.
    let gizmoKey = '';
    function updateGizmo() { const box = $('#orientGizmo'); if (!box || mode !== '3d')
        return; const { M } = viewBasis(), W = mMul(M, mT(poseR)), R = 31, f = v => [v[0] * R, -v[1] * R], fx = n => n.toFixed(1), key = M.concat(W).map(x => x.toFixed(3)).join() + poseFraction() + camFrame + POSE_KEYS.map(k => Math.round(poseShown[k])).join(); if (key === gizmoKey)
        return; gizmoKey = key; const line = (a, b, color, width, opacity, dash) => `<line x1="${fx(a[0])}" y1="${fx(a[1])}" x2="${fx(b[0])}" y2="${fx(b[1])}" stroke="${color}" stroke-width="${width}" stroke-linecap="round" opacity="${opacity}"${dash ? ` stroke-dasharray="${dash}"` : ''}/>`, label = (p, s, color, opacity) => `<text x="${fx(p[0])}" y="${fx(p[1] + 3.2)}" fill="${color}" opacity="${opacity}">${s}</text>`, g = mApply(W, [0, 0, -1]), tile = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => f(mApply(W, [a * .4, b * .4, -1.08]))); let svg = `<polygon points="${tile.map(p => fx(p[0]) + ',' + fx(p[1])).join(' ')}" fill="#e9c35a24" stroke="#e9c35a99" stroke-width="1"/>`; const items = [{ d: mApply(M, [0, 1, 0]), text: 'S', color: '#a3db7f' }, { d: mApply(M, [0, 0, 1]), text: 'A', color: '#82bdff' }, { d: mApply(M, [-1, 0, 0]), text: 'R', color: '#ff9483' }, { d: mApply(M, [1, 0, 0]), text: 'L', color: '#ff9483', thin: true }, { d: g, text: 'Floor', color: '#f0c45a', gravity: true }].sort((a, b) => a.d[2] - b.d[2]); for (const it of items) {
        const tip = f(it.d), op = it.d[2] < -.05 ? .42 : 1, len = Math.hypot(tip[0], tip[1]);
        if (it.gravity) {
            const u = len > 1 ? [tip[0] / len, tip[1] / len] : [0, 1], end = [tip[0] * .9, tip[1] * .9];
            svg += line([0, 0], end, it.color, 2, op, '3 2.5') + (len > 4 ? `<polygon points="${fx(end[0] + u[0] * 5)},${fx(end[1] + u[1] * 5)} ${fx(end[0] - u[1] * 3.6)},${fx(end[1] + u[0] * 3.6)} ${fx(end[0] + u[1] * 3.6)},${fx(end[1] - u[0] * 3.6)}" fill="${it.color}" opacity="${op}"/>` : `<circle r="4" fill="none" stroke="${it.color}" stroke-width="1.6" opacity="${op}"/>`) + label(len < 12 ? [0, 15] : [tip[0] * 1.78, tip[1] * 1.78 + 2], it.text, it.color, op);
        }
        else // an axis pointing at the viewer is short: push its label off the centre
            svg += line([0, 0], tip, it.color, it.thin ? 1.3 : 2.6, op) + label(len < 12 ? [tip[0] + (tip[0] >= 0 ? 9 : -9), tip[1] - 8] : [tip[0] * 1.3, tip[1] * 1.3], it.text, it.color, op);
    } box.querySelector('svg').innerHTML = svg + '<circle r="2.4" fill="#d7e5da"/>'; const p = poseShown, posed = POSE_KEYS.some(k => Math.abs(p[k]) > .5); box.querySelector('small').textContent = (posed ? `Pose ${Math.round(p.rotation)}° rot · ${Math.round(p.extension)}° ext${Math.round(p.lateral) ? ` · ${Math.round(p.lateral)}° tilt` : ''}` : 'Anatomical position') + '\n' + (camFrame === 'world' ? 'View fixed to room' : 'View follows head'); }
    $$('[data-view]').forEach(b => b.onclick = () => setView(b.dataset.view));
    $('#resetView').onclick = () => applyRecommendedView();
    $('#manifestDownload').onclick = () => downloadBlob(new Blob([JSON.stringify(manifest, null, 2)], { type: 'application/json' }), 'pterional-mesh-manifest.json');
    $('#exportPlan').onclick = () => downloadBlob(new Blob([JSON.stringify({ model: modelIndex() + 1, config, steps, papers, side: operativeSide(), contextSide, palette, layers: { relevant: { ...enabled }, all: { ...allEnabled } }, positioning: { operativePose: { ...poseSettings }, defaults: { ...POSE_DEFAULT }, units: 'degrees', pivot: [...POSE_PIVOT], note: 'Rigid rotation of the atlas head about an estimated craniocervical pivot: contralateral rotation, then lateral flexion (positive toward the contralateral shoulder), then extension (vertex toward the floor). Defaults follow the example in Rodriguez Rubio et al.; head holder, pins and table are schematic.' }, rendering: (palette === 'surgical' ? 'Surgical palette: per-tissue shading with baked ambient occlusion and procedural surface texture (porosity, pores, fibres, lobules, faint vessel-like patterns); the texture is decorative, not anatomical.' : 'Atlas palette: manifest teaching colours with simple shading and baked ambient occlusion.') + ' Scalp skin, galea and pericranium are illustrative layered shells. Flaps (scalp, pericranium, temporal fascia, temporalis, dural leaf) peel from the cut toward a straight illustrative hinge and roll over it; reflected scalp-layer flaps are laid toward the face and the dural leaf onto the bone beyond the rim; the incision edges part along the cut. These motions are schematic, not soft-tissue simulation. Colored markings are drawn as smooth tubes resting just above the surface they mark, kept at least about 2.6 device pixels wide on screen; the incision line draws in behind the knife.', limitations: manifest.limitations }, null, 2)], { type: 'application/json' }), 'pterional-model-' + String(modelIndex() + 1).padStart(2, '0') + '-reading-plan.json');
    $('#pdfRequests').onclick = () => downloadBlob(new Blob([papers.map(p => `${p.title}\n${p.citation}\n${p.url}\n`).join('\n')], { type: 'text/plain' }), 'pterional-requested-papers.txt');
    mobileQuery.addEventListener?.('change', syncMobileLayout);
    $('#layerControls').addEventListener('toggle', () => { if (isMobile) updateStepLabels(); });
    window.addEventListener?.('resize', () => { if (isMobile)
        requestAnimationFrame(updateMobileReaderHeight); });
    syncMobileLayout();
    new ResizeObserver(() => { if (isMobile)
        updateMobileReaderHeight(); }).observe($('.layout'));
    function ensureGeometry(m) { if (m.geometryReady)
        return; const bytes = decode64(m.data), floats = new Float32Array(bytes.buffer); if (floats.length !== m.count * 6)
        throw Error('Invalid geometry buffer ' + m.id); if (m.group === 'skin')
        tiltSkinWalls(floats); m.vertexData = floats; gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferSubData(gl.ARRAY_BUFFER, m.first * 24, floats); m.geometryReady = true; delete m.data; }
    // Skin slabs: cut-wall triangles (face normal across the shell radial) span the whole slab; tilt each wall vertex
    // normal by about 20 degrees outward (outer-surface vertex) or inward (inner-surface vertex), keeping its length
    // (baked AO). The fragment shader reads the interpolated tilt as depth across the wall (dermis over fat) and the
    // tilt also rounds the shading of the cut edge.
    function tiltSkinWalls(f) { const k = .36; for (let t = 0; t + 18 <= f.length; t += 18) {
        const P = [0, 1, 2].map(i => [f[t + i * 6], f[t + i * 6 + 1], f[t + i * 6 + 2]]), fn = vNorm(vCross(vSub(P[1], P[0]), vSub(P[2], P[0]))), rc = vNorm(vSub(vScale(vAdd(vAdd(P[0], P[1]), P[2]), 1 / 3), SHELL_C));
        if (Math.abs(vDot(fn, rc)) >= .5)
            continue;
        const r = P.map(q => Math.hypot(...vSub(q, SHELL_C))), mid = (Math.max(...r) + Math.min(...r)) / 2;
        if (Math.max(...r) - Math.min(...r) < 1e-4)
            continue;
        for (let i = 0; i < 3; i++) {
            const o = t + i * 6 + 3, n = [f[o], f[o + 1], f[o + 2]], len = Math.hypot(...n) || 1, out = vNorm(vSub(P[i], SHELL_C)), nn = vScale(vNorm(vAdd(vScale(n, 1 / len), vScale(out, r[i] > mid ? k : -k))), len);
            f[o] = nn[0]; f[o + 1] = nn[1]; f[o + 2] = nn[2];
        }
    } }
    function shader(type, text) { const s = gl.createShader(type); gl.shaderSource(s, text); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
        throw Error(gl.getShaderInfoLog(s)); return s; }
    function initGL() {
        gl = canvas.getContext('webgl', { antialias: true, alpha: false, preserveDrawingBuffer: true });
        if (!gl)
            throw Error('WebGL is unavailable');
        const fragmentVectors = gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS) || 16;
        fissureN = fragmentVectors >= 64 ? 12 : fragmentVectors >= 40 ? 8 : 4;
        program = gl.createProgram();
        gl.attachShader(program, shader(gl.VERTEX_SHADER, `attribute vec3 aPosition;attribute vec3 aNormal;uniform mediump float uLine;uniform float uLinePx,uLineNudge;uniform float uDistance,uAspect,uScale,uMotion,uSide;uniform mat3 uView;uniform vec3 uGravity;uniform mediump float uAction;uniform vec2 uPan;uniform vec3 uTarget;uniform vec3 uOffset;uniform float uFlapAngle;uniform vec3 uFlapAxis,uFlapPivot;
uniform float uCurlOn,uCurlIA;uniform vec3 uCurlPivot,uCurlAxis,uCurlAway,uCurlLift;uniform vec4 uCurlP;uniform float uCurlRow[16];uniform float uDrapeW;uniform vec4 uDrapeP;uniform float uDrape[96];uniform float uPartOn,uIncReveal;uniform vec4 uPart;uniform vec3 uInc[24];uniform mediump float uSplitKind;uniform vec4 uSplitPlane;uniform vec2 uSplitInc;
varying mediump vec3 vNormal;varying highp vec3 vOriginal;varying highp vec3 vEye;varying mediump float vAO;varying mediump vec3 vObjN;varying highp float vSplit;varying highp float vIncT;
/* Curl / peel (see curlPoint in JS): hinge travelled uCurlP.z along away, pivot raised along lift to the tabulated
   hinge height uCurlRow over the axial coordinate (start uCurlP.w, 1/step uCurlIA); radius uCurlP.x, angle uCurlP.y.
   Drape (see drapeTable): beyond the roll, shift along li by uDrapeW x the 12 x 8 table uDrape over (a, d - R*Tf);
   uDrapeP = (a0, 1/da, 1/de, R*Tf). */
vec3 curlF(vec3 q){vec3 v=q-uCurlPivot;float d=dot(v,uCurlAway),dd=d-uCurlP.z;if(dd<=0.)return q;float a=dot(v,uCurlAxis),fi=clamp((a-uCurlP.w)*uCurlIA,0.,15.);int i0=int(floor(fi));int i1=i0<15?i0+1:15;float hp=mix(uCurlRow[i0],uCurlRow[i1],fi-float(i0)),hh=dot(v,uCurlLift),R=uCurlP.x,s=min(dd,R*uCurlP.y),ph=s/R,c=cos(ph),sn=sin(ph);vec3 aw=c*uCurlAway+sn*uCurlLift,li=c*uCurlLift-sn*uCurlAway;
 vec3 r=uCurlPivot+(uCurlP.z+R*sn)*uCurlAway+(hp+R*(1.-c))*uCurlLift+(v-d*uCurlAway-hh*uCurlLift)+(dd-s)*aw+(hh-hp)*li;
 if(uDrapeW>0.&&dd>s){float fa=clamp((a-uDrapeP.x)*uDrapeP.y,0.,11.),fe=clamp((d-uDrapeP.w)*uDrapeP.z,0.,7.);int j0=int(floor(fa));int k0=int(floor(fe));int j1=j0<11?j0+1:11;int k1=k0<7?k0+1:7;float u=fa-float(j0),w=fe-float(k0);
  r+=li*(uDrapeW*smoothstep(0.,.4,dd-s)*mix(mix(uDrape[k0*12+j0],uDrape[k0*12+j1],u),mix(uDrape[k1*12+j0],uDrape[k1*12+j1],u),w));}
 return r;}
void main(){vAO=length(aNormal);vOriginal=vec3(aPosition.x*uSide,aPosition.y,aPosition.z);vec3 q=aPosition;vec3 norm=aNormal/max(vAO,1e-4);vSplit=1.;vIncT=0.;
/* Incision (uInc, 24 points of this side): nearest segment in the plane tangent to the scalp shell (centre 0,-.4,-.4).
   uPart = (knife progress, gap, side sign, falloff): skin/galea edges part behind the knife. uSplitKind 3: signed
   offset to the flap side (uSplitInc = flap sign, margin), optionally limited by the plane uSplitPlane (n.p > w). */
if(uPartOn>.5||uSplitKind>2.5||uIncReveal>.5){vec3 rad=normalize(aPosition-vec3(0.,-.4,-.4));float best=1e3,bt=0.;vec3 tg=vec3(0.,1.,0.),be=vec3(0.);
 for(int i=0;i<23;i++){vec3 a=uInc[i],ab=uInc[i+1]-a;float t=clamp(dot(aPosition-a,ab)/max(dot(ab,ab),1e-8),0.,1.);vec3 e=aPosition-a-ab*t;e-=dot(e,rad)*rad;float d=length(e);if(d<best){best=d;bt=(float(i)+t)/23.;tg=ab;be=e;}}
 vec3 perp=normalize(cross(normalize(tg),rad));vIncT=bt;
 if(uSplitKind>2.5)vSplit=min(dot(be,perp)*uSplitInc.x-uSplitInc.y,dot(aPosition,uSplitPlane.xyz)-uSplitPlane.w);
 if(uPartOn>.5)q+=perp*(uPart.z*uPart.y*(1.-smoothstep(0.,uPart.w,best))*smoothstep(0.,.05,uPart.x*1.05-bt)*(1.-smoothstep(.94,1.,bt)));}
/* Curl: the normal is taken from the deformed surface itself (curlF at two offsets along the rest tangent plane), so
   the roll, the hinge-height shear and the drape all shade correctly. */
if(uCurlOn>.5){vec3 t1=normalize(cross(norm,abs(norm.x)<.9?vec3(1.,0.,0.):vec3(0.,1.,0.))),t2=cross(norm,t1);vec3 q0=curlF(q);norm=normalize(cross(curlF(q+.002*t1)-q0,curlF(q+.002*t2)-q0));q=q0;}
if(uFlapAngle!=0.){
  float ct=cos(uFlapAngle),st=sin(uFlapAngle);vec3 k=uFlapAxis,v=q-uFlapPivot;
  q=uFlapPivot+v*ct+cross(k,v)*st+k*dot(k,v)*(1.-ct);
  norm=norm*ct+cross(k,norm)*st+k*dot(k,norm)*(1.-ct);
}
q+=uOffset;
// View: uView = head(model)->view rotation (camera orientation x head pose), uTarget = orbit centre in model space.
// q/norm are model-space (after flap motion); deformations belong above this line. uGravity (unit, model space) is
// provided for later gravity-driven deformation and is optimised away while unused.
if(uLine>0.){float w0=uDistance-(uView*(q-uTarget)).z;q+=norm*max(0.,w0*uLinePx-uLine);}vec3 p=uView*(q-uTarget);vNormal=uView*norm;vObjN=vec3(aNormal.x*uSide,aNormal.yz)/max(vAO,1e-4);float w=uDistance-p.z;vEye=vec3(-(p.x+uPan.x),-(p.y+uPan.y),w);float wz=uLine>0.?max(w-uLineNudge,.05):w;gl_Position=vec4((p.x+uPan.x)*uScale/uAspect,(p.y+uPan.y)*uScale,(1.0125786-1.0062893/wz)*w,w);}`));
        gl.attachShader(program, shader(gl.FRAGMENT_SHADER, `#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
#define FISSURE_N ${fissureN}
uniform mediump float uLine;uniform vec4 uColor;uniform float uSelected,uPick,uRegion,uCut,uChange,uWindowAction,uBurrs,uIsMarker,uMarkKind,uMarkAxis,uMarkReveal,uMaterial,uSurgical,uQuality;uniform vec3 uMarkCenter;uniform vec2 uMarkRange,uViewport;uniform mediump float uAction;uniform vec3 uFissure[FISSURE_N];uniform vec3 uCamPos;uniform float uFisVeil;uniform mediump float uSplitKind;uniform float uSplitPass;uniform vec4 uGroove;uniform vec3 uInner;varying mediump vec3 vNormal;varying vec3 vOriginal;varying vec3 vEye;varying mediump float vAO;varying mediump vec3 vObjN;varying float vSplit;varying float vIncT;
#ifdef GL_FRAGMENT_PRECISION_HIGH
float hsh(vec3 p){p=fract(p*.1031);p+=dot(p,p.zyx+31.32);return fract((p.x+p.y)*p.z);}
#define NOISE 1.
#else
float hsh(vec3 p){return .5;}
#define NOISE 0.
#endif
float vnoise(vec3 x){vec3 i=floor(x),f=fract(x);f=f*f*(3.-2.*f);return mix(mix(mix(hsh(i),hsh(i+vec3(1.,0.,0.)),f.x),mix(hsh(i+vec3(0.,1.,0.)),hsh(i+vec3(1.,1.,0.)),f.x),f.y),mix(mix(hsh(i+vec3(0.,0.,1.)),hsh(i+vec3(1.,0.,1.)),f.x),mix(hsh(i+vec3(0.,1.,1.)),hsh(i+vec3(1.,1.,1.)),f.x),f.y),f.z);}
float fbm(vec3 p){float f=.5*vnoise(p)+.25*vnoise(p*2.03);f+=uQuality>.5?.125*vnoise(p*4.07):.0625;return f/.875;}
float ridge(vec3 p){return NOISE*(1.-abs(2.*vnoise(p)-1.));}
float fissureDist(vec3 p,out float t){float best=1e3;t=.5;for(int i=0;i<FISSURE_N-1;i++){vec3 a=uFissure[i],ab=uFissure[i+1]-a;float h=clamp(dot(p-a,ab)/max(dot(ab,ab),1e-6),0.,1.);float d=distance(p,a+ab*h);if(d<best){best=d;t=(float(i)+h)/float(FISSURE_N-1);}}return best;}
void main(){if(uIsMarker>.5){if(uMarkReveal<=.001)discard;float t=0.;if(uMarkKind==1.)t=(atan(vOriginal.z-uMarkCenter.z,vOriginal.y-uMarkCenter.y)+3.14159265)/6.2831853;else if(uMarkKind==2.){float v=uMarkAxis==2.?vOriginal.z:uMarkAxis==0.?vOriginal.x:vOriginal.y;t=(v-uMarkRange.x)/max(.001,uMarkRange.y-uMarkRange.x);}else if(uMarkKind==3.)t=vIncT;if(t>uMarkReveal)discard;}vec2 q=vec2((vOriginal.y-.3)/1.55,(vOriginal.z-1.25)/1.65);bool window=dot(q,q)<uWindowAction*uWindowAction&&vOriginal.x< -1.9;vec3 cq=(vOriginal-vec3(-.8,-1.0,1.25))/vec3(.46,.55,.67);bool clinoid=dot(cq,cq)<uAction*uAction;vec2 dq=vec2((vOriginal.y-.3)/1.08,(vOriginal.z-1.25)/1.08);vec2 lq=vec2((vOriginal.y+.62)/.38,(vOriginal.z-1.65)/.55);if((uCut==1.||uCut==4.)&&window)discard;if(uCut==7.&&!window)discard;if((uCut==4.||uCut==6.)&&clinoid)discard;if(uCut==3.&&dot(dq,dq)<uAction*uAction)discard;if(uCut==5.&&dot(lq,lq)<uAction*uAction)discard;if(uCut==9.&&distance(vOriginal,uMarkCenter)>uMarkRange.x)discard;if(uSplitKind>.5){bool ins=uSplitKind<1.5?dot(dq,dq)<1.:uSplitKind<2.5?dot(lq,lq)<1.:vSplit>0.;if(uSplitPass<1.5?ins:!ins)discard;}if(uBurrs>0.&&vOriginal.x< -1.9){float a=min(1.,uBurrs*2.)*.255,b=max(0.,uBurrs*2.-1.)*.255;if(distance(vOriginal.yz,vec2(-.7,2.45))<a||distance(vOriginal.yz,vec2(-1.,.8))<b)discard;}
float fisT=.5,fisE=10.,veil=0.;if(uFisVeil>.5&&uCut!=8.){float d=fissureDist(vOriginal,fisT);veil=(1.-smoothstep(.06,.22,d))*smoothstep(0.,.12,fisT)*smoothstep(1.,.88,fisT);}if(uCut==8.){float d=fissureDist(vOriginal,fisT),hw=.18*uAction*smoothstep(0.,.14,fisT)*smoothstep(1.,.86,fisT);fisE=d-hw;if(uAction>0.&&fisE<0.)discard;}
if(uPick>.5){gl_FragColor=uColor;return;}if(uLine>0.){vec3 n=normalize(vNormal),V=normalize(vEye);float nv=abs(dot(n,V));vec3 c=uColor.rgb*(.82+.18*max(0.,dot(n,normalize(vec3(-.35,.55,1.)))))+vec3(.10)*pow(nv,30.);c=mix(uColor.rgb*.45,c,smoothstep(.12,.45,nv));if(uSelected>.5)c=mix(c,vec3(.66,1.,.62),.5);gl_FragColor=vec4(c,1.);return;}
vec3 n=normalize(vNormal),V=normalize(vEye);float nv=dot(n,V),ao=clamp(vAO,0.,1.),alpha=uColor.a,rim=pow(1.-abs(nv),2.);bool tissue=uMaterial>-.5&&uIsMarker<.5,arach=uMaterial>8.5&&uMaterial<9.5,skin=uMaterial>9.5&&uMaterial<10.5;
/* skin slab faces (rest pose): outer skin srad > .45, inner (subcutaneous) < -.45, cut walls between; wall vertex
   normals are tilted +-.34 toward/away from the shell centre at load, so srad runs from the epidermis (+.34) to the
   deep subcutaneous edge (-.34) across a wall. */
float srad=skin?dot(normalize(vObjN),normalize(vOriginal-vec3(0.,-.4,-.4))):1.;
float light=.30+.55*abs(dot(n,normalize(vec3(-.5,.7,1.))))+.12*abs(dot(n,normalize(vec3(.9,-.2,.3))));vec3 c=uColor.rgb*light;
if(tissue&&uSurgical<.5){if(skin&&srad<.45)c=uInner*light;c*=mix(1.,ao,.75);if(arach)alpha=clamp(alpha*(.55+.8*rim),0.,.85);}
if(tissue&&uSurgical>.5){
 bool cutBone=uMaterial<.5&&nv<0.&&!gl_FrontFacing;if(nv<0.)n=-n;nv=abs(nv);
 vec3 P=vOriginal,base=uColor.rgb;float shin=30.,specI=.2,wrap=.3,sss=0.;
 if(uMaterial<.5){base*=.88+.16*fbm(P*38.);shin=14.;specI=.05;wrap=.1;if(cutBone){
  vec3 rd=normalize(P-uCamPos);vec2 o=vec2((uCamPos.y-.3)/1.55,(uCamPos.z-1.25)/1.65),d2=vec2(rd.y/1.55,rd.z/1.65);float A=dot(d2,d2),B=dot(o,d2),Cq=dot(o,o)-uWindowAction*uWindowAction,disc=B*B-A*Cq,h=.1;
  if(disc>0.&&A>1e-6){float t1=(-B+sqrt(disc))/A,tp=distance(P,uCamPos);if(t1<tp)h=(tp-t1)*abs(dot(rd,normalize(vObjN)));}
  bool outerT=dot(vObjN,P-vec3(0.,.3,.6))>0.;float dip=smoothstep(.025,.05,h)*(1.-smoothstep(.13,.17,h));
  vec3 cort=vec3(.86,.82,.74)*(.92+.1*vnoise(P*60.)),dipl=vec3(.74,.53,.47)*(.78+.3*vnoise(P*80.))-.12*smoothstep(.6,.9,vnoise(P*140.));
  base=mix(cort,dipl,dip*(outerT?.85:1.));specI=.06;shin=20.;wrap=.3;sss=.1*dip;}}
 else if(uMaterial<1.5){float m=fbm(P*9.);base*=.91+.13*m;float v=smoothstep(.975,.995,ridge(P*3.+m*1.2))*.12+smoothstep(.985,.997,ridge(P*8.+3.1))*.06;base=mix(base,vec3(.66,.24,.25),v*mix(.6,1.,ao));shin=70.;specI=.42;wrap=.55;sss=.45;}
 else if(uMaterial<2.5){base*=.74+.42*vnoise(vec3(P.x*70.,P.y*3.5,P.z*70.));shin=36.;specI=.28;wrap=.45;sss=.35;}
 else if(uMaterial<3.5){base*=.88+.16*vnoise(vec3(P.x*55.,P.y*6.,P.z*55.));shin=95.;specI=.55;wrap=.35;}
 else if(uMaterial<4.5){float l=fbm(P*9.);base*=.92+.1*smoothstep(.25,.75,l)-.05*smoothstep(.95,.995,ridge(P*9.+l));shin=60.;specI=.3;wrap=.65;sss=.4;}
 else if(uMaterial<5.5){base*=.94+.08*vnoise(P*30.);shin=90.;specI=.55;wrap=.35;sss=.3;}
 else if(uMaterial<6.5){base*=.9+.12*vnoise(P*48.);shin=55.;specI=.32;wrap=.45;sss=.15;}
 else if(uMaterial<7.5){float m=fbm(P*5.);base*=.94+.06*m+.05*vnoise(vec3(P.x*30.,P.y*90.,P.z*30.));base=mix(base,vec3(.64,.67,.74),.22*smoothstep(.35,.75,fbm(P*2.3+3.)));shin=150.;specI=.24;wrap=.35;sss=.1;}
 else if(uMaterial<8.5){base*=.9+.12*vnoise(P*26.);shin=110.;specI=.6;wrap=.3;sss=.2;}
 else if(uMaterial<9.5){shin=130.;specI=.75;wrap=.6;}
 else if(uMaterial<10.5){
  /* skin: mottled warm epidermis with fine pores and a faint sheen; cut edge = pink-white dermis over lobulated
     yellow subcutaneous fat with a thin bloody line at the epidermis; inner face = subcutaneous fat */
  if(srad>.45){float m=fbm(P*6.);base*=.9+.13*m;base=mix(base,base*vec3(1.07,.86,.84),.3*smoothstep(.45,.8,fbm(P*2.3+5.)));base*=1.-.07*NOISE*smoothstep(.68,.9,vnoise(P*95.));shin=24.;specI=.13;wrap=.5;sss=.6;}
  else if(srad>-.45){float f=clamp(.5-srad*1.47,0.,1.);vec3 derm=vec3(.9,.72,.67)*(.93+.1*vnoise(P*120.)),fat=uInner*(.86+.2*fbm(P*45.));base=mix(derm,fat,smoothstep(.5,.62,f));base=mix(base,vec3(.52,.12,.1),.55*(1.-smoothstep(.02,.1,f)));shin=80.;specI=.4;wrap=.5;sss=.35;}
  else{base=uInner*(.86+.2*fbm(P*40.));shin=70.;specI=.35;wrap=.6;sss=.4;}}
 else if(uMaterial<11.5){
  /* galea: glistening white-grey aponeurosis with fibres running front to back and faint small vessels */
  float fib=vnoise(vec3(P.x*55.,P.y*55.,P.z*7.));base*=.9+.12*fib+.04*vnoise(P*30.);base=mix(base,vec3(.7,.42,.42),.1*smoothstep(.986,.998,ridge(P*6.+fib)));shin=140.;specI=.65;wrap=.35;sss=.12;}
 else{
  /* pericranium: thin translucent bluish-white membrane over the outer table (bone tone shows through), fine vessels */
  float m=fbm(P*12.);vec3 bone=vec3(.86,.8,.7)*(.9+.1*m);base=mix(bone,base,.5+.25*smoothstep(.3,.8,fbm(P*3.5+2.)));base=mix(base,vec3(.72,.4,.42),.12*smoothstep(.985,.997,ridge(P*7.+m)));shin=110.;specI=.5;wrap=.4;sss=.15;}
 vec3 alb=base*base,L1=normalize(vec3(-.35,.55,1.));float n1=dot(n,L1),dAO=mix(1.,ao,.75);
 float d1=max(0.,(n1+wrap)/(1.+wrap)),d2=max(0.,(nv+wrap)/(1.+wrap));
 vec3 col=alb*(.50*d1+.36*d2)*dAO;
 col+=sss*alb*vec3(1.,.32,.22)*max(0.,(n1+.9)/1.9-max(0.,n1))*.9*dAO;
 col+=alb*mix(vec3(.07,.06,.06),vec3(.17,.17,.19),.5+.5*n.y)*ao;
 float F=.04+.96*pow(1.-nv,5.),s1=pow(max(0.,dot(n,normalize(L1+V))),shin),s2=pow(nv,shin*2.);
 col+=vec3(1.,.97,.93)*specI*(s1*.9+s2*.7)*(1.+2.*F)*mix(1.,ao,.8);
 vec2 sc=(gl_FragCoord.xy/uViewport-.5)*vec2(uViewport.x/uViewport.y,1.);
 col*=.8*mix(1.1,.66,smoothstep(.15,.9,length(sc)));
 col=(col*(2.51*col+.03))/(col*(2.43*col+.59)+.14);
 c=sqrt(clamp(col,0.,1.));light=clamp(dot(c,vec3(.33)),0.,1.);
 if(arach)alpha=clamp(alpha*(.55+1.3*pow(1.-nv,2.))+specI*(s1+s2)*.5,0.,.92);
}
if(arach&&veil>0.){c=mix(c,vec3(.92,.94,.95),.4*veil);alpha=max(alpha,.62*veil);}if(arach&&fisE<.03){float e=1.-fisE/.03;c=mix(c,vec3(.93,.95,.95),.45*e);alpha=max(alpha,.75*e);}
float edge=10.;if(uCut==1.||uCut==4.)edge=abs(length(q)-uWindowAction);if(uCut==3.)edge=abs(length(dq)-uAction);if(uCut==5.)edge=abs(length(lq)-uAction);if(uCut==4.||uCut==6.)edge=min(edge,abs(length(cq)-uAction));if(uSplitKind==1.)edge=min(edge,abs(length(dq)-1.));if(uSplitKind==2.)edge=min(edge,abs(length(lq)-1.));if(uSplitKind>2.5)edge=min(edge,abs(vSplit)*.6);if(edge<.025)c*=uSurgical>.5?.72:.55;
/* dural C-cut drawn in along the rim before the leaf folds: uGroove = (revealed share, start angle, arc, 1 circle | 2 ellipse) */
float grooveT=-1.;if(uGroove.z>0.){vec2 g=uGroove.w<1.5?dq:lq;float gr=abs(length(g)-1.),gt=mod(atan(g.y,g.x)-uGroove.y,6.2831853)/uGroove.z;grooveT=gt;if(gt<uGroove.x&&gr<.035)c=mix(c,vec3(.3,.05,.05),.8*(1.-smoothstep(.012,.035,gr)));}
float region=0.,hs=0.,od=10.;
if(uChange==1.&&vOriginal.x< -1.9){float r=length(q);region=1.-smoothstep(1.12,1.2,r);hs=.65;od=abs(r-1.)*1.6;}
if(uChange==2.){float r=length(cq);region=1.-smoothstep(1.55,1.7,r);hs=.85;od=abs(r-uAction)*.55;}
if(uChange==3.){float r=length(dq);region=1.-smoothstep(1.12,1.2,r);hs=.65;od=abs(r-uAction)*1.08;}
if(uChange==4.){float r=length(lq);region=1.-smoothstep(1.15,1.25,r);hs=.65;od=abs(r-uAction)*.45;}
/* while the dural C-cut is drawn (uGroove arc > 0) the amber outline of the opening follows it: only the cut part */
if(uGroove.z>0.&&(uChange==3.||uChange==4.)&&grooveT>=uGroove.x)od=10.;
if(uChange==5.){region=1.;hs=.35;}
if(uChange==7.&&vOriginal.x< -1.9){float r=min(distance(vOriginal.yz,vec2(-.7,2.45)),distance(vOriginal.yz,vec2(-1.,.8)));region=1.-smoothstep(.36,.42,r);hs=.8;od=abs(r-.255);}
if(uChange==6.){region=1.-smoothstep(1.,1.25,distance(vOriginal,vec3(-2.1,-.5,1.0)));hs=.5;if(uCut==8.){region=1.;hs=.2;od=abs(fisE);}}
float highlight=region*(hs*(.22+.78*rim)+.75*(1.-smoothstep(.012,.04,od)));
c=mix(c,vec3(1.,.65,.17)*(.6+.4*light),clamp(highlight,0.,.9));if(arach)alpha=max(alpha,.8*highlight);
if(uSelected>.5)c=mix(c,vec3(.66,1.,.62),.50);if(uIsMarker>.5)c=mix(c,uColor.rgb,.8);gl_FragColor=vec4(c,alpha);}`));
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS))
            throw Error(gl.getProgramInfoLog(program));
        gl.useProgram(program);
        ['View', 'Gravity', 'Distance', 'Aspect', 'Scale', 'Pan', 'Color', 'Selected', 'Pick', 'Region', 'Target', 'Cut', 'Motion', 'Action', 'Offset', 'Change', 'WindowAction', 'FlapAngle', 'Side', 'Burrs', 'IsMarker', 'Line', 'LinePx', 'LineNudge', 'MarkKind', 'MarkAxis', 'MarkReveal', 'MarkCenter', 'MarkRange', 'Material', 'Surgical', 'Quality', 'Viewport', 'Fissure', 'FlapAxis', 'FlapPivot', 'CamPos', 'FisVeil', 'CurlOn', 'CurlIA', 'CurlPivot', 'CurlAxis', 'CurlAway', 'CurlLift', 'CurlP', 'CurlRow', 'DrapeW', 'DrapeP', 'Drape', 'PartOn', 'Part', 'Inc', 'IncReveal', 'SplitKind', 'SplitPlane', 'SplitInc', 'SplitPass', 'Groove', 'Inner'].forEach(n => uniforms[n] = gl.getUniformLocation(program, 'u' + n));
        let vertexCount = 0;
        meshes.forEach((m, i) => { m.index = i; m.hidden = false; m.first = vertexCount; vertexCount += m.count; m.rgb = m.color.match(/\w\w/g).map(x => parseInt(x, 16) / 255); m.innerRgb = m.innerColor ? hexRgb(m.innerColor) : null; m.surgicalTint = /^ILL-sulcal-backing/.test(m.id) ? surgicalRgb.brain.map(x => x * .62) : surgicalLayerTints[baseId(m)] ? hexRgb(surgicalLayerTints[baseId(m)]) : null; });
        buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertexCount * 24, gl.STATIC_DRAW);
        meshes.filter(m => !m.contextOnly).forEach(ensureGeometry);
        for (const [name, start] of [['aPosition', 0], ['aNormal', 12]]) {
            const loc = gl.getAttribLocation(program, name);
            gl.enableVertexAttribArray(loc);
            gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 24, start);
        }
        available = true;
        window.guideQA = { solidDepth: true, zoomBy, syncMobileLayout, alignActiveStep, updateMobileReaderHeight, changeKind, procedureLabelRegion, activeLabelMesh, fixedLabelRect, visible, indicationGroups, chooseIndication, setProcedureSide, operativeSide, sideSign, canonicalPoint, onOperativeSide, relevantMesh, stepView, baseId, meshSide, meshOffset, scanPickAnchors, pickIdAt, meshCount: meshes.length, vertexCount, draw, activate, choose, configs, select, setTab, setView, enabled, allEnabled, getMeshes: () => meshes.map(({ data, vertexData, procedureCandidates, fissureCache, ...m }) => m), visibleCount: () => meshes.filter(visible).length, getSteps: () => steps, getState: () => ({ palette, viewZoom, isMobile, config: { ...config }, model: modelIndex() + 1, stage, mode, selected, contextAlpha, contextSide, prelabelsOn, opening, beforeCurrent, regionalContext, instrumentsOn, separateLayers, ...cameraState(), distance, pan: [...pan], target: [...target], enabled: { ...enabled }, allEnabled: { ...allEnabled } }), surgicalState, setPalette, opacity, refreshLabels: updateStepLabels,
            // Camera / head-pose API (see the orientation and head-positioning blocks near the top of this file).
            getCamera: () => ({ q: [...viewQ], frame: camFrame, target: [...target], distance, pan: [...pan], viewZoom }), setCamera, rotateByDrag: (dx, dy) => { stopGlide(); rotateByDrag(dx, dy); render(); }, rollView: a => { stopGlide(); rollView(a); render(); }, attachCamera: f => { attachCamera(f); render(); }, viewBasis: () => ({ M: [...viewBasis().M], target: [...viewBasis().target] }), project, transform, pickAt: (x, y) => { draw(true); cacheLabelVisibility(); const id = pickIdAt(x, y); draw(); return id; }, setSurgeonView,
            getPose: poseState, setPose, setPoseMode, resetPose, gravityInModel, posePoint, poseInverse, posePivot: [...POSE_PIVOT], poseDefault: { ...POSE_DEFAULT },
            // Soft-tissue deformation (curl / peel, incision parting, two-pass regions); see the curl block above.
            stateProgress, movePoint: (i, p) => movePoint(meshes[i], p), getDeformation: i => { const d = deformation(meshes[i]); return d && { curl: d.curl && { key: d.curl.mo.key, progress: d.curl.P, hingeTravel: d.curl.dh, angle: d.curl.T, radius: d.curl.mo.R, dmax: d.curl.mo.dmax }, split: d.split && { kind: d.split.kind, margin: d.split.margin, plane: d.split.plane }, part: d.part && { reveal: d.part.reveal, gap: d.part.gap, sign: d.part.sign }, groove: d.groove, leafCut: !!d.leafCut }; }, curlTuning: CURL_TUNING, variantSnapshot };
        $('#loading').hidden = true;
        render();
    }
    let pickTarget = null, fissureN = 12;
    function fissureUniform(m) { if (m.fissureCache?.n !== fissureN) {
        // Resample the record's fissure polyline to fissureN points at equal arc length.
        const pts = m.fissureLine, lens = [0];
        for (let i = 1; i < pts.length; i++)
            lens.push(lens[i - 1] + Math.hypot(...pts[i].map((x, j) => x - pts[i - 1][j])));
        const total = lens[lens.length - 1] || 1, out = [];
        for (let k = 0, i = 1; k < fissureN; k++) {
            const d = total * k / (fissureN - 1);
            while (i < pts.length - 1 && lens[i] < d)
                i++;
            const f = Math.max(0, Math.min(1, (d - lens[i - 1]) / ((lens[i] - lens[i - 1]) || 1)));
            out.push(pts[i - 1].map((x, j) => x + (pts[i][j] - x) * f));
        }
        m.fissureCache = { n: fissureN, points: out };
    } return m.fissureCache.points.flatMap(canonicalPoint); }
    function preparePickTarget(w, h) { if (pickTarget && pickTarget.w === w && pickTarget.h === h)
        return; if (pickTarget) {
        gl.deleteFramebuffer(pickTarget.fbo);
        gl.deleteTexture(pickTarget.texture);
        gl.deleteRenderbuffer(pickTarget.depth);
    } const fbo = gl.createFramebuffer(), texture = gl.createTexture(), depth = gl.createRenderbuffer(); gl.bindTexture(gl.TEXTURE_2D, texture); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null); gl.bindRenderbuffer(gl.RENDERBUFFER, depth); gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0); gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth); if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
        throw Error('Selection buffer is unavailable'); pickTarget = { fbo, texture, depth, w, h }; gl.bindTexture(gl.TEXTURE_2D, null); gl.bindRenderbuffer(gl.RENDERBUFFER, null); }
    // Deformation uniforms (see the vertex shader): curl / peel, incision parting and two-pass regions.
    function setCurlUniforms(cs) { gl.uniform1f(uniforms.CurlOn, cs ? 1 : 0); if (!cs)
        return; const mo = cs.mo; gl.uniform3fv(uniforms.CurlPivot, mo.pivot); gl.uniform3fv(uniforms.CurlAxis, mo.axis); gl.uniform3fv(uniforms.CurlAway, mo.away); gl.uniform3fv(uniforms.CurlLift, mo.lift); gl.uniform4f(uniforms.CurlP, mo.R, cs.T, cs.dh, mo.prof.a0); gl.uniform1f(uniforms.CurlIA, mo.prof.ia); gl.uniform1fv(uniforms.CurlRow, cs.row); const dr = cs.wD > 0 && mo.drape; gl.uniform1f(uniforms.DrapeW, dr ? cs.wD : 0); if (dr) {
        gl.uniform4f(uniforms.DrapeP, dr.a0, dr.ia, dr.ie, dr.RT);
        gl.uniform1fv(uniforms.Drape, dr.tab);
    } }
    function setPartUniforms(pt, sp) { gl.uniform1f(uniforms.PartOn, pt ? 1 : 0); const inc = pt?.inc || (sp?.kind === 3 ? sp.inc : null); if (inc)
        gl.uniform3fv(uniforms.Inc, inc.flat); if (pt)
        gl.uniform4f(uniforms.Part, pt.reveal, pt.gap, pt.sign, PART_FALLOFF); }
    // pass 0: none; 1: stationary (discard inside the region); 2: moving part (discard outside).
    function setSplitUniforms(sp, pass) { gl.uniform1f(uniforms.SplitKind, sp && pass ? sp.kind : 0); gl.uniform1f(uniforms.SplitPass, pass); if (sp?.kind === 3) {
        gl.uniform2f(uniforms.SplitInc, sp.inc.sign, sp.margin);
        gl.uniform4fv(uniforms.SplitPlane, sp.plane || [0, 0, 0, -1e4]);
    } }
    function boneFlapTravel(state) {
        if (beforeCurrent || state.cut !== 1 && state.cut !== 4 || steps[stage].change !== 'bone')
            return null;
        const progress = state.action;
        if (progress >= 1)
            return null;
        // Lift from the inferior edge first, then carry the freed piece laterally.
        const lift = Math.min(1, progress / .38);
        const carry = Math.max(0, (progress - .38) / .62);
        const ease = t => t * t * (3 - 2 * t);
        return {
            angle: sideSign() * .31 * ease(lift),
            offset: [-sideSign() * (.32 * ease(lift) + 3.8 * ease(carry)), 2.6 * ease(carry), -.8 * ease(carry)]
        };
    }
    // Largest drawable size (an oversized layout, e.g. a very wide page, must not break the selection buffer).
    let drawLimitCache = 0;
    function drawLimit() { if (!drawLimitCache) {
        const vp = gl.getParameter(gl.MAX_VIEWPORT_DIMS);
        drawLimitCache = Math.max(256, Math.min(gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) || 4096, gl.getParameter(gl.MAX_TEXTURE_SIZE) || 4096, vp ? Math.min(vp[0], vp[1]) : 4096));
    } return drawLimitCache; }
    function draw(picking = false) {
        if (!available)
            return;
        updateGlide();
        updatePose();
        const ratio = picking ? 1 : Math.min(devicePixelRatio || 1, interactionActive ? 1 : 2), maxDim = drawLimit(), w = Math.max(1, Math.min(maxDim, Math.round(host.clientWidth * ratio))), h = Math.max(1, Math.min(maxDim, Math.round(host.clientHeight * ratio)));
        if (!picking && (canvas.width !== w || canvas.height !== h)) {
            canvas.width = w;
            canvas.height = h;
        }
        aspect = w / h;
        if (picking)
            preparePickTarget(w, h);
        gl.bindFramebuffer(gl.FRAMEBUFFER, picking ? pickTarget.fbo : null);
        gl.viewport(0, 0, w, h);
        gl.clearColor(picking ? 0 : .058, picking ? 0 : .093, picking ? 0 : .12, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.enable(gl.DEPTH_TEST);
        gl.disable(gl.CULL_FACE);
        gl.disable(gl.DITHER);
        gl.uniformMatrix3fv(uniforms.View, false, mT(viewBasis().M)); // GLSL mat3 is column-major
        gl.uniform3fv(uniforms.Gravity, gravityInModel());
        gl.uniform1f(uniforms.Distance, distance);
        gl.uniform1f(uniforms.Aspect, aspect);
        gl.uniform1f(uniforms.Scale, scale * viewZoom); gl.uniform1f(uniforms.LinePx, 2 * LINE_MIN_HALF_PX / (scale * viewZoom * gl.drawingBufferHeight)); gl.uniform1f(uniforms.LineNudge, LINE_NUDGE);
        gl.uniform2fv(uniforms.Pan, pan);
        gl.uniform3fv(uniforms.Target, viewBasis().target);
        gl.uniform1f(uniforms.Pick, picking ? 1 : 0);
        gl.uniform1f(uniforms.Surgical, palette === 'surgical' ? 1 : 0);
        gl.uniform1f(uniforms.Quality, isMobile || interactionActive ? 0 : 1);
        gl.uniform2f(uniforms.Viewport, w, h);
        gl.uniform3fv(uniforms.CamPos, cameraPosition());
        // Opaque meshes first; translucent ones afterwards, back to front, without depth writes. In the picking
        // pass translucent meshes are drawn after the opaque ones with the depth test, so a click identifies the
        // nearest surface; the arachnoid veil is left out (pickThrough) so clicks reach the cortex and vessels.
        const visibleList = meshes.filter(visible), opaque = visibleList.filter(m => opacity(m) >= 1), depth = m => transform(movePoint(m, m.anchor))[2];
        const translucent = visibleList.filter(m => opacity(m) < 1 && opacity(m) > 0 && !(picking && m.group === 'arachnoid')).map(m => [depth(m), m]).sort((a, b) => a[0] - b[0]).map(x => x[1]);
        const list = [...opaque, ...translucent];
        gl.disable(gl.BLEND);
        gl.depthFunc(gl.LEQUAL);
        gl.depthMask(true);
        drawProps(picking); // schematic head holder and table (room views only); restores the anatomy buffer
        for (const m of list) {
            ensureGeometry(m);
            const flat = m.marker || m.instrument || !(m.group in materialIds), see = !picking && opacity(m) < 1;
            // The illustrative sulcal backing gets a darker, recessed cortex tone in the Surgical palette too.
            let rgb = palette === 'surgical' && !flat ? (m.surgicalTint || surgicalRgb[m.group]) : m.rgb;
            if (see) {
                gl.enable(gl.BLEND);
                gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
            }
            gl.uniform1f(uniforms.Material, flat ? -1 : materialIds[m.group]);
            if (picking) {
                const id = m.index + 1;
                rgb = [(id & 255) / 255, ((id >> 8) & 255) / 255, ((id >> 16) & 255) / 255];
            }
            const state = surgicalState(m);
            if (state.cut === 8)
                gl.uniform3fv(uniforms.Fissure, fissureUniform(m));
            gl.uniform1f(uniforms.Side, sideSign());
            gl.uniform1f(uniforms.Burrs, state.burrAction);
            gl.uniform1f(uniforms.IsMarker, m.marker ? 1 : 0); gl.uniform1f(uniforms.Line, m.marker || m.instrument ? (m.markerRadius || .035) : 0);
            const incMark = m.marker ? incisionMarker(m) : null;
            gl.uniform1f(uniforms.MarkKind, incMark ? 3 : m.markerKind || 0);
            gl.uniform1f(uniforms.MarkAxis, m.markerAxis || 1);
            gl.uniform1f(uniforms.MarkReveal, m.marker ? markReveal(m) : 1);
            gl.uniform3fv(uniforms.MarkCenter, canonicalPoint(m.markerCenter || [0, 0, 0]));
            gl.uniform2fv(uniforms.MarkRange, m.markerRange || [0, 1]);
            gl.uniform1f(uniforms.WindowAction, state.windowAction);
            gl.uniform1f(uniforms.Change, changeKind(m));
            gl.uniform3fv(uniforms.Offset, meshOffset(m));
            gl.uniform1f(uniforms.Cut, state.cut);
            gl.uniform1f(uniforms.Motion, state.motion);
            gl.uniform1f(uniforms.Action, state.action);
            gl.depthMask(!see);
            gl.uniform4fv(uniforms.Color, [...rgb, picking ? 1 : opacity(m)]);
            gl.uniform1f(uniforms.Selected, m.index === selected ? 1 : 0);
            gl.uniform1f(uniforms.Region, procedure && m.name === steps[stage].focus ? 1 : 0);
            gl.uniform1f(uniforms.FlapAngle, 0);
            gl.uniform3fv(uniforms.Inner, m.innerRgb || m.rgb);
            const def = deformation(m);
            setPartUniforms(def?.part || null, def?.split || null);
            gl.uniform1f(uniforms.IncReveal, incMark ? 1 : 0);
            if (incMark)
                gl.uniform3fv(uniforms.Inc, incMark.flat);
            gl.uniform4fv(uniforms.Groove, def?.groove || [0, 0, 0, 1]);
            gl.uniform1f(uniforms.FisVeil, m.group === 'arachnoid' && m.fissureLine?.length > 1 ? 1 : 0);
            if (m.group === 'arachnoid' && m.fissureLine?.length > 1 && state.cut !== 8)
                gl.uniform3fv(uniforms.Fissure, fissureUniform(m));
            if (state.keep) {
                gl.uniform3fv(uniforms.MarkCenter, canonicalPoint(state.keep.center));
                gl.uniform2fv(uniforms.MarkRange, [state.keep.radius, 0]);
            }
            if (def?.split && (def.curl || def.leafCut)) {
                // Two passes with complementary rest-pose regions: the stationary tissue, then the moving part
                // (curled), so the cut edge is exact in both.
                setSplitUniforms(def.split, 1);
                setCurlUniforms(null);
                gl.drawArrays(gl.TRIANGLES, m.first, m.count);
                setSplitUniforms(def.split, 2);
                setCurlUniforms(def.curl);
                if (def.leafCut)
                    gl.uniform1f(uniforms.Cut, 0);
                gl.drawArrays(gl.TRIANGLES, m.first, m.count);
                gl.uniform1f(uniforms.Cut, state.cut);
            }
            else {
                setSplitUniforms(null, 0);
                setCurlUniforms(def?.curl || null);
                gl.drawArrays(gl.TRIANGLES, m.first, m.count);
            }
            const flap = boneFlapTravel(state);
            if (flap) {
                setSplitUniforms(null, 0);
                setCurlUniforms(null);
                setPartUniforms(null, null);
                gl.uniform1f(uniforms.Cut, 7);
                gl.uniform1f(uniforms.FlapAngle, flap.angle);
                gl.uniform3fv(uniforms.FlapAxis, [0, 0, 1]);
                gl.uniform3fv(uniforms.FlapPivot, [-2.4 * sideSign(), -1.25, 1.25]);
                gl.uniform3fv(uniforms.Offset, flap.offset);
                gl.uniform1f(uniforms.Change, 0);
                gl.drawArrays(gl.TRIANGLES, m.first, m.count);
            }
        }
        gl.disable(gl.BLEND);
        gl.depthMask(true);
        if (!picking && !capturing) {
            updateStepLabels();
            updateGizmo();
            syncPoseControls();
        }
    }
    function render(refreshPicking = true) { if (interactionActive)
        pickDirty = false;
    else if (refreshPicking)
        pickDirty = true; if (scheduled)
        return; scheduled = true; requestAnimationFrame(now => { scheduled = false; poseClock = now; if (mode === '3d') {
        if (pickDirty) { pickDirty = false; draw(true); cacheLabelVisibility(); }
        draw();
    } }); }
    // Plain one-pointer dragging orbits around the head on both axes while keeping the horizon level. Alt/Option drag
    // rolls the camera about its line of sight. Shift/right-drag pans; two pointers pan, pinch and twist.
    let pointers = new Map(), last = null, moved = 0, dragging = false, pinch = 0, hadMulti = false, pinchCenter = null, twist = null;
    const pairAngle = p => Math.atan2(p[1][1] - p[0][1], p[1][0] - p[0][0]), wrapAngle = a => Math.atan2(Math.sin(a), Math.cos(a));
    canvas.onpointerdown = e => { stopGlide(false); try {
        canvas.setPointerCapture(e.pointerId);
    }
    catch (error) { } /* synthetic / already-released pointers cannot be captured */ if (pointers.size === 0) {
        hadMulti = false;
        dragging = false;
        moved = 0;
    } pointers.set(e.pointerId, [e.clientX, e.clientY]); if (pointers.size > 1) {
        hadMulti = true;
        dragging = true;
    } last = [e.clientX, e.clientY]; if (pointers.size === 2) {
        const p = [...pointers.values()];
        pinch = Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]);
        pinchCenter = [(p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2];
        twist = pairAngle(p);
    } };
    canvas.onpointermove = e => { if (!pointers.has(e.pointerId))
        return; if (!last) {
        last = [e.clientX, e.clientY];
        return;
    } const dx = e.clientX - last[0], dy = e.clientY - last[1]; moved += Math.abs(dx) + Math.abs(dy); if (dx || dy)
        stopGlide(moved >= 5 || pointers.size > 1); pointers.set(e.pointerId, [e.clientX, e.clientY]); if (pointers.size === 1 && !dragging && moved < 5) {
        last = [e.clientX, e.clientY];
        return;
    } dragging = true; if (pointers.size === 2) {
        const p = [...pointers.values()], d = Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]);
        markInteraction();
        if (pinch > 0 && d > 0)
            zoomBy(d / pinch, false);
        const center = [(p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2], unit = 2 * distance / (scale * viewZoom * Math.max(1, host.clientHeight));
        if (pinchCenter) {
            pan[0] += (center[0] - pinchCenter[0]) * unit;
            pan[1] -= (center[1] - pinchCenter[1]) * unit;
        }
        pinchCenter = center;
        pinch = d;
        const angle = pairAngle(p);
        if (twist !== null)
            rollView(-wrapAngle(angle - twist));
        twist = angle;
    }
    else if (e.shiftKey || e.buttons === 2) {
        markInteraction();
        const unit = 2 * distance / (scale * viewZoom * Math.max(1, host.clientHeight));
        pan[0] += dx * unit;
        pan[1] -= dy * unit;
    }
    else if (e.altKey) {
        const r = canvas.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
        const angle = (x, y) => Math.atan2(y - cy, x - cx);
        markInteraction(); rollView(-wrapAngle(angle(e.clientX, e.clientY) - angle(last[0], last[1])));
    }
    else {
        markInteraction(); orbitByDrag(dx, dy);
    } last = [e.clientX, e.clientY]; render(false); };
    canvas.onpointerup = e => { pointers.delete(e.pointerId); twist = null; if (moved < 5 && available && !hadMulti) {
        draw(true);
        cacheLabelVisibility();
        const r = canvas.getBoundingClientRect(), id = pickIdAt(e.clientX - r.left, e.clientY - r.top);
        select(id);
        render();
    } last = null; };
    canvas.onpointercancel = e => { pointers.delete(e.pointerId); last = null; twist = null; };
    canvas.oncontextmenu = e => e.preventDefault();
    canvas.addEventListener('wheel', e => { e.preventDefault(); stopGlide(false); markInteraction(); const delta = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? host.clientHeight : 1); zoomBy(Math.exp(-delta * .0015), false); }, { passive: false });
    canvas.onkeydown = e => { if (e.key === 'Escape') {
        clearSelection();
        return;
    } if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '-', '=', 'Home', 'q', 'Q', 'e', 'E'].includes(e.key)) {
        e.preventDefault();
        stopGlide();
        // Arrows turn about the screen axes (same steps as before, now unclamped); Q/E roll.
        if (e.key === 'ArrowLeft')
            rotateByDrag(-.13, 0, 1);
        if (e.key === 'ArrowRight')
            rotateByDrag(.13, 0, 1);
        if (e.key === 'ArrowUp')
            rotateByDrag(0, .1, 1);
        if (e.key === 'ArrowDown')
            rotateByDrag(0, -.1, 1);
        if (e.key === 'q' || e.key === 'Q')
            rollView(.1);
        if (e.key === 'e' || e.key === 'E')
            rollView(-.1);
        if (['+', '='].includes(e.key))
            zoomBy(1.15);
        if (e.key === '-')
            zoomBy(1 / 1.15);
        if (e.key === 'Home')
            $('#resetView').click();
        render();
    } };
    canvas.addEventListener('webglcontextlost', e => { e.preventDefault(); available = false; $('#noGL').hidden = false; });
    canvas.addEventListener('webglcontextrestored', () => location.reload());
    new ResizeObserver(() => { updateMobilePanelHeight(); render(); }).observe(host);
    rebuild({ reset: true });
    applyRecommendedView();
    requestAnimationFrame(() => { try {
        initGL();
    }
    catch (error) {
        $('#loading').hidden = true;
        $('#noGL').hidden = false;
        $('#noGL').textContent = 'The 3D renderer could not start: ' + error.message + '. The paper library and reading sequence remain available.';
        console.error(error);
    } });
})();
