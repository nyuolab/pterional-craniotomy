(() => {
    'use strict';
    const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
    const { meshes, manifest } = window.__PTERIONAL_DATA__;
    const papers = manifest.papers;
    const escape = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const mobileQuery = matchMedia('(max-width:760px)');
    let isMobile = mobileQuery.matches, desktopLabelPreference = true, mobileLabelPreference = false, viewZoom = 1;
    let config = { clinoid: 'none', muscle: 'subfascial', fissure: 'yes' }, steps = [], stage = 0, mode = '3d', selected = -1, contextAlpha = 1, contextSide = 'right', procedure = true, prelabelsOn = !isMobile, opening = 1, animationToken = 0, separateLayers = false, instrumentsOn = true, capturing = false, beforeCurrent = false, regionalContext = true, contextLayerSnapshot = null;
    const enabled = { bone: true, brain: true, muscle: true, fascia: true, deepfascia: true, fat: true, arteries: true, nerves: true, dura: true };
    const allEnabled = Object.fromEntries(Object.keys(enabled).map(k => [k, false]));
    const canvas = $('#canvas'), host = $('#viewport');
    let gl = null, program = null, uniforms = {}, buffer = null, available = false, yaw = 1.0, pitch = .25, screenOrientation = 1, distance = 15, pan = [0, 0], target = [-1, 0, .5], aspect = 1, scheduled = false;
    const scale = 2.05;
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
    function meshSide(m) { if (m.procedureSide)
        return m.procedureSide; if (m.nativeName.endsWith('.l'))
        return 'left'; if (m.nativeName.endsWith('.r'))
        return 'right'; return 'both'; }
    function onOperativeSide(m) { const side = meshSide(m); return side === 'both' || side === operativeSide(); }
    function baseId(m) { return m.baseId || m.id; }
    function relevantMesh(m) { if (m.contextOnly)
        return false; const s = steps[stage]; if (m.group === 'bone' && s?.isolateBone && !s.isolateBone.includes(m.nativeName))
        return false; return true; }
    function stepView(s) { const sign = sideSign(); return { ...s.view, yaw: s.view.yaw * sign, target: [s.view.target[0] * sign, ...s.view.target.slice(1)] }; }
    function setProcedureSide(side) { if (!['right', 'left', 'both'].includes(side))
        return; contextSide = side; clearSelection(); syncControls(); updateChangeControls(); if (mode === '2d')
        showDiagram();
    else
        render(); }
    function toggleFullContext() { const expanded = Object.keys(enabled).filter(k => enabled[k]).every(k => allEnabled[k]); if (!expanded) {
        contextLayerSnapshot = { ...allEnabled };
        for (const k in allEnabled)
            allEnabled[k] = enabled[k];
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
    function zoomBy(factor) { if (!Number.isFinite(factor) || factor <= 0)
        return; viewZoom = Math.max(.25, Math.min(16, viewZoom * factor)); render(); }
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
    function setMode(next) { animationToken++; mode = next; $('#mode3d').setAttribute('aria-pressed', mode === '3d'); $('#mode2d').setAttribute('aria-pressed', mode === '2d'); for (const s of ['#canvas', '#viewinfo', '.floating', '#gesturehint', '.viewtools', '#mobileModelDock', '#stepLabels', '#labelLines'])
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
    function hasStageChange() { const s = steps[stage]; return !!s && (!!s.change && s.change !== 'fissure' && (s.change !== 'muscle' || config.muscle === 'subfascial') || meshes.some(m => m.marker && m.markerStages.includes(s.id))); }
    function updateChangeControls() { const s = steps[stage]; if (!s)
        return; $('#regionalContext').hidden = false; $('#regionalContext').textContent = 'Full Head Context'; const keys = Object.keys(enabled).filter(k => enabled[k]), expanded = keys.length > 0 && keys.every(k => allEnabled[k]), partial = Object.values(allEnabled).some(Boolean); regionalContext = !expanded; $('#regionalContext').setAttribute('aria-pressed', expanded ? 'true' : partial ? 'mixed' : 'false'); $('#viewlabel').textContent = 'Isolated Skull Base'; $('#viewlabel').hidden = !(s.isolateBone && !allEnabled.bone); $('#sideCaption').textContent = contextSide === 'both' ? 'Both Sides · Right Procedure' : (operativeSide() === 'left' ? 'Left' : 'Right') + '-Sided Procedure'; $('#focusChange').hidden = mode !== '3d'; $('#compareChange').hidden = mode !== '3d' || !hasStageChange(); const tools = meshes.filter(m => m.instrument && visible(m)); $('#visualnote').textContent = (tools.length ? 'Cyan: instrument · ' : '') + 'Colored markings are schematic · ' + (operativeSide() === 'left' ? 'Left' : 'Right') + ' procedure'; $('#operativeView').hidden = mode !== '3d'; }
    function changeKind(m) { if (!procedure || m.instrument || m.marker || !onOperativeSide(m))
        return 0; const s = steps[stage], id = s.change; if (id === 'clinoid' && m.nativeName === 'Sphenoid bone')
        return 2; if (id === 'burr' && m.group === 'bone')
        return 7; if (id === 'bone' && m.group === 'bone')
        return 1; if (id === 'dura' && m.group === 'dura')
        return config.fissure === 'no' ? 4 : 3; if (id === 'muscle' && ['muscle', 'fascia', 'deepfascia', 'fat'].includes(m.group) && !m.contextOnly)
        return 5; if (id === 'fissure' && ['brain', 'arteries'].includes(m.group) && !m.contextOnly)
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
        return manifest.study_plates || []; const saved = { yaw, pitch, screenOrientation, distance, viewZoom, target: [...target], pan: [...pan], selected, opening, procedure, contextSide, separateLayers, enabled: { ...enabled }, allEnabled: { ...allEnabled }, beforeCurrent, regionalContext, hidden: meshes.map(m => m.hidden) }; const s = steps[stage], vside = stepView(s), images = []; capturing = true; try {
        viewZoom = 1;
        regionalContext = saved.regionalContext;
        selected = -1;
        opening = 1;
        procedure = true;
        for (const k in enabled)
            enabled[k] = persistentLayers.has(k) || s.groups.includes(k);
        meshes.forEach(m => m.hidden = false);
        const views = hasStageChange() ? [{ title: 'Before this step', before: true, yaw: vside.yaw, pitch: vside.pitch, distance: vside.distance, target: vside.target }, { title: 'After this step', before: false, yaw: vside.yaw, pitch: vside.pitch, distance: vside.distance, target: vside.target }] : [{ title: 'Approach view', yaw: vside.yaw, pitch: vside.pitch, distance: vside.distance, target: vside.target }, { title: s.id === 'muscle' ? 'Fascial planes' : 'Superior oblique view', yaw: (s.id === 'muscle' ? .72 : .8) * sideSign(), pitch: s.id === 'muscle' ? .12 : 1.02, distance: s.id === 'muscle' ? 15 : s.view.distance + 1, target: vside.target }];
        for (const v of views) {
            beforeCurrent = v.before ?? saved.beforeCurrent;
            yaw = v.yaw;
            pitch = v.pitch;
            screenOrientation = 1;
            distance = v.distance;
            target = [...v.target];
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
        yaw = saved.yaw;
        pitch = saved.pitch;
        screenOrientation = saved.screenOrientation;
        distance = saved.distance;
        viewZoom = saved.viewZoom;
        target = saved.target;
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
        let state = { bone: false, dura: false, clinoid: false, muscle: false, fissure: false, burr: false };
        const list = [];
        const surface = ['bone', 'muscle', 'fascia', 'deepfascia', 'fat'], cranial = ['bone', 'brain', 'dura', 'arteries', 'nerves'];
        const add = (id, title, body, options = {}) => { const beforeState = { ...state }; if (options.change)
            Object.assign(state, options.change); const sources = options.sources || ['anatomy']; list.push({ id, title, body, note: options.note || '', sources, groups: options.groups || cranial, labels: options.labels || ['Frontal bone'], state: { ...state }, beforeState, change: options.change ? Object.keys(options.change)[0] : null, visualAction: options.visualAction || id, markings: meshes.filter(m => m.marker && m.markerStages.includes(id) && m.procedureSide === 'right').map(m => m.id), isolateBone: options.isolateBone || null, view: { yaw: 1.0, pitch: .28, distance: 14, target: [-1, 0, .7], alpha: 1, ...options.view }, evidence: sources.map(paper => ({ paper, section: options.sections?.[paper] || papers.find(p => p.id === paper).section, origin: options.origin || 'Technique synthesis; preparation and closure also follow the procedural outline supplied for this guide.' })), figure: null, page: 1 }); };
        const clinoidView = { distance: 7, target: [-.8, -1.1, 1.35], pitch: 1.25, yaw: .6 };
        add('orient', 'Identify the pterion and target', 'Identify the meeting point of the frontal, parietal, temporal and sphenoid bones and relate it to the planned intracranial target. This establishes the frontotemporal route and the amount of frontal, temporal or basal exposure that will be needed.', { groups: ['bone', 'brain', 'arteries', 'nerves'], view: { distance: 17 } });
        add('position', 'Position and immobilize the head', 'Position supine with the head elevated for venous drainage and secured in a Mayfield or Sugita head frame. Rotate away from the operative side and tailor extension to the target so gravity can relax the frontal lobe without allowing the orbital roof to obscure the optic–carotid corridor.', { sources: ['anatomy', 'general'], groups: ['bone', 'brain'], note: 'Rodriguez Rubio et al. describe about 15–20° contralateral rotation and 20° extension in their example, with target-dependent adjustment. Operative View uses the field orientation in Figures 10–11; it does not simulate the head holder or patient position.', view: { distance: 17 }, sections: { anatomy: 'PDF pp.6–7, Positioning', general: 'Positioning' } });
        add('prepare', 'Prepare and mark the incision', 'Clip hair as needed, prepare the skin and mark a curving frontotemporal incision behind the hairline. Infiltrate the marked line with local anesthetic and epinephrine as appropriate to reduce incisional pain and scalp bleeding.', { sources: ['general', 'radiopaedia'], groups: surface, note: 'Preparation, skin and medication delivery are described, not modeled.', sections: { general: 'Approach context only; preparation and infiltration details come from the supplied procedural outline' }, origin: 'Preparation details follow the procedural outline supplied for this guide; they are not attributed to a measured atlas or to a specific passage in the linked review.' });
        add('sterile', 'Drape the operative field', 'Apply sterile drapes around the prepared operative area while keeping the marked incision and required landmarks accessible. This isolates the field before the skin is opened.', { sources: ['general', 'radiopaedia'], groups: surface, note: 'Drapes and the sterile field are not modeled.', sections: { general: 'Approach context only; draping detail comes from the supplied procedural outline' }, origin: 'Draping step follows the procedural outline supplied for this guide; the linked review provides general approach context.' });
        add('incision', 'Make the curving scalp incision', 'Begin at the superior border of the zygomatic arch about 1 cm anterior to the tragus and curve toward the midline, preferably behind the hairline. Keep the superficial temporal artery and auriculotemporal region in view while opening the scalp so the flap retains its blood supply.', { groups: surface, note: 'The incision trajectory is marked on the outer reference surfaces; an actual scalp segmentation is not reconstructed.', sections: { anatomy: 'PDF p.7, Incision of skin; p.8, Figure 3' } });
        if (config.muscle === 'subfascial')
            add('scalp-flap', 'Mobilize the scalp flap', 'Elevate the scalp in the loose tissue plane, exposing pericranium above the superior temporal line and temporal fascia below it. Keep the superficial temporal artery with the flap and protect the temporal facial nerve branches to preserve flap perfusion and forehead movement.', { groups: surface, labels: ['Right superficial fascial lamina — illustration'], sections: { anatomy: 'PDF pp.8–10, Dissection of skin flap; Figures 3–4' } });
        else
            add('scalp-flap', 'Preserve the myocutaneous layers', 'Preserve continuity between scalp and temporal fascia while defining the edges of the planned myocutaneous flap. Protect the superficial temporal vessels and temporal facial nerve branches so the scalp and muscle can subsequently be mobilized together.', { sources: ['muscles'], groups: surface, labels: ['Right temporalis muscle'], sections: { muscles: 'PDF p.2, Methods: scalp and temporalis reflected together without separating scalp from temporal fascia' }, note: 'No preliminary separate scalp–fascial dissection is included in this branch; scalp is not modeled.' });
        if (config.muscle === 'subfascial')
            add('muscle', 'Develop the subfascial plane', 'Incise both temporal fascial laminae and elevate them together, keeping the interfascial fat within the superficial flap. Separate this fascial envelope from temporalis so the facial nerve branches remain protected in the superficial tissues.', { sources: ['anatomy', 'subfascial'], groups: surface, labels: ['Right superficial fascial lamina — illustration'], change: { muscle: true }, note: 'Fascial laminae and fat pads are original illustrations; displacement exaggerates the plane for visibility.' });
        else
            add('muscle', 'Mobilize the myocutaneous flap', 'Mobilize the scalp and temporalis together as a myocutaneous flap while protecting the superficial temporal vessels and temporal facial nerve branches. Keeping these tissues together exposes the pterion without opening a separate scalp–fascial plane.', { sources: ['muscles'], groups: surface, labels: ['Right temporalis muscle'], change: { muscle: true }, note: 'Muscle-splitting terminology is retained from the comparative paper; an exact intramuscular fiber split is not specified or simulated.' });
        add('retract-flap', 'Reflect and secure the flap', 'Reflect the mobilized tissues anteriorly over the orbital margin and secure them with scalp hooks or an equivalent retraction system. Distribute tension to keep the bony field accessible while avoiding excessive traction on the superficial tissues.', { sources: ['anatomy', 'general'], groups: surface, labels: ['Right temporalis muscle'], note: 'Scalp hooks and the skin flap are not modeled.', sections: { anatomy: 'PDF pp.7–10, Anterior flap reflection', general: 'Skin incision and soft tissue dissection' } });
        add('periosteum', 'Raise pericranium and expose bone', 'Elevate the pericranium and detach the temporalis attachment from the bone as required, preserving tissue for reconstruction. Expose the superior temporal line, frontozygomatic region and temporal squama so burr holes and the bone flap can be planned from visible landmarks.', { sources: ['anatomy', 'general'], groups: ['bone', 'muscle'], labels: ['Right temporal bone'], sections: { anatomy: 'PDF pp.11–13, Temporal fascia and temporalis mobilization', general: 'Bony landmarks' } });
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
            add('fissure', 'Open the Sylvian arachnoid corridor', 'Identify the superficial Sylvian veins and open the arachnoid interval between the frontal and temporal opercula while preserving veins and arterial branches. Develop the interval toward the insular and basal cisterns, releasing cerebrospinal fluid as appropriate to expose the middle cerebral artery and gain working space with less fixed cortical retraction.', { sources: ['sylvian'], groups: ['brain', 'arteries', 'nerves', 'bone'], labels: ['Right middle cerebral artery (m1-segment)'], change: { fissure: true }, view: { distance: 9, target: [-1.9, -.5, 1], pitch: .4 }, note: 'The corridor is highlighted on native cortex; arachnoid, small veins and exact fissure opening are not reconstructed.' });
        else
            add('cistern', 'Use the limited subfrontal corridor', 'Work through the low subfrontal opening toward the optico-carotid cistern without a routine wide Sylvian split. This concentrates exposure around the skull-base target while preserving the covered frontal and temporal surfaces.', { sources: ['no-split'], groups: cranial, labels: ['Right optic nerve (ii)'], view: clinoidView });
        if (config.clinoid === 'intra')
            add('clinoid', 'Remove the anterior clinoid intradurally', 'After dural opening, identify the optic nerve, carotid artery and their relationship to the anterior clinoid before removing the required bone under direct intradural observation. Tailor optic-canal unroofing and clinoid removal to the exposure needed while protecting the adjacent neurovascular structures.', { sources: ['intra'], groups: ['bone', 'nerves', 'arteries'], isolateBone: ['Sphenoid bone'], labels: ['Sphenoid bone'], change: { clinoid: true }, view: clinoidView, note: 'The conventional description includes Sylvian splitting; the no-split combination retains its evidence note in Papers. Bone removal is approximate.' });
        if (config.clinoid === 'none')
            add('retain', 'Keep the anterior clinoid intact', 'Leave the anterior clinoid in place when the target can be reached through the available corridor without additional optic-canal or paraclinoid exposure. This preserves the bony boundary beside the optic nerve and carotid artery.', { groups: ['bone', 'nerves', 'arteries'], isolateBone: ['Sphenoid bone'], labels: ['Sphenoid bone'], view: clinoidView });
        add('intradural-work', 'Inspect the target corridor', 'Identify the optic nerve, carotid artery and relevant arterial branches at the end of the selected approach. The lesion-specific treatment proceeds from this exposure; its maneuvers depend on the pathology and are outside this approach sequence.', { sources: ['general'], groups: cranial, labels: ['Right internal carotid artery'], sections: { general: 'Intradural corridors' }, note: 'No lesion, aneurysm clipping or tumor resection is simulated.' });
        add('dura-close', 'Reapproximate and close the dura', 'After the intradural work and hemostasis, reapproximate the dural margins and obtain a watertight closure, adding a graft if necessary. Restoring this barrier helps contain cerebrospinal fluid and separates the intradural compartment from the wound.', { sources: ['general'], groups: cranial, labels: ['Right local dura — original illustration'], change: { dura: false, fissure: false }, visualAction: 'dura-close', sections: { general: 'Closure' }, note: 'Restoration of the illustrative dural surface represents closure; suture lines and grafts are not modeled.' });
        add('bone-replace', 'Replace and secure the bone flap', 'Return the bone flap to its original position and secure it with suitable plates and screws, adding a central dural hitch if required. Stable fixation restores the cranial contour and protects the underlying dural closure.', { sources: ['general'], groups: ['bone', 'dura'], labels: ['Right temporal bone'], change: { bone: false, burr: false }, sections: { general: 'Closure' }, note: 'The flap returns along the reverse path and fixation sites are marked schematically. Any removed clinoid remains removed.' });
        add('muscle-close', 'Reattach temporalis and fascia', 'Reapproximate temporalis to its preserved attachment or reconstructive cuff, then restore the fascial layers as appropriate. Re-establishing the muscle position and tissue planes supports mastication and reduces contour deformity.', { sources: ['general', 'anatomy'], groups: surface, labels: ['Right temporalis muscle'], change: { muscle: false }, sections: { general: 'Closure', anatomy: 'PDF pp.11–12, Preserved musculo-fascial cuff for temporalis reattachment' }, note: 'Tissue displacement returns to baseline; muscle sutures are not modeled.' });
        add('drain', 'Consider a subgaleal drain', 'If persistent wound oozing suggests a risk of extracranial blood collection, consider a subgaleal drain before the soft tissues are closed. Drain use is conditional on hemostasis and the operative situation rather than an obligatory part of every exposure.', { sources: ['general', 'radiopaedia'], groups: surface, sections: { general: 'Closure discusses drainage; conditional use follows the supplied procedural outline' }, origin: 'Conditional drain use follows the supplied procedural outline; the review describes drainage without establishing a universal indication.', note: 'The drain is described, not modeled.' });
        add('galea-close', 'Close the galea', 'Reapproximate the galea with a layered closure after confirming hemostasis and the position of any drain. This restores the supportive scalp layer and reduces tension on the skin edges.', { sources: ['general', 'radiopaedia'], groups: surface, sections: { general: 'Closure context; separate galeal-layer step follows the supplied procedural outline' }, origin: 'The separate galeal closure step follows the supplied procedural outline; it is not inferred from the article discussion of galeal grafts.', note: 'Galea and suture lines are not reconstructed.' });
        add('skin-close', 'Close the skin', 'Align and close the skin edges with the selected sutures or staples. This restores the external barrier and completes the wound closure with attention to scalp perfusion and contour.', { sources: ['general', 'radiopaedia'], groups: surface, sections: { general: 'Closure' }, note: 'Skin and skin closure are not modeled.' });
        return list;
    }
    function rebuild({ reset = false } = {}) { animationToken++; const old = steps[stage]?.id; steps = buildSteps(); $('#steps').replaceChildren(); steps.forEach((s, i) => { const card = document.createElement('article'); card.className = 'step'; card.tabIndex = 0; card.innerHTML = `<div class="step-heading"><span class="step-number">${String(i + 1).padStart(2, '0')}</span><div class="step-heading-text"><h2>${escape(titleCase(s.title))}</h2></div></div><p>${escape(s.body)}</p>`; card.onclick = () => activate(i); card.onkeydown = e => { if (e.target === card && ['Enter', ' '].includes(e.key)) {
        e.preventDefault();
        activate(i);
    } }; $('#steps').append(card); }); const index = modelIndex(); $('#conflict').hidden = !(config.clinoid === 'intra' && config.fissure === 'no'); $('#conflict').textContent = 'Evidence conflict: this intradural clinoidectomy source describes wide Sylvian splitting. This selectable combination is shown for comparison, but is not validated by the cited sequence.'; renderPapers(); renderIndications(); activate(reset ? 0 : Math.max(0, steps.findIndex(s => s.id === old))); $('#planLive').textContent = steps.length + ' guide steps. Configuration updated.'; }
    function choose(c) { config = { ...c }; for (const [k, v] of Object.entries(config))
        $(`input[name="${k}"][value="${v}"]`).checked = true; rebuild(); }
    $$('.config input').forEach(i => i.onchange = e => { config[e.target.name] = e.target.value; rebuild(); });
    function syncControls() { $$('[data-layer]').forEach(e => e.checked = enabled[e.dataset.layer]); $$('[data-all-layer]').forEach(e => e.checked = allEnabled[e.dataset.allLayer]); $('#side').value = contextSide; $('#procedure').checked = procedure; $('#prelabels').checked = prelabelsOn; $('#instruments').checked = instrumentsOn; }
    const persistentLayers = new Set(['brain', 'arteries', 'nerves']);
    function applyRecommendedView({ preserveCamera = false } = {}) { const s = steps[stage]; for (const k in enabled)
        enabled[k] = persistentLayers.has(k) || s.groups.includes(k); contextAlpha = 1; procedure = true; meshes.forEach(m => m.hidden = false); if (!preserveCamera) {
        viewZoom = 1;
        regionalContext = true;
        for (const k in allEnabled)
            allEnabled[k] = false;
        ({ yaw, pitch, distance } = stepView(s));
        screenOrientation = 1;
        target = [...stepView(s).target];
        pan = [0, 0];
    } $('#presetname').textContent = titleCase(s.title); syncControls(); clearSelection(); playStepAnimation(); updateChangeControls(); render(); }
    function activate(i) { animationToken++; beforeCurrent = false; $('#compareChange').setAttribute('aria-pressed', false); $('#compareChange').textContent = 'Before'; stage = Math.max(0, Math.min(steps.length - 1, i)); opening = 1; $$('.step').forEach((c, j) => { c.classList.toggle('active', j === stage); c.setAttribute('aria-current', j === stage ? 'step' : 'false'); }); $('#stagecaption').textContent = String(stage + 1).padStart(2, '0') + ' · ' + titleCase(steps[stage].title); $('#stagecount').textContent = String(stage + 1).padStart(2, '0') + ' / ' + steps.length; $('#previous').disabled = stage === 0; $('#next').disabled = stage === steps.length - 1; applyRecommendedView({ preserveCamera: true }); updateChangeControls(); updateMobileReaderHeight(); alignActiveStep(); if (mode === '2d')
        showDiagram(); }
    $('#previous').onclick = () => { setTab('steps'); activate(stage - 1); };
    $('#next').onclick = () => { setTab('steps'); activate(stage + 1); };
    $('#recommendedView').onclick = () => applyRecommendedView();
    $$('[data-layer]').forEach(e => e.onchange = () => { enabled[e.dataset.layer] = e.checked; clearSelection(); updateChangeControls(); render(); });
    $$('[data-all-layer]').forEach(e => e.onchange = () => { allEnabled[e.dataset.allLayer] = e.checked; regionalContext = !Object.values(allEnabled).some(Boolean); clearSelection(); updateChangeControls(); render(); });
    $('#side').onchange = e => setProcedureSide(e.target.value);
    $('#procedure').onchange = e => { procedure = e.target.checked; playStepAnimation(); render(); };
    $('#prelabels').onchange = e => { prelabelsOn = e.target.checked; render(); };
    $('#instruments').onchange = e => { instrumentsOn = e.target.checked; syncControls(); updateChangeControls(); if (mode === '2d')
        showDiagram();
    else
        render(); };
    $('#operativeView').onclick = setOperativeView;
    $('#focusChange').onclick = () => applyRecommendedView();
    $('#regionalContext').onclick = toggleFullContext;
    $('#compareChange').onclick = () => { beforeCurrent = !beforeCurrent; animationToken++; opening = 1; $('#compareChange').setAttribute('aria-pressed', beforeCurrent); $('#compareChange').textContent = beforeCurrent ? 'After' : 'Before'; if (mode === '2d')
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
    function visible(m) { if (m.hidden)
        return false; const side = meshSide(m); if (contextSide !== 'both' && side !== 'both' && side !== contextSide)
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
    } return !!allEnabled[m.group] || !!enabled[m.group] && relevantMesh(m); }
    function opacity(m) { return 1; }
    function surgicalState(m) { let cut = 0, motion = 0, action = 1, windowAction = 1, burrAction = 0; if (!procedure || m.marker || m.instrument || !onOperativeSide(m))
        return { cut, motion, action, windowAction, burrAction }; const s = steps[stage], state = { ...(beforeCurrent ? s.beforeState : s.state) }; const a = id => { if (beforeCurrent || s.change !== id)
        return 1; return s.state[id] ? opening : 1 - opening; }; if (!beforeCurrent && s.change && opening < 1)
        state[s.change] = s.beforeState[s.change] || s.state[s.change]; if (m.group === 'bone') {
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
    } if (m.group === 'dura' && state.dura) {
        cut = config.fissure === 'no' ? 5 : 3;
        action = a('dura');
    } if (!m.contextOnly && (['fascia', 'deepfascia'].includes(m.group) || m.group === 'fat' && !baseId(m).includes('subfat')) && state.muscle) {
        motion = config.muscle === 'subfascial' ? 1 : 2;
        action = a('muscle');
    } if ((m.nativeName === 'Temporalis muscle.' + (operativeSide() === 'left' ? 'l' : 'r') || baseId(m) === 'ILL-subfat') && state.muscle) {
        motion = 2;
        action = a('muscle');
    } return { cut, motion, action, windowAction, burrAction }; }
    function transform(p) { p = p.map((x, i) => x - target[i]); const c = Math.cos(yaw), s = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch), x = c * p[0] + s * p[2], z = -s * p[0] + c * p[2]; return [screenOrientation * x, screenOrientation * (cp * p[1] - sp * z), sp * p[1] + cp * z]; }
    function project(p) { const t = transform(p), w = distance - t[2]; return [(t[0] + pan[0]) * scale * viewZoom / aspect / w * host.clientWidth / 2 + host.clientWidth / 2, host.clientHeight / 2 - (t[1] + pan[1]) * scale * viewZoom / w * host.clientHeight / 2]; }
    function reflectionOffset(motion, progress) {
        if (!motion)
            return [0, 0, 0];
        // Detach laterally before sweeping across the temporal squama and zygomatic arch.
        // Closure retraces the same path by decreasing progress.
        const lift = Math.min(1, progress / .4);
        const sweep = Math.max(0, (progress - .4) / .6);
        const smooth = t => t * t * (3 - 2 * t);
        const outward = 1.12 * smooth(lift);
        const travel = smooth(sweep);
        return motion === 1 ? [-outward * sideSign(), 0, 1.6 * travel] :
            [-outward * sideSign(), -1.5 * travel, -.4 * travel];
    }
    function meshOffset(m) { if ((m.instrument || m.marker) && m.follow) {
        const follow = meshes.find(x => x.id === m.follow || x.nativeName === m.follow);
        return follow ? meshOffset(follow) : [0, 0, 0];
    } const s = surgicalState(m); const d = reflectionOffset(s.motion, s.action); if (separateLayers) {
        const n = ({ 'ILL-subfat': 1, 'ILL-deepfascia': 2, 'ILL-interfat': 3, 'ILL-superfascia': 4, 'ILL-suprafat': 5 })[baseId(m)] || 0;
        d[0] += (meshSide(m) === 'left' ? 1 : -1) * n * .35;
    } return d; }
    function labelPoint(m) { const p = m.labelAnchor || m.anchor, d = meshOffset(m); return p.map((x, i) => x + d[i]); }
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
    } if (m.instrument) {
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
        } points.sort((a, b) => a.d - b.d); m.procedureCandidates = { key, points: points.map(x => x.p) }; return m.procedureCandidates.points; }
    function regionSurfaceAnchor(m) { const offset = meshOffset(m), points = candidatesForLabel(m), spec = procedureLabelRegion(m); for (const p of points) {
        if (!pointRetained(m, p))
            continue;
        const world = p.map((x, i) => x + offset[i]), q = project(world), x = Math.floor(q[0] / host.clientWidth * pickWidth), y = Math.floor((host.clientHeight - q[1]) / host.clientHeight * pickHeight);
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
    } const q = project(spec.target.map((x, i) => x + offset[i])); return { x: q[0], y: q[1], visible: false }; }
    function activeLabelMesh() { if (selected >= 0)
        return visible(meshes[selected]) ? meshes[selected] : null; if (!prelabelsOn)
        return null; const s = steps[stage]; if (s.id === 'orient')
        return meshes.find(m => m.marker && baseId(m) === 'MARK-pterion' && visible(m)); if (s.id === 'incision')
        return meshes.find(m => m.marker && baseId(m) === 'MARK-incision' && visible(m)); if (s.id === 'burr-holes')
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
    } popup.hidden = false; const a = regionSurfaceAnchor(m), r = fixedLabelRect(popup.offsetWidth || 190, popup.offsetHeight || 75); popup.style.left = r.x + 'px'; popup.style.top = r.y + 'px'; $('#labelStatus').textContent = a.visible ? '' : procedureLabelRegion(m).cranial ? 'Cranial segment hidden in this view' : 'Region hidden in this view'; if (a.visible) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', a.x);
        line.setAttribute('y1', a.y);
        line.setAttribute('x2', Math.max(r.x, Math.min(r.x + r.w, a.x)));
        line.setAttribute('y2', Math.max(r.y, Math.min(r.y + r.h, a.y)));
        svg.append(line);
    } }
    function setOperativeView() {
        const step = steps[stage];
        const sign = sideSign();
        const narrowOpening = config.fissure === 'no' && step.state.dura;
        const basal = narrowOpening || ['clinoid', 'cistern', 'intradural-work'].includes(step.id);
        // Right-side atlas landmarks in the window and toward the sphenoid/optico-carotid corridor.
        // Rodriguez Rubio et al., Figs. 10C–D and 11, inform the view direction and image orientation.
        const entry = narrowOpening ? [-2.9, -.62, 1.65] :
            basal ? [-3.05, .05, 1.85] : [-3.05, .30, 1.25];
        const focus = basal ? [-.85, -1.05, 1.35] : [-1.55, -.3, .60];
        const [dx, dy, dz] = entry.map((value, i) => value - focus[i]);
        yaw = sign * Math.atan2(-dx, dz);
        pitch = Math.atan2(dy, Math.hypot(dx, dz));
        // The operative illustrations show the skull-base side at the top of the field.
        screenOrientation = -1;
        distance = basal ? 9 : 10.5;
        target = [entry[0] * sign, entry[1], entry[2]];
        viewZoom = 1;
        pan = [0, 0];
        render();
    }
    function setView(v) { const views = { anterior: [0, 0], right: [Math.PI / 2, 0], left: [-Math.PI / 2, 0], superior: [0, 1.5] }; [yaw, pitch] = views[v]; screenOrientation = 1; pan = [0, 0]; render(); }
    $$('[data-view]').forEach(b => b.onclick = () => setView(b.dataset.view));
    $('#resetView').onclick = () => applyRecommendedView();
    $('#manifestDownload').onclick = () => downloadBlob(new Blob([JSON.stringify(manifest, null, 2)], { type: 'application/json' }), 'pterional-mesh-manifest.json');
    $('#exportPlan').onclick = () => downloadBlob(new Blob([JSON.stringify({ model: modelIndex() + 1, config, steps, papers, side: operativeSide(), contextSide, limitations: manifest.limitations }, null, 2)], { type: 'application/json' }), 'pterional-model-' + String(modelIndex() + 1).padStart(2, '0') + '-reading-plan.json');
    $('#pdfRequests').onclick = () => downloadBlob(new Blob([papers.map(p => `${p.title}\n${p.citation}\n${p.url}\n`).join('\n')], { type: 'text/plain' }), 'pterional-requested-papers.txt');
    mobileQuery.addEventListener?.('change', syncMobileLayout);
    window.addEventListener?.('resize', () => { if (isMobile)
        requestAnimationFrame(updateMobileReaderHeight); });
    syncMobileLayout();
    new ResizeObserver(() => { if (isMobile)
        updateMobileReaderHeight(); }).observe($('.layout'));
    function ensureGeometry(m) { if (m.geometryReady)
        return; const bytes = decode64(m.data), floats = new Float32Array(bytes.buffer); if (floats.length !== m.count * 6)
        throw Error('Invalid geometry buffer ' + m.id); m.vertexData = floats; gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferSubData(gl.ARRAY_BUFFER, m.first * 24, floats); m.geometryReady = true; delete m.data; }
    function shader(type, text) { const s = gl.createShader(type); gl.shaderSource(s, text); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
        throw Error(gl.getShaderInfoLog(s)); return s; }
    function initGL() {
        gl = canvas.getContext('webgl', { antialias: true, alpha: false, preserveDrawingBuffer: true });
        if (!gl)
            throw Error('WebGL is unavailable');
        program = gl.createProgram();
        gl.attachShader(program, shader(gl.VERTEX_SHADER, `attribute vec3 aPosition;attribute vec3 aNormal;uniform float uYaw,uPitch,uScreenOrientation,uDistance,uAspect,uScale,uMotion,uSide;uniform mediump float uAction;uniform vec2 uPan;uniform vec3 uTarget;uniform vec3 uOffset;uniform float uFlapAngle;varying mediump vec3 vNormal;varying mediump vec3 vOriginal;void main(){vOriginal=vec3(aPosition.x*uSide,aPosition.y,aPosition.z);vec3 q=aPosition+uOffset;vec3 norm=aNormal;
if(uFlapAngle!=0.){
  float ct=cos(uFlapAngle),st=sin(uFlapAngle);
  mat3 tilt=mat3(ct,st,0.,-st,ct,0.,0.,0.,1.);
  vec3 edge=vec3(-2.4*uSide,-1.25,1.25);
  q=edge+tilt*(aPosition-edge)+uOffset;
  norm=tilt*aNormal;
}
float c=cos(uYaw),s=sin(uYaw),cp=cos(uPitch),sp=sin(uPitch);mat3 ry=mat3(c,0.,-s,0.,1.,0.,s,0.,c);mat3 rx=mat3(1.,0.,0.,0.,cp,sp,0.,-sp,cp);vec3 p=rx*ry*(q-uTarget);vNormal=rx*ry*norm;p.xy*=uScreenOrientation;vNormal.xy*=uScreenOrientation;float w=uDistance-p.z;gl_Position=vec4((p.x+uPan.x)*uScale/uAspect,(p.y+uPan.y)*uScale,1.0125786*w-1.0062893,w);}`));
        gl.attachShader(program, shader(gl.FRAGMENT_SHADER, `precision mediump float;uniform vec4 uColor;uniform float uSelected,uPick,uRegion,uCut,uChange,uWindowAction,uBurrs,uIsMarker,uMarkKind,uMarkAxis,uMarkReveal;uniform vec3 uMarkCenter;uniform vec2 uMarkRange;uniform mediump float uAction;varying mediump vec3 vNormal;varying mediump vec3 vOriginal;void main(){if(uIsMarker>.5){if(uMarkReveal<=.001)discard;float t=0.;if(uMarkKind==1.)t=(atan(vOriginal.z-uMarkCenter.z,vOriginal.y-uMarkCenter.y)+3.14159265)/6.2831853;else if(uMarkKind==2.){float v=uMarkAxis==2.?vOriginal.z:uMarkAxis==0.?vOriginal.x:vOriginal.y;t=(v-uMarkRange.x)/max(.001,uMarkRange.y-uMarkRange.x);}if(t>uMarkReveal)discard;}vec2 q=vec2((vOriginal.y-.3)/1.55,(vOriginal.z-1.25)/1.65);bool window=dot(q,q)<uWindowAction*uWindowAction&&vOriginal.x< -1.9;vec3 cq=(vOriginal-vec3(-.8,-1.0,1.25))/vec3(.46,.55,.67);bool clinoid=dot(cq,cq)<uAction*uAction;vec2 dq=vec2((vOriginal.y-.3)/1.08,(vOriginal.z-1.25)/1.08);vec2 lq=vec2((vOriginal.y+.62)/.38,(vOriginal.z-1.65)/.55);if((uCut==1.||uCut==4.)&&window)discard;if(uCut==7.&&!window)discard;if((uCut==4.||uCut==6.)&&clinoid)discard;if(uCut==3.&&dot(dq,dq)<uAction*uAction)discard;if(uCut==5.&&dot(lq,lq)<uAction*uAction)discard;if(uBurrs>0.&&vOriginal.x< -1.9){float a=min(1.,uBurrs*2.)*.255,b=max(0.,uBurrs*2.-1.)*.255;if(distance(vOriginal.yz,vec2(-.7,2.45))<a||distance(vOriginal.yz,vec2(-1.,.8))<b)discard;}if(uPick>.5){gl_FragColor=uColor;return;}vec3 n=normalize(vNormal);float light=.30+.55*abs(dot(n,normalize(vec3(-.5,.7,1.))))+.12*abs(dot(n,normalize(vec3(.9,-.2,.3))));vec3 c=uColor.rgb*light;float edge=10.;if(uCut==1.||uCut==4.)edge=abs(length(q)-uWindowAction);if(uCut==3.)edge=abs(length(dq)-uAction);if(uCut==5.)edge=abs(length(lq)-uAction);if(uCut==4.||uCut==6.)edge=min(edge,abs(length(cq)-uAction));if(edge<.025)c*=.55;float highlight=0.;if(uChange==1.&&vOriginal.x< -1.9&&length(q)<1.16)highlight=.65;if(uChange==2.&&length(cq)<1.65)highlight=.85;if(uChange==3.&&length(dq)<1.16)highlight=.65;if(uChange==4.&&length(lq)<1.2)highlight=.65;if(uChange==5.)highlight=.35;if(uChange==7.&&vOriginal.x< -1.9&&(distance(vOriginal.yz,vec2(-.7,2.45))<.4||distance(vOriginal.yz,vec2(-1.,.8))<.4))highlight=.8;if(uChange==6.&&distance(vOriginal,vec3(-2.1,-.5,1.0))<1.2)highlight=.5;c=mix(c,vec3(1.,.65,.17)*(.6+.4*light),highlight);if(uSelected>.5)c=mix(c,vec3(.66,1.,.62),.50);if(uIsMarker>.5)c=mix(c,uColor.rgb,.8);gl_FragColor=vec4(c,uColor.a);}`));
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS))
            throw Error(gl.getProgramInfoLog(program));
        gl.useProgram(program);
        ['Yaw', 'Pitch', 'ScreenOrientation', 'Distance', 'Aspect', 'Scale', 'Pan', 'Color', 'Selected', 'Pick', 'Region', 'Target', 'Cut', 'Motion', 'Action', 'Offset', 'Change', 'WindowAction', 'FlapAngle', 'Side', 'Burrs', 'IsMarker', 'MarkKind', 'MarkAxis', 'MarkReveal', 'MarkCenter', 'MarkRange'].forEach(n => uniforms[n] = gl.getUniformLocation(program, 'u' + n));
        let vertexCount = 0;
        meshes.forEach((m, i) => { m.index = i; m.hidden = false; m.first = vertexCount; vertexCount += m.count; m.rgb = m.color.match(/\w\w/g).map(x => parseInt(x, 16) / 255); });
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
        window.guideQA = { solidDepth: true, zoomBy, syncMobileLayout, alignActiveStep, updateMobileReaderHeight, changeKind, procedureLabelRegion, activeLabelMesh, fixedLabelRect, visible, indicationGroups, chooseIndication, setProcedureSide, operativeSide, sideSign, canonicalPoint, onOperativeSide, relevantMesh, stepView, baseId, meshSide, meshOffset, scanPickAnchors, pickIdAt, meshCount: meshes.length, vertexCount, draw, activate, choose, configs, select, setTab, setView, enabled, allEnabled, getMeshes: () => meshes.map(({ data, vertexData, procedureCandidates, ...m }) => m), visibleCount: () => meshes.filter(visible).length, getSteps: () => steps, getState: () => ({ viewZoom, isMobile, config: { ...config }, model: modelIndex() + 1, stage, mode, selected, contextAlpha, contextSide, prelabelsOn, opening, beforeCurrent, regionalContext, instrumentsOn, separateLayers, yaw, pitch, screenOrientation, distance, pan: [...pan], target: [...target], enabled: { ...enabled }, allEnabled: { ...allEnabled } }), surgicalState, refreshLabels: updateStepLabels };
        $('#loading').hidden = true;
        render();
    }
    let pickTarget = null;
    function preparePickTarget(w, h) { if (pickTarget && pickTarget.w === w && pickTarget.h === h)
        return; if (pickTarget) {
        gl.deleteFramebuffer(pickTarget.fbo);
        gl.deleteTexture(pickTarget.texture);
        gl.deleteRenderbuffer(pickTarget.depth);
    } const fbo = gl.createFramebuffer(), texture = gl.createTexture(), depth = gl.createRenderbuffer(); gl.bindTexture(gl.TEXTURE_2D, texture); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null); gl.bindRenderbuffer(gl.RENDERBUFFER, depth); gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0); gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth); if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
        throw Error('Selection buffer is unavailable'); pickTarget = { fbo, texture, depth, w, h }; gl.bindTexture(gl.TEXTURE_2D, null); gl.bindRenderbuffer(gl.RENDERBUFFER, null); }
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
    function draw(picking = false) {
        if (!available)
            return;
        const ratio = picking ? 1 : Math.min(devicePixelRatio || 1, 2), w = Math.max(1, Math.round(host.clientWidth * ratio)), h = Math.max(1, Math.round(host.clientHeight * ratio));
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
        gl.uniform1f(uniforms.Yaw, yaw);
        gl.uniform1f(uniforms.Pitch, pitch);
        gl.uniform1f(uniforms.ScreenOrientation, screenOrientation);
        gl.uniform1f(uniforms.Distance, distance);
        gl.uniform1f(uniforms.Aspect, aspect);
        gl.uniform1f(uniforms.Scale, scale * viewZoom);
        gl.uniform2fv(uniforms.Pan, pan);
        gl.uniform3fv(uniforms.Target, target);
        gl.uniform1f(uniforms.Pick, picking ? 1 : 0);
        let list = meshes.filter(visible);
        gl.disable(gl.BLEND);
        gl.depthFunc(gl.LEQUAL);
        gl.depthMask(true);
        for (const m of list) {
            ensureGeometry(m);
            let rgb = m.rgb;
            if (picking) {
                const id = m.index + 1;
                rgb = [(id & 255) / 255, ((id >> 8) & 255) / 255, ((id >> 16) & 255) / 255];
            }
            const state = surgicalState(m);
            gl.uniform1f(uniforms.Side, sideSign());
            gl.uniform1f(uniforms.Burrs, state.burrAction);
            gl.uniform1f(uniforms.IsMarker, m.marker ? 1 : 0);
            gl.uniform1f(uniforms.MarkKind, m.markerKind || 0);
            gl.uniform1f(uniforms.MarkAxis, m.markerAxis || 1);
            gl.uniform1f(uniforms.MarkReveal, m.marker ? (beforeCurrent ? 0 : opening) : 1);
            gl.uniform3fv(uniforms.MarkCenter, canonicalPoint(m.markerCenter || [0, 0, 0]));
            gl.uniform2fv(uniforms.MarkRange, m.markerRange || [0, 1]);
            gl.uniform1f(uniforms.WindowAction, state.windowAction);
            gl.uniform1f(uniforms.Change, changeKind(m));
            gl.uniform3fv(uniforms.Offset, meshOffset(m));
            gl.uniform1f(uniforms.Cut, state.cut);
            gl.uniform1f(uniforms.Motion, state.motion);
            gl.uniform1f(uniforms.Action, state.action);
            gl.depthMask(true);
            gl.uniform4fv(uniforms.Color, [...rgb, picking ? 1 : opacity(m)]);
            gl.uniform1f(uniforms.Selected, m.index === selected ? 1 : 0);
            gl.uniform1f(uniforms.Region, procedure && m.name === steps[stage].focus ? 1 : 0);
            gl.uniform1f(uniforms.FlapAngle, 0);
            gl.drawArrays(gl.TRIANGLES, m.first, m.count);
            const flap = boneFlapTravel(state);
            if (flap) {
                gl.uniform1f(uniforms.Cut, 7);
                gl.uniform1f(uniforms.FlapAngle, flap.angle);
                gl.uniform3fv(uniforms.Offset, flap.offset);
                gl.uniform1f(uniforms.Change, 0);
                gl.drawArrays(gl.TRIANGLES, m.first, m.count);
            }
        }
        gl.depthMask(true);
        if (!picking && !capturing) {
            updateStepLabels();
        }
    }
    function render() { if (scheduled)
        return; scheduled = true; requestAnimationFrame(() => { scheduled = false; if (mode === '3d') {
        draw(true);
        cacheLabelVisibility();
        draw();
    } }); }
    let pointers = new Map(), last = null, moved = 0, pinch = 0, hadMulti = false, pinchCenter = null;
    canvas.onpointerdown = e => { canvas.setPointerCapture(e.pointerId); if (pointers.size === 0)
        hadMulti = false; pointers.set(e.pointerId, [e.clientX, e.clientY]); if (pointers.size > 1)
        hadMulti = true; last = [e.clientX, e.clientY]; moved = 0; if (pointers.size === 2) {
        const p = [...pointers.values()];
        pinch = Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]);
        pinchCenter = [(p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2];
    } };
    canvas.onpointermove = e => { if (!pointers.has(e.pointerId))
        return; if (!last) {
        last = [e.clientX, e.clientY];
        return;
    } const dx = e.clientX - last[0], dy = e.clientY - last[1]; moved += Math.abs(dx) + Math.abs(dy); pointers.set(e.pointerId, [e.clientX, e.clientY]); if (pointers.size === 2) {
        const p = [...pointers.values()], d = Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]);
        if (pinch > 0 && d > 0)
            zoomBy(d / pinch);
        const center = [(p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2], unit = 2 * distance / (scale * viewZoom * Math.max(1, host.clientHeight));
        if (pinchCenter) {
            pan[0] += (center[0] - pinchCenter[0]) * unit;
            pan[1] -= (center[1] - pinchCenter[1]) * unit;
        }
        pinchCenter = center;
        pinch = d;
    }
    else if (e.shiftKey || e.buttons === 2) {
        const unit = 2 * distance / (scale * viewZoom * Math.max(1, host.clientHeight));
        pan[0] += dx * unit;
        pan[1] -= dy * unit;
    }
    else {
        yaw += screenOrientation * dx * .008;
        pitch = Math.max(-1.55, Math.min(1.55, pitch + screenOrientation * dy * .008));
    } last = [e.clientX, e.clientY]; render(); };
    canvas.onpointerup = e => { pointers.delete(e.pointerId); if (moved < 5 && available && !hadMulti) {
        draw(true);
        cacheLabelVisibility();
        const r = canvas.getBoundingClientRect(), id = pickIdAt(e.clientX - r.left, e.clientY - r.top);
        select(id);
        render();
    } last = null; };
    canvas.onpointercancel = e => { pointers.delete(e.pointerId); last = null; };
    canvas.oncontextmenu = e => e.preventDefault();
    canvas.addEventListener('wheel', e => { e.preventDefault(); const delta = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? host.clientHeight : 1); zoomBy(Math.exp(-delta * .0015)); }, { passive: false });
    canvas.onkeydown = e => { if (e.key === 'Escape') {
        clearSelection();
        return;
    } if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '-', '=', 'Home'].includes(e.key)) {
        e.preventDefault();
        if (e.key === 'ArrowLeft')
            yaw -= screenOrientation * .13;
        if (e.key === 'ArrowRight')
            yaw += screenOrientation * .13;
        if (e.key === 'ArrowUp')
            pitch = Math.max(-1.55, Math.min(1.55, pitch + screenOrientation * .1));
        if (e.key === 'ArrowDown')
            pitch = Math.max(-1.55, Math.min(1.55, pitch - screenOrientation * .1));
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
