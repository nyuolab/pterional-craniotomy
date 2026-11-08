(async () => {
    'use strict';
    const files = ["assets/anatomy-01.bin", "assets/anatomy-02.bin", "assets/anatomy-03.bin", "assets/anatomy-04.bin", "assets/anatomy-05.bin", "assets/anatomy-06.bin"], message = document.getElementById('anatomy-loading-message');
    try {
        if (typeof DecompressionStream === 'undefined')
            throw new Error('Please open this guide in a current version of Chrome, Edge, Firefox or Safari.');
        let completed = 0;
        const stream = new ReadableStream({ async start(controller) { try {
                for (const file of files) {
                    const response = await fetch(file);
                    if (!response.ok)
                        throw new Error('Anatomy data could not be loaded (' + response.status + ').');
                    const reader = response.body.getReader();
                    while (true) {
                        const { value, done } = await reader.read();
                        if (done)
                            break;
                        controller.enqueue(value);
                    }
                    completed++;
                    message.textContent = 'Loading Anatomy… ' + Math.round(completed / files.length * 100) + '%';
                }
                controller.close();
            }
            catch (error) {
                controller.error(error);
            } } });
        const [manifest, meshes] = await Promise.all([
            fetch('data/manifest.json').then(response => {
                if (!response.ok)
                    throw new Error('The anatomy manifest could not be loaded.');
                return response.json();
            }),
            new Response(stream.pipeThrough(new DecompressionStream('gzip'))).json()
        ]);
        if (!Array.isArray(meshes) || meshes.length !== manifest.structure_count)
            throw new Error('Anatomy data is incomplete. Please refresh the page.');
        window.__PTERIONAL_DATA__ = { manifest, meshes };
        const script = document.createElement('script');
        script.src = 'assets/app.js';
        script.onload = () => document.getElementById('anatomy-loading')?.remove();
        script.onerror = () => { message.textContent = 'The guide could not be loaded. Please refresh the page.'; };
        document.body.append(script);
    }
    catch (error) {
        message.textContent = error.message || 'Anatomy could not be loaded. Please refresh the page.';
        console.error(error);
    }
})();
