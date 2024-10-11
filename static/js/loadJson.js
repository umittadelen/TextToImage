// ------------------- MODELS ------------------------

fetch('/static/json/models.json')
.then(response => {
    if (!response.ok) {
        throw new Error('Error loading models.json');
    }
    return response.json();
})
.then(data => {
    const select = document.getElementById('model');

    // Iterate over the items in models.json and create option elements
    for (const [modelName, modelUrl] of Object.entries(data)) {
        const option = document.createElement('option');
        option.value = modelUrl;
        option.textContent = modelName;
        select.appendChild(option);
    }
})
.catch(error => {
    console.error('Error:', error);
});

// ---------------- EXAMPLE PROMPTS --------------------

fetch('/static/json/examplePrompts.json')
.then(response => {
    if (!response.ok) {
        throw new Error('Error loading examplePrompts.json');
    }
    return response.json();
})
.then(data => {
    const select = document.getElementById('example_prompt');

    // Iterate over the examples and create option elements
    data.examples.forEach(prompt => {
        const option = document.createElement('option');
        option.value = prompt;
        option.textContent = prompt;
        select.appendChild(option);
    });
})
.catch(error => {
    console.error('Error:', error);
});