// Reusable function to load JSON and populate a select element
function loadJsonAndPopulateSelect(location, selectId, dataHandler) {
    fetch(location)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error loading ${location}`);
            }
            return response.json();
        })
        .then(data => dataHandler(data, document.getElementById(selectId)))
        .catch(error => console.error('Error:', error));
}

// Handler to populate the model select element
function populateModels(data, select) {
    Object.entries(data).forEach(([modelName, modelUrl]) => {
        const option = document.createElement('option');
        option.value = modelUrl;
        option.textContent = modelName;
        select.appendChild(option);
    });
}

// Handler to populate the example prompts select element
function populateExamplePrompts(data, select) {
    data.examples.forEach(prompt => {
        const option = document.createElement('option');
        option.value = prompt;
        option.textContent = prompt;
        select.appendChild(option);
    });
}

// Determine which prompts file to load based on the URL
const promptsFile = window.location.pathname.includes('/hidden') 
    ? '/static/json/prompts-hidden.json' 
    : '/static/json/examplePrompts.json';

// Load and populate both selects
loadJsonAndPopulateSelect('/static/json/models.json', 'model', populateModels);
loadJsonAndPopulateSelect(promptsFile, 'example_prompt', populateExamplePrompts);
