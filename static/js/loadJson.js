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
        option.value = modelUrl[0];
        option.dataset.cfg = modelUrl[1] || 7;
        option.textContent = modelName;
        if (modelUrl[2] === "disabled") {
            option.disabled = true;
        }
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

function populateExampleSizes(data, select) {
    Object.entries(data).forEach(([sizeName, sizeDimensions]) => {
        const option = document.createElement('option');
        option.value = sizeDimensions.join('x'); // Format as "width x height"
        option.textContent = sizeName; // Display the size name
        select.appendChild(option);
    });
}

function populateSchedulers(data, select) {
    data.schedulers.forEach(scheduler => {
        const option = document.createElement('option');
        option.value = scheduler;
        option.textContent = scheduler;
        select.appendChild(option);
    });
    loadFormData();
}

// Load and populate both selects
loadJsonAndPopulateSelect('/static/json/models.json', 'model', populateModels);
loadJsonAndPopulateSelect('/static/json/examplePrompts.json', 'example_prompt', populateExamplePrompts);
loadJsonAndPopulateSelect('/static/json/dimensions.json', 'example_size', populateExampleSizes);
loadJsonAndPopulateSelect('/static/json/schedulers.json', 'scheduler', populateSchedulers);
