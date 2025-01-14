function loadFormData() {
    const savedData = JSON.parse(localStorage.getItem('formData'));
    const expiryTime = localStorage.getItem('formExpiry');

    if (savedData && expiryTime && Date.now() < expiryTime) {
        const form = document.getElementById('generateForm');
        for (const [key, value] of Object.entries(savedData)) {
            const field = form.elements[key];
            if (field && ['TEXTAREA', 'SELECT', 'INPUT'].includes(field.tagName)) {
                if (field.tagName === 'SELECT') {
                    // Set the selected option for <select>
                    Array.from(field.options).forEach(option => {
                        option.selected = option.value === value;
                    });
                } else {
                    // Set the value for <textarea> and <input>
                    field.value = value;
                }
            }
        }
    } else {
        // Clear expired data
        localStorage.removeItem('formData');
        localStorage.removeItem('formExpiry');
    }
}

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
    Object.entries(data).forEach(([modelName, modelData]) => {
        const option = document.createElement('option');
        option.value = modelData.path;
        option.dataset.cfg = modelData.cfg || 7;
        option.dataset.type = modelData.type || "SDXL";
        option.textContent = modelName;
        if (modelData.disabled) {
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

function populateThemes(data, select) {
    Object.entries(data).forEach(([themeName, themeTones]) => {
        const option = document.createElement('option');
        // Convert the theme tones object to a JSON string for the option value
        option.value = JSON.stringify(themeTones);
        option.textContent = themeName;
        select.appendChild(option);
    });

    loadTheme();
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
loadJsonAndPopulateSelect('/static/json/themes.json', 'theme_select', populateThemes);
