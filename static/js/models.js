// Function to handle form submission
function submitModel(event) {
    event.preventDefault(); // Prevent default form submission

    const modelName = document.getElementById('model-name').value;
    alert(`Model submitted: ${modelName}`);

    // Use event.target to get the form element
    const formData = new FormData(event.target);
    fetch('/addmodel', {
        method: 'POST',
        body: formData
    })
    .then((response) => response.json())
    .then((data) => {
        console.log('Success:', data);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}


// Load JSON file into the editor
function loadJson() {
    fetch('/static/json/models.json')
        .then(response => {
            if (!response.ok) {
                throw new Error('Error loading JSON file');
            }
            return response.json();
        })
        .then(data => {
            document.getElementById('json-editor').value = JSON.stringify(data, null, 4);
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('json-status').textContent = `Error loading JSON: ${error.message}`;
        });
}

// Save the JSON content
function saveJson() {
    const jsonContent = document.getElementById('json-editor').value;

    fetch('/changejson', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: jsonContent // Send raw JSON string
    })
    .then((response) => {
        if (!response.ok) {
            throw new Error('Failed to save JSON');
        }
        return response.json();
    })
    .then((data) => {
        console.log('Success:', data);
        document.getElementById('json-status').textContent = 'JSON saved successfully!';
    })
    .catch((error) => {
        console.error('Error:', error);
        document.getElementById('json-status').textContent = `Error saving JSON: ${error.message}`;
    });
}


// Validate JSON input
function validateJson() {
    const jsonContent = document.getElementById('json-editor').value;
    try {
        JSON.parse(jsonContent);
        document.getElementById('json-status').textContent = 'Valid JSON';
    } catch {
        document.getElementById('json-status').textContent = 'Invalid JSON';
    }
}

// Load JSON when the page loads
document.addEventListener('DOMContentLoaded', loadJson);